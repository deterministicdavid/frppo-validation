import os
import argparse
import glob
import yaml
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import FRPPO

from run import evaluate_model


def run_evaluation_and_process(config_file_list: list, eval_episodes: int):
    results = []
    
    print(f"{'Environment':<20} | {'Model File':<50} | {'Mean Reward':<12}")
    print("-" * 90)

    common_env_name = None
    
    for conf_path in config_file_list:
        if not os.path.exists(conf_path):
            print(f"Warning: Config file {conf_path} not found. Skipping.")
            continue

        # Load the configuration
        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract necessary parameters
        env_name = config.get("env_name", "Unknown")
        if common_env_name is None:
            common_env_name = env_name
        elif env_name.casefold() != common_env_name.casefold():
            raise ValueError(f"Trying to compare rewards of {common_env_name} and {env_name} which makes no sense.")

        # Safe access to nested keys
        train_config = config.get("train", {})
        log_config = config.get("logging", {})
        
        algo_name = train_config.get("algo", "Unknown")
        run_id = train_config.get("run_id", "Unknown")
        log_dir = log_config.get("log_dir", f"./logs_{env_name}")
        name_prefix = log_config.get("name_prefix", "")

        # Check if log_dir exists
        if not os.path.exists(log_dir):
            print(f"Skipping {env_name} ({conf_path}): Log dir {log_dir} not found.")
            continue

        # 2. Find matching .zip files
        # We use the name_prefix from the config to filter specifically for these runs
        search_pattern = os.path.join(log_dir, f"{name_prefix}*.zip")
        zip_files = glob.glob(search_pattern)
        zip_files.sort()

        if not zip_files:
            print(f"Skipping {env_name}: No models found in {log_dir} matching prefix '{name_prefix}'")
            continue

        # 3. Iterate over found models
        for model_path in zip_files:
            file_name = os.path.basename(model_path)
            
            try:
                # Load the model (without env, evaluate_model handles env creation)
                if algo_name == "PPO":
                    model = PPO.load(model_path)
                elif algo_name == "FRPPO":
                    model = FRPPO.load(model_path)
                else:
                    print(f"Unknown algorithm '{algo_name}' in {conf_path}. Skipping.")
                    break # Skip other files for this invalid config

                # Note: evaluate_model inside run.py will handle loading stats if appropriate i.e if the env is mujoco
                stats_path = os.path.join(os.path.dirname(model_path), f"{os.path.basename(model_path).replace('.zip', '')}_env.pkl")
                mean_reward = evaluate_model(
                    model=model, 
                    config=config, 
                    num_eval_runs=eval_episodes,
                    stats_path=stats_path,
                )

                print(f"{env_name:<20} | {file_name:<50} | {mean_reward:>10.2f}")
                
                results.append({
                    "config_file": conf_path,
                    "env": env_name,
                    "file": file_name,
                    "mean_reward": mean_reward,
                    "algo": algo_name,
                    "run_id": run_id,
                    "full_path": model_path
                })
                
                del model

            except Exception as e:
                print(f"{env_name:<20} | {file_name:<50} | ERROR: {e}")

    if len(results)==0:
        print("No results generated.")
        exit(1)


    rewards = [r["mean_reward"] for r in results]
    global_max = np.max(rewards)
    global_min = np.min(rewards)
    
    print("-" * 100)
    print(f"Global Max Reward: {global_max:.2f}")
    print(f"Global Min Reward: {global_min:.2f}")
    
    # Avoid division by zero if all rewards are identical
    denom = global_max - global_min
    if denom == 0:
        denom = 1.0

    for res in results:
        res["norm_reward"] = (res["mean_reward"] - global_min) / denom

    # Create DataFrame
    df = pd.DataFrame(results)

    # --- OUTPUT 1: Detailed Table ---
    print("\n" + "="*120)
    print(f"DETAILED RESULTS (Max: {global_max:.2f}, Min: {global_min:.2f})")
    print("="*120)
    print(f"{'Algo':<10} | {'Run ID':<10} | {'Environment':<20} | {'Mean Reward':<12} | {'Norm Reward':<12} | {'File'}")
    print("-" * 120)

    # Sort for detailed view
    df_sorted = df.sort_values(by=['algo', 'run_id', 'file'])
    
    for _, row in df_sorted.iterrows():
        print(f"{row['algo']:<10} | {row['run_id']:<10} | {row['env']:<20} | "
              f"{row['mean_reward']:>11.2f}  | {row['norm_reward']:>11.4f}  | {row['file']}")

    # --- OUTPUT 2: Aggregated Summary ---
    print("\n" + "="*80)
    print("SUMMARY: AVERAGE NORMALIZED REWARD BY ALGO & RUN ID")
    print("="*80)
    print(f"{'Algo':<10} | {'Run ID':<20} | {'Avg Norm Reward':<15} | {'Count':<5}")
    print("-" * 80)

    # Group by Algo and Run ID and calculate mean of normalized reward
    summary = df.groupby(['algo', 'run_id'])['norm_reward'].agg(['mean', 'count']).reset_index()
    
    # Sort specifically to group PPO then FRPPO if desired, or just alphabetical
    summary = summary.sort_values(by=['algo', 'run_id'])

    for _, row in summary.iterrows():
        print(f"{row['algo']:<10} | {row['run_id']:<20} | {row['mean']:>15.4f} | {row['count']:<5}")
    
    print("="*80)

    # 4. Save detailed CSV
    cols = ['algo', 'run_id', 'env', 'mean_reward', 'norm_reward', 'file', 'config_file', 'full_path']
    full_results_filename=f"evaluation_{common_env_name}_full.csv"
    df[cols].to_csv(full_results_filename, index=False)
    print(f"Detailed results saved to {full_results_filename}")

    summary['env'] = common_env_name
    return summary
    


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate models based on config files.")
    
    # CHANGED: Now accepts a list of config files instead of envnames + base config
    parser.add_argument("--config_files", nargs="+", required=True, 
                        help="List of YAML config files (e.g. configs/beamrider_config.yaml)")
    
    parser.add_argument("--eval_episodes", type=int, default=20, 
                        help="Number of episodes to average over per model")
    
    parser.add_argument("--all_results_file", type=str, default=None,
                        help="Path to a CSV file to append the summary results to (e.g. global_results.csv)")

    args = parser.parse_args()

    if args.all_results_file is None:
        print("Need --all_results_file argument")
        exit(1)

    config_file_list = []
    for pattern in args.config_files:
        # glob.glob returns a list of matching paths (or empty list if none)
        # If pattern is a specific filename without *, it just returns [filename] if it exists
        matches = glob.glob(pattern)
        if not matches:
            print(f"Warning: No files matched pattern '{pattern}'")
        config_file_list.extend(matches)
    
    # Remove duplicates and sort for consistent order
    config_file_list = sorted(list(set(config_file_list)))
    
    if not config_file_list:
        print("Error: No valid config files found.")
        exit(1)
    
    print(f"Found {len(config_file_list)} config files to process.")



    summary = run_evaluation_and_process(config_file_list=config_file_list, eval_episodes=args.eval_episodes)
    
    print(f"\nProcessing global results file: {args.all_results_file}")
    
    
    # 2. Rename columns for clarity in global file
    summary_to_save = summary.rename(columns={'mean': 'avg_norm_reward'})
    
    # 3. Reorder columns
    cols_order = ['env', 'algo', 'run_id', 'avg_norm_reward', 'count']
    summary_to_save = summary_to_save[cols_order]

    # 4. Load existing or create new
    if os.path.exists(args.all_results_file):
        try:
            existing_df = pd.read_csv(args.all_results_file, dtype={'run_id': str})
            combined_df = pd.concat([existing_df, summary_to_save], ignore_index=True)
            updated_df = combined_df.drop_duplicates(subset=['env', 'algo', 'run_id'], keep='last')
        except Exception as e:
            print(f"Error reading existing file {args.all_results_file}: {e}")
            print("Creating a new file instead.")
            updated_df = summary_to_save
    else:
        print("File does not exist. Creating new one.")
        updated_df = summary_to_save

    # 5. Write back
    updated_df.to_csv(args.all_results_file, index=False)
    print(f"Successfully appended summary to {args.all_results_file}")

    # --- NEW: COMPARATIVE SUMMARY ---
    print("\n" + "="*60)
    print("GLOBAL COMPARISON (PPO vs FRPPO)")
    print("="*60)

    # 1. Total Reward Sums
    total_ppo_reward = updated_df[updated_df['algo'] == 'PPO']['avg_norm_reward'].sum()
    total_frppo_reward = updated_df[updated_df['algo'] == 'FRPPO']['avg_norm_reward'].sum()

    # 2. Environment Wins
    # Group by Environment and Algo to sum rewards for that specific env
    # (This handles cases where you might have multiple run_ids per algo per env)
    env_scores = updated_df.groupby(['env', 'algo'])['avg_norm_reward'].sum().unstack()

    # Count wins
    # We check if PPO column > FRPPO column. 
    # fillna(0) ensures we don't crash if an algo is missing for an env
    env_scores = env_scores.fillna(0)
    
    ppo_better_count = 0
    frppo_better_count = 0
    
    if 'PPO' in env_scores.columns and 'FRPPO' in env_scores.columns:
        ppo_better_count = (env_scores['PPO'] > env_scores['FRPPO']).sum()
        frppo_better_count = (env_scores['FRPPO'] > env_scores['PPO']).sum()
    elif 'PPO' in env_scores.columns:
         # Only PPO exists
         ppo_better_count = len(env_scores)
    elif 'FRPPO' in env_scores.columns:
         # Only FRPPO exists
         frppo_better_count = len(env_scores)

    print(f"Total Normalized Reward (PPO)   : {total_ppo_reward:.4f}")
    print(f"Total Normalized Reward (FRPPO) : {total_frppo_reward:.4f}")
    print("-" * 60)
    print(f"Environments where PPO > FRPPO  : {ppo_better_count}")
    print(f"Environments where FRPPO > PPO  : {frppo_better_count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()