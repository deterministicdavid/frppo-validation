import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def load_runs(pattern, metric="rollout/ep_rew_mean", step="time/total_timesteps"):
    dfs = []
    if len(glob.glob(pattern)) == 0:
        # Warning instead of error to allow other patterns to proceed
        print(f"Warning: Dir indicated by {pattern} doesn't exist.")
        return []
        
    for folder in glob.glob(pattern):
        csv_path = os.path.join(folder, "progress.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if step in df.columns and metric in df.columns:
                    df = df[[step, metric]].dropna()
                    dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        
    return dfs

def interpolate_runs(dfs, num_points=500):
    """Interpolate all runs to a common grid of timesteps."""
    if not dfs:
        return np.array([]), np.array([])
        
    # Determine global x-range
    xmin = max(df.iloc[0,0] for df in dfs)
    xmax = min(df.iloc[-1,0] for df in dfs)
    x_common = np.linspace(xmin, xmax, num_points)

    interpolated = []
    for df in dfs:
        x = df.iloc[:,0].values
        y = df.iloc[:,1].values
        # Using np.interp for linear interpolation
        y_interp = np.interp(x_common, x, y)
        interpolated.append(y_interp)

    return x_common, np.vstack(interpolated)

def get_auto_label(folder):
    """
    Reads progress.csv metadata to generate a label.
    PPO -> PPO clip <val>
    FRPPO -> FRPPO tau <val>
    """
    csv_path = os.path.join(folder, "progress.csv")
    if not os.path.exists(csv_path):
        return "Unknown"

    try:
        # Read only the first row to get hyperparameters
        df = pd.read_csv(csv_path, nrows=1)
        
        # Identify columns (handle potential naming variations if needed)
        algo_col = "hyperparameters/train.algo"
        
        if algo_col not in df.columns:
            return "Unknown Algo"

        algo = df[algo_col].iloc[0]

        if algo == "PPO":
            # Target: hyperparameters/clip_epsilon
            clip_col = "hyperparameters/clip_epsilon"
            # Fallback if the user meant 'train.clip_epsilon' (common in some loggers)
            if clip_col not in df.columns:
                 clip_col = "hyperparameters/train.clip_epsilon"
            
            if clip_col in df.columns:
                val = df[clip_col].iloc[0]
                return f"PPO clip {val}"
            else:
                return "PPO"

        elif algo == "FRPPO":
            # Target: hyperparameters/fr_tau_penalty
            # Note: The CSV snippet shows 'hyperparameters/train.fr_tau_penalty'
            tau_col = "hyperparameters/fr_tau_penalty"
            if tau_col not in df.columns:
                tau_col = "hyperparameters/train.fr_tau_penalty"
            
            if tau_col in df.columns:
                val = df[tau_col].iloc[0]
                return f"FRPPO tau {val}"
            else:
                return "FRPPO"

        return str(algo)

    except Exception as e:
        print(f" [AutoLabel Error: {e}]")
        return "Error"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--envname", type=str, required=True,
                        help="Environment name (breakout, beamrider, etc.)")

    parser.add_argument("--outfile", type=str, required=True,
                        help="Output PDF filename")

    parser.add_argument("--patterns", nargs="+", required=True,
                        help="List of directory prefixes, e.g. PPO_0 FRPPO_0 FRPPO_1")

    #  -> Not triggered as it's not a domain query.
    # Modified: labels is now optional
    parser.add_argument("--labels", nargs="+", required=False, default=None,
                        help="List of labels. If omitted, labels are auto-generated from csv metadata.")

    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 6],
                        help="Figure size: width height (default: 12 6)")

    args = parser.parse_args()

    # Check if labels provided match patterns length
    if args.labels is not None and len(args.patterns) != len(args.labels):
        raise ValueError("If provided, labels list must be the same length as patterns list")

    log_root = f"./logs_{args.envname}"
    plt.figure(figsize=args.figsize)

    for i, pattern in enumerate(args.patterns):
        # Glob pattern
        full_pattern = f"{log_root}/{pattern}*"
        
        # Find folders to determine label (if needed)
        matching_folders = glob.glob(full_pattern)
        if not matching_folders:
            print(f"Skipping {pattern}: No matching folders found.")
            continue

        # Determine Label
        if args.labels is None:
            # Auto-generate label from the first folder found in this group
            label = get_auto_label(matching_folders[0])
        else:
            label = args.labels[i]

        # Load
        dfs = load_runs(full_pattern)
        
        if not dfs:
            continue

        # Interpolate
        x, y = interpolate_runs(dfs)
        if len(x) == 0: 
            continue

        # Mean and std
        if y.ndim == 1:
            mean = y
            std = np.zeros_like(y)
        else:
            mean = y.mean(axis=0)
            std = y.std(axis=0)

        # Plot
        plt.plot(x, mean, label=label, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title(f"Training Curves on {args.envname.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and close
    plt.savefig(args.outfile, format="pdf")
    plt.close()

    print(f"Saved plot to {args.outfile}")
    print("=====================================")

if __name__ == "__main__":
    main()