import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def load_runs(pattern, metric="rollout/ep_rew_mean", step="time/total_timesteps"):
    dfs = []
    if len(glob.glob(pattern)) == 0:
        raise ValueError(f"Dir indicated by {pattern} doesn't exist.")
    for folder in glob.glob(pattern):
        csv_path = os.path.join(folder, "progress.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if step in df.columns and metric in df.columns:
                df = df[[step, metric]].dropna()
                dfs.append(df)
        
        
    return dfs

def interpolate_runs(dfs, num_points=500):
    """Interpolate all runs to a common grid of timesteps."""
    # Determine global x-range
    xmin = max(df.iloc[0,0] for df in dfs)
    xmax = min(df.iloc[-1,0] for df in dfs)
    x_common = np.linspace(xmin, xmax, num_points)

    interpolated = []
    for df in dfs:
        x = df.iloc[:,0].values
        y = df.iloc[:,1].values
        y_interp = np.interp(x_common, x, y)
        interpolated.append(y_interp)

    return x_common, np.vstack(interpolated)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--envname", type=str, required=True,
                        help="Environment name (breakout, beamrider, etc.)")

    parser.add_argument("--outfile", type=str, required=True,
                        help="Output PDF filename")

    parser.add_argument("--patterns", nargs="+", required=True,
                        help="List of directory prefixes, e.g. PPO_0 FRPPO_0 FRPPO_1")

    parser.add_argument("--labels", nargs="+", required=True,
                        help="List of labels, same length/order as --patterns")

    args = parser.parse_args()

    if len(args.patterns) != len(args.labels):
        raise ValueError("patterns and labels must be the same length")

    log_root = f"./logs_{args.envname}"
    plt.figure(figsize=(12,6))


    for pattern, label in zip(args.patterns, args.labels):
        # Glob pattern
        full_pattern = f"{log_root}/{pattern}*"

        # Load
        dfs = load_runs(full_pattern)

        # Interpolate
        x, y = interpolate_runs(dfs)

        # Mean and std
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

    print("=====================================")
    print(f"Saved plot to {args.outfile}")
    print("=====================================")

if __name__ == "__main__":
    main()