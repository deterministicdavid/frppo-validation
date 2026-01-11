import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import math
from collections import defaultdict


def load_df_from_folder(
    folder, metric="rollout/ep_rew_mean", step="time/total_timesteps"
):
    """
    Loads a single run dataframe from a folder.
    """
    csv_path = os.path.join(folder, "progress.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if step in df.columns and metric in df.columns:
                return df[[step, metric]].dropna()
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    return None


def load_runs_by_pattern(
    pattern, metric="rollout/ep_rew_mean", step="time/total_timesteps"
):
    """
    Existing logic: loads runs matching a glob pattern.
    """
    dfs = []
    folders = glob.glob(pattern)
    for folder in folders:
        df = load_df_from_folder(folder, metric, step)
        if df is not None:
            dfs.append(df)
    return dfs


def interpolate_runs(dfs, num_points=500):
    if not dfs:
        return np.array([]), np.array([])

    try:
        xmin = max(df.iloc[0, 0] for df in dfs)
        xmax = min(df.iloc[-1, 0] for df in dfs)
    except (IndexError, ValueError):
        return np.array([]), np.array([])

    if xmin >= xmax:
        return np.array([]), np.array([])

    x_common = np.linspace(xmin, xmax, num_points)

    interpolated = []
    for df in dfs:
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        # Linear interpolation
        y_interp = np.interp(x_common, x, y)
        interpolated.append(y_interp)

    return x_common, np.vstack(interpolated)


def get_auto_label(folder):
    """
    Generates a label based on hyperparameters in progress.csv.
    """
    csv_path = os.path.join(folder, "progress.csv")
    if not os.path.exists(csv_path):
        return "Unknown"

    try:
        # Read just the header and first row
        df = pd.read_csv(csv_path, nrows=1)
        df.columns = df.columns.str.strip()

        # Check Algo
        algo_col = "hyperparameters/train.algo"
        if algo_col not in df.columns:
            return "Unknown Algo"

        algo = df[algo_col].iloc[0]

        # Detailed labeling for known algorithms
        if algo == "PPO":
            clip_col = "hyperparameters/clip_epsilon"
            if clip_col not in df.columns:
                clip_col = "hyperparameters/train.clip_epsilon"

            if clip_col in df.columns:
                return f"PPO clip {df[clip_col].iloc[0]}"

        elif algo == "FRPPO":
            tau_col = "hyperparameters/fr_tau_penalty"
            if tau_col not in df.columns:
                tau_col = "hyperparameters/train.fr_tau_penalty"

            if tau_col in df.columns:
                return f"FRPPO tau {df[tau_col].iloc[0]}"

        return str(algo)

    except Exception:
        return "Unknown"


def plot_single_env(ax, dir_prefix, envname, patterns, user_labels):
    log_root = f"{dir_prefix}{envname}"

    # Dictionary to hold data: {'Label Name': [df1, df2, ...]}
    plot_groups = defaultdict(list)

    # --- STRATEGY 1: User provided patterns (Backward Compatibility) ---
    if patterns is not None:
        for i, pattern in enumerate(patterns):
            full_pattern = f"{log_root}/{pattern}*"
            dfs = load_runs_by_pattern(full_pattern)

            if not dfs:
                continue

            # Determine label
            if user_labels:
                label = user_labels[i]
            else:
                # Use metadata from the first folder found to label this group
                matching = glob.glob(full_pattern)
                label = get_auto_label(matching[0]) if matching else pattern

            # Store
            plot_groups[label].extend(dfs)

    # --- STRATEGY 2: Auto-Group by Hyperparameters (New Logic) ---
    else:
        # 1. Find all subdirectories
        all_folders = [f.path for f in os.scandir(log_root) if f.is_dir()]

        if not all_folders:
            print(f"[{envname}] No folders found in {log_root}")
            ax.text(0.5, 0.5, "No Data", ha="center")
            return

        # 2. Iterate every folder, extract label, and group
        for folder in all_folders:
            label = get_auto_label(folder)
            df = load_df_from_folder(folder)

            if df is not None:
                plot_groups[label].append(df)

    # --- Plotting Phase ---
    has_data = False

    # Sort keys to ensure consistent legend order
    sorted_labels = sorted(plot_groups.keys())

    for label in sorted_labels:
        dfs = plot_groups[label]
        if not dfs:
            continue

        x, y = interpolate_runs(dfs)
        if len(x) == 0:
            continue

        # Calculate Mean and Std
        if y.ndim == 1:
            mean = y
            std = np.zeros_like(y)
        else:
            mean = y.mean(axis=0)
            std = y.std(axis=0)

        ax.plot(x, mean, label=label, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        has_data = True

    ax.set_title(envname.capitalize())
    ax.grid(True, alpha=0.3)

    if has_data:
        ax.legend(fontsize="small")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir_prefix",
        required=False,
        default="./logs_",
        help="What goes before envname when looking for data.",
    )

    parser.add_argument(
        "--envnames", nargs="+", required=True, help="List of Environment names"
    )
    parser.add_argument(
        "--outfile", type=str, required=True, help="Output PDF filename"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        required=False,
        default=None,
        help="Optional list of patterns.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=False,
        default=None,
        help="Optional list of labels.",
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[15, 5], help="Figure size"
    )

    args = parser.parse_args()

    # Layout Setup
    n_envs = len(args.envnames)
    cols = 3
    rows = math.ceil(n_envs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=args.figsize, constrained_layout=True)

    if n_envs == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()

    print(f"Plotting {n_envs} environments...")

    for i, envname in enumerate(args.envnames):
        plot_single_env(
            ax=axes_flat[i],
            dir_prefix=args.dir_prefix,
            envname=envname,
            patterns=args.patterns,
            user_labels=args.labels,
        )

        # Shared axis labels
        if i % cols == 0:
            axes_flat[i].set_ylabel("Episode Reward")
        if i >= n_envs - cols:
            axes_flat[i].set_xlabel("Timesteps")

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.savefig(args.outfile, format="pdf")
    plt.close()
    print(f"Saved to {args.outfile}")


if __name__ == "__main__":
    main()
