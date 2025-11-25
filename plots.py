import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_runs(pattern, metric="rollout/ep_rew_mean", step="time/total_timesteps"):
    dfs = []
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

# ----------------------------
# LOAD PPO RUNS
# ----------------------------
ppo_runs = load_runs("./logs_breakout/PPO_0_*")

# LOAD FRPPO RUNS
frppo_runs = load_runs("./logs_breakout/FRPPO_0_*")

# ----------------------------
# INTERPOLATE
# ----------------------------
x_ppo, y_ppo = interpolate_runs(ppo_runs)
x_frppo, y_frppo = interpolate_runs(frppo_runs)

# Compute mean + std
ppo_mean = y_ppo.mean(axis=0)
ppo_std  = y_ppo.std(axis=0)

frppo_mean = y_frppo.mean(axis=0)
frppo_std  = y_frppo.std(axis=0)

# ----------------------------
# PLOTTING
# ----------------------------
plt.figure(figsize=(12,6))

# PPO curve
plt.plot(x_ppo, ppo_mean, label="PPO", linewidth=2)
plt.fill_between(x_ppo, ppo_mean - ppo_std, ppo_mean + ppo_std, alpha=0.2)

# FRPPO curve
plt.plot(x_frppo, frppo_mean, label="FRPPO", linewidth=2)
plt.fill_between(x_frppo, frppo_mean - frppo_std, frppo_mean + frppo_std, alpha=0.2)

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Breakout Training Performance: PPO vs FRPPO (mean ± std)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_vs_frppo.pdf", format="pdf")
plt.close()
