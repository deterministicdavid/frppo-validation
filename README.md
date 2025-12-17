# frppo-validation

The repo contains code for validating performance of FR-PPO. 

## Getting started

1. Create a Python virtual environment with PIP and add `torch`, `numpy`, `matplotlib` and all the Farama `gymnasium` with Atari and Mujoco support. 
1. Next you need a local copy of SB3 with FR-PPO implementation. If you got here via my github and view this online then the link [stable-baselines3-contrib-with-frppo](../stable-baselines3-contrib-with-frppo) should work. Otherwise download the [anonymyzed version](https://anonymous.4open.science/r/stable-baselines3-contrib-with-frppo-F821/). 
1. Add the local copy of SB3 to your Python virtual environment.

## General use

Use `python run.py --train --config=configs/<choose a config>.yaml --parallel_runs=<number of identical runs with different random seed>`

Use the `--visualise` flag to run a visualisation of how a trained model performs.

Use the `--optuna` flag if your config file has an Optuna section for metaparameter tuning.

