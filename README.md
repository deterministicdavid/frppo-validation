# frppo-validation

The repo contains code for validating performance of FR-PPO. 

## Getting started


1. Create a Python virtual environment with PIP and add `torch`, `numpy`, `matplotlib` and all the Farama `gymnasium` with Atari and Mujoco support. 
1. Next you need a local copy of SB3 with FR-PPO implementation. If you got here via my github then go to my projects and find `stable-baselines3-contrib-with-frppo`. Otherwise download the [anonymyzed version](https://anonymous.4open.science/r/sb3-contrib-frppo/). 
1. Add the local copy of SB3 to your Python virtual environment.

## General use

Use `--train` to run training. For example 
```
python run.py --train --config=configs/breakout_config_frppo1.yaml --parallel_runs=1
```

Use the `--visualise` flag to run a visualisation of how a trained model performs, for example
```
python run.py --visualise --config configs/breakout_config_frppo1.yaml --model logs_breakout/frppo1_breakout_latest_1_0_run_0_gpu_0.zip --vis_scaleup
```

Use the `--optuna` flag if your config file has an Optuna section for metaparameter tuning. 
For example
```
python run.py --optuna --config configs/hopper_config_optuna_frppo0.yaml --parallel_runs 2
```

