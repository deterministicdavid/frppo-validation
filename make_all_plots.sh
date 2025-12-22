#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

echo "This assumes all training has run; if not no plots will be produced"

# Individual plots
python plots.py --figsize 6 3 --envname breakout --outfile ppo_vs_frppo_breakout.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname walker2d --outfile ppo_vs_frppo_walker2d.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1  

python plots.py --figsize 6 3 --envname hopper --outfile ppo_vs_frppo_hopper.pdf --patterns PPO_0 PPO_1 FRPPO_0  FRPPO_2 

python plots.py --figsize 6 3 --envname beamrider --outfile ppo_vs_frppo_beamrider.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 FRPPO_2 

python plots.py --figsize 6 3 --envname pong --outfile ppo_vs_frppo_pong.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname reacher --outfile ppo_vs_frppo_reacher.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

# Grouped plots

python ./multi_plots.py --envnames pong breakout beamrider --outfile ppo_vs_frppo_atarti.pdf --figsize 11 2.5

python ./multi_plots.py --envnames reacher hopper walker2d --outfile ppo_vs_frppo_mujoco.pdf --figsize 11 2.5