#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

echo "This assumes all training has run; if not no plots will be produced"

python plots.py --figsize 6 3 --envname breakout --outfile ppo_vs_frppo_breakout.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname walker2d --outfile ppo_vs_frppo_walker2d.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1  

python plots.py --figsize 6 3 --envname hopper --outfile ppo_vs_frppo_hopper.pdf --patterns PPO_0 FRPPO_0 FRPPO_1 FRPPO_2 FRPPO_3 

python plots.py --figsize 6 3 --envname beamrider --outfile ppo_vs_frppo_beamrider.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 FRPPO_2 