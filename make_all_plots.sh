#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

python plots.py --figsize 6 3 --envname walker2d --outfile ppo_vs_frppo_walker2d.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 FRPPO_2 --labels "PPO clip 0.2" "PPO clip 0.1" "FRPPO tau 0.001" "FRPPO tau 0.005" "FRPPO tau 0.0005"

#python plots.py --figsize 6 3 --envname walker2d --outfile ppo_vs_frppo_walker2d_select.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 FRPPO_2 --labels "PPO clip 0.2"  "PPO clip 0.1" "FRPPO tau 0.001" "FRPPO tau 0.001; final lr frac 0.5" "FRPPO tau 0.001; final lr frac 0.5"
