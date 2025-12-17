#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

# this must match the logdir specified in all the config files
rm -rf ./logs_breakout/*

for config_path in configs/breakout_config_*.yaml
do
    echo "Starting training for $config_path..."
    python run.py --train --config="$config_path" --parallel_runs=12
done


python plots.py --figsize 6 3 --envname breakout --outfile ppo_vs_frppo_breakout.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 --labels "PPO clip 0.1" "PPO clip 0.2" "FRPPO tau 0.1" "FRPPO tau 0.05"

