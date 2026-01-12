#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

env_name=walker2d


for config_path in configs_alt/${env_name}_config_*.yaml
do
    echo "Starting training for $config_path..."
    python run.py --train --config="$config_path" --parallel_runs=6
done

python plots.py --figsize 6 3 --envname $env_name --outfile ppo_vs_frppo_${env_name}.pdf  --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1

