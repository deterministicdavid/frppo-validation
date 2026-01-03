#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

echo "This assumes all training has run; otherwise it will fail on model loading."

eval_runs=20

# skip "invertedpendulum_config_*" as everything gets full score
mujoco_envs=(
    "halfcheetah_config_*"
    "hopper_config_*"
    "reacher_config_*"
    "swimmer_config_*"
    "walker2d_config_*"
)

for pattern in "${mujoco_envs[@]}"
do
    echo "Processing configs for pattern: $pattern"
    
    python multi_evaluate.py \
        --eval_episodes=$eval_runs \
        --all_results_file=mujoco_all_results.csv \
        --config_files="./configs/$pattern"
        
    echo "---------------------------------------------------"
done

atari_envs=(
    "beamrider_config_*"
    "breakout_config_*"
    "kungfumaster_config_*"
    "mspacman_config_*"
    "pong_config_*"
    "spaceinvaders_config_*"
)

for pattern in "${atari_envs[@]}"
do
    echo "Processing configs for pattern: $pattern"
    
    python multi_evaluate.py \
        --eval_episodes=$eval_runs \
        --all_results_file=atari_all_results.csv \
        --config_files="./configs/$pattern"
        
    echo "---------------------------------------------------"
done


