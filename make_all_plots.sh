#!/bin/bash

# we want to use python libraries from the env
. venv/bin/activate

echo "This assumes all training has run; if not empty plots will result"

# Individual plots Atari
python plots.py --figsize 6 3 --envname breakout --outfile ppo_vs_frppo_breakout.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname beamrider --outfile ppo_vs_frppo_beamrider.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1

python plots.py --figsize 6 3 --envname kungfumaster --outfile ppo_vs_frppo_kungfumaster.pdf  --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1

python plots.py --figsize 6 3 --envname mspacman --outfile ppo_vs_frppo_mspacman.pdf  --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1

python plots.py --figsize 6 3 --envname pong --outfile ppo_vs_frppo_pong.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname robotank --outfile ppo_vs_frppo_robotank.pdf  --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1

# Individual plots Mujoco
python plots.py --figsize 6 3 --envname walker2d --outfile ppo_vs_frppo_walker2d.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1  

python plots.py --figsize 6 3 --envname hopper --outfile ppo_vs_frppo_hopper.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname reacher --outfile ppo_vs_frppo_reacher.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname invertedpendulum --outfile ppo_vs_frppo_invertedpendulum.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname swimmer --outfile ppo_vs_frppo_swimmer.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

python plots.py --figsize 6 3 --envname halfcheetah --outfile ppo_vs_frppo_halfcheetah.pdf --patterns PPO_0 PPO_1 FRPPO_0 FRPPO_1 

# Grouped plots
python ./multi_plots.py --envnames breakout beamrider kungfumaster mspacman pong spaceinvaders --outfile ppo_vs_frppo_all_atari.pdf --figsize 11 5

python ./multi_plots.py --envnames invertedpendulum reacher hopper walker2d swimmer halfcheetah --outfile ppo_vs_frppo_all_mujoco.pdf --figsize 11 5