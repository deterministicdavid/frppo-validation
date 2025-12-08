import os
os.environ["MUJOCO_GL"] = "egl"

import multiprocessing as mp
import argparse
import copy


import gymnasium as gym
import ale_py
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.exceptions import TrialPruned

from gymnasium.wrappers import AtariPreprocessing, TransformObservation
from stable_baselines3 import PPO
from sb3_contrib import FRPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecVideoRecorder,
    VecNormalize,
)
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.env_util import make_atari_env, make_vec_env, unwrap_wrapper


import glob
import torch
import random
import yaml

from run_utils import (
    OptunaPruningCallback,
    ForceFloat32Wrapper,
    OverwriteCheckpointCallback,
    select_free_gpu_or_fallback,
    get_free_cuda_gpus,
    post_process_video,
    log_hyper_parameters,
)
from own_policy import CustomActorCriticCnnPolicy


# --- Helper function to create environments ---
def make_env_default(env_name: str, seed: int, training_mode: bool):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if not training_mode:
            env = ForceFloat32Wrapper(env)
        
        env.reset(seed=seed)
        return env

    return _init


def make_env_mujoco(config: dict, seed: int):
    def _init():
        env = gym.make(config["env_name"])
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10.0, 10.0), None)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10.0, 10.0))

        # env.seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init

def make_configured_vec_env(num_envs: int, config: dict, seed: int | None, training_mode=True, video_recording=False, stats_path=None ):
    env_name = config["env_name"]
    env_is_atari = config.get("env_is_atari", True)
    env_is_mujoco = config.get("env_is_mujoco", False)
    assert not (env_is_atari and env_is_mujoco)
    
    n_stack = config["n_stack"]
    
    if env_is_atari:
        env = make_atari_env(
            env_id=env_name,
            n_envs=num_envs,
            seed=seed,
            wrapper_kwargs={
                "noop_max": 30,
                "frame_skip": 4,
                "screen_size": 84,
                "terminal_on_life_loss": True,
                "clip_reward": True,
                "action_repeat_probability": 0.0,
            },
            vec_env_cls=DummyVecEnv,
        )
        # Using frame stacking with n_stack=4 is #7 of "The 37 implementation details of Proximal Policy Optimization"
        env = VecFrameStack(env, n_stack=n_stack)
        # env = VecMonitor(env, filename=None, info_keywords=())

    elif env_is_mujoco:
        if n_stack > 0:
            raise ValueError(f"Frame stacking of {n_stack} configured but Mujoco doesn't do frame stacking.")
        
        def custom_wrappers(env: gym.Env) -> gym.Env:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
            if not training_mode:
                env = ForceFloat32Wrapper(env)
            return env

        env = make_vec_env(env_name, n_envs=num_envs, wrapper_class=custom_wrappers)
        # env = VecMonitor(env, filename=None, info_keywords=())
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        if not training_mode and stats_path is not None:
            # Load the stats file
            if os.path.exists(stats_path):
                print(f"Loading normalization stats from {stats_path}")
                env = VecNormalize.load(stats_path, env)
                env.training = False # Freeze stats!
                env.norm_reward = False # See raw rewards
            else:
                raise ValueError("ERROR: Stats file not found. Retraining required.")
            
    else:
        env_fns = [make_env_default(env_name, seed=i) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)
        
        env = VecFrameStack(env, n_stack=n_stack)
        # env = VecMonitor(env, filename=None, info_keywords=())


    if not training_mode and video_recording:
        MAX_STEPS = 10_000
        learning_algo = config["train"]["algo"]
        video_folder = config["visualize"]["video_folder"]
        os.makedirs(video_folder, exist_ok=True)
    
        env = VecVideoRecorder(
            venv=env,
            video_folder=video_folder,
            record_video_trigger=lambda x: x == 0,  # record first episode
            video_length=MAX_STEPS,
            name_prefix=f"{learning_algo}",
        )
    return env

def make_configured_algo_and_model(env, config: dict, assigned_device: torch.device, seed: int):
    learning_algo = config["train"]["algo"]
    env_is_atari = config.get("env_is_atari", True)
    env_is_mujoco = config.get("env_is_mujoco", False)
    assert not (env_is_atari and env_is_mujoco)

    log_dir = config["logging"]["log_dir"]
    n_steps = config["train"]["n_steps"]
    batch_size = config["train"]["batch_size"]
    
    ent_coef = config["train"]["ent_coef"]

    env_is_atari = config.get("env_is_atari", True)
    env_is_mujoco = config.get("env_is_mujoco", False)
    
    policy = None
    policy_kwargs = {}

    if env_is_atari:
        # Note that using CnnPolicy makes SB3 normalize images to [0,1],
        # which is #9 of "The 37 implementation details of Proximal Policy Optimization"
        policy = "CnnPolicy"
        policy_kwargs = {
            "ortho_init": True,
            "features_extractor_class": NatureCNN,
            "share_features_extractor": True,
            "normalize_images": True,
        }

        if config["train"]["policy"] == "own":
            policy = CustomActorCriticCnnPolicy  # this will also normalize to [0,1]

    elif env_is_mujoco:
        policy = "MlpPolicy"  # TODO: check whether we need to set more options to track "The 37 implementation details"
        policy_kwargs = {
            "ortho_init": True,
            "share_features_extractor": False,
        }
    else:
        policy = "CnnPolicy"  # TODO: this will need more work as it won't work for both car racing and the classic stuff
    
    model = None
    
    lr_decay_init = float(config.get("train", {}).get("decay_lr_init", "3e-4"))
    lr_decay_final_frac = float(config.get("train", {}).get("decay_lr_final_frac", "1.0")) # default value 1.0 means we don't decay
    assert(lr_decay_final_frac >= 0.0 and lr_decay_final_frac <= 1.0)
    lr_decay_final = lr_decay_final_frac * lr_decay_init
    learning_rate = lambda f: f * lr_decay_init + (1.0-f) * lr_decay_final

    default_n_opt_epochs = 4
    n_opt_epochs = config.get("train", {}).get("n_opt_epochs", default_n_opt_epochs)

    if learning_algo == "FRPPO":
        fr_tau_penalty = config["train"]["fr_tau_penalty"]
        fr_penalty_scale_by_adv = config["train"]["fr_scale_by_adv"]
        model = FRPPO(
            policy=policy,
            env=env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            n_epochs=n_opt_epochs,
            batch_size=batch_size,
            fr_penalty_tau=fr_tau_penalty,
            fr_penalty_scale_by_adv=fr_penalty_scale_by_adv,
            ent_coef=ent_coef,
            tensorboard_log=log_dir,
            device=assigned_device,
            policy_kwargs=policy_kwargs,
        )
    elif learning_algo == "PPO":
        default_clip_epsilon = 0.2
        clip_epsilon = config.get("train", {}).get("clip_epsilon", default_clip_epsilon)
        model = PPO(
            policy=policy,
            env=env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            n_epochs=n_opt_epochs,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_epsilon,
            tensorboard_log=log_dir,
            device=assigned_device,
            policy_kwargs=policy_kwargs,
        )
    else:
        print(f"Learning algorithm {learning_algo} may be in SB3 but not it's not been setup here.")
        return

    return model

def evaluate_model(model, config: dict, num_eval_runs: int = 10):
    """
    Evaluates the trained model over multiple episodes using unnormalized rewards.

    Returns:
        float: The mean unnormalized total reward across all evaluation runs.
    """
    env_is_mujoco = config.get("env_is_mujoco", False)
    
    # 1. Create a dedicated evaluation environment (num_envs=1)
    # Crucially, we use training_mode=False to ensure reward wrappers are absent 
    # or disabled for raw score reporting.
    eval_env = make_configured_vec_env(
        num_envs=1, 
        config=config, 
        seed=None, # Use random seed for evaluation
        training_mode=False,
        video_recording=False
    )
    
    all_episode_rewards = []

    for _ in range(num_eval_runs):
        obs = eval_env.reset()
        done = False
        rewards = 0
        
        # Determine if we need to load VecNormalize stats (Mujoco only)
        if env_is_mujoco:
            normalized_env = unwrap_wrapper(eval_env, VecNormalize)
            if normalized_env is not None and normalized_env.norm_reward:
                # If reward normalization is mistakenly active, issue a warning.
                print("Warning: Mujoco evaluation environment has active reward normalization.")

        while not done:
            # Use deterministic actions for a stable evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _ = eval_env.step(action)
            done = done[0] # Unwrap from VecEnv structure
            rewards += rew[0]

        all_episode_rewards.append(rewards)

    eval_env.close()
    
    mean_unnormalized_reward = np.mean(all_episode_rewards)
    return mean_unnormalized_reward

def optuna_objective(trial: optuna.Trial, base_config: dict, assigned_device: torch.device, seed: int) -> float:
    """
    Optuna objective function to sample hyperparameters, train the model,
    and return the performance metric (e.g., negative mean reward).
    """
    
    # 1. Deep copy the config to safely modify it for the current trial
    trial_config = copy.deepcopy(base_config)
    
    # Learning rate (log uniform)
    if trial_config["optuna"]["search_space"]["learning_rate_log"] is not None:
        lr_low, lr_high = trial_config["optuna"]["search_space"]["learning_rate_log"]
        lr = trial.suggest_float("learning_rate", low=float(lr_low), high=float(lr_high),log=True)
        trial_config["train"]["learning_rate"] = lr # Store fixed LR in config
    
    # Sample n_steps 
    # pow10_low, pow10_high = trial_config["optuna"]["search_space"].get("total_timesteps_pow10", [3, 4])
    # total_timesteps_exponent = trial.suggest_int("total_timesteps_exponent", low=pow10_low, high=pow10_high)
    # trial_config["train"]["total_timesteps"] = 10**total_timesteps_exponent
    
    # Sample entropy coefficient (uniform)
    ent_coeff_entry = trial_config["optuna"]["search_space"].get("ent_coef_uniform", None)
    if ent_coeff_entry is not None:
        ent_low, ent_high = ent_coeff_entry
        ent_coef = trial.suggest_float("ent_coef", ent_low, ent_high,log=False)
        trial_config["train"]["ent_coef"] = ent_coef
    
    # Final lr as fraction of initial LR
    if trial_config["optuna"]["search_space"]["decay_lr_final_frac"] is not None:
        decay_lr_final_frac_low, decay_lr_final_frac_high = trial_config["optuna"]["search_space"]["decay_lr_final_frac"]
        decay_lr_final_frac = trial.suggest_float("decay_lr_final_frac", low=float(decay_lr_final_frac_low), high=float(decay_lr_final_frac_high))
        trial_config["train"]["decay_lr_final_frac"] = decay_lr_final_frac

    # FRPPO penalty parameter
    if trial_config["train"]["algo"] == "FRPPO" and trial_config["optuna"]["search_space"]["fr_tau_penalty_log"] is not None:
        fr_low, fr_high = trial_config["optuna"]["search_space"]["fr_tau_penalty_log"]
        fr_tau_penalty = trial.suggest_float("fr_tau_penalty", fr_low, fr_high,log=True)
        trial_config["train"]["fr_tau_penalty"] = fr_tau_penalty

    # PPO clip parameter
    if trial_config["train"]["algo"] == "PPO" and trial_config["optuna"]["search_space"]["clip_epsilon_uniform"] is not None:
        clip_low, clip_high = trial_config["optuna"]["search_space"]["clip_epsilon_uniform"]
        clip = trial.suggest_float("clip_epsilon", clip_low, clip_high, log=False)
        trial_config["train"]["clip_epsilon"] = clip

    
    # 3. Update Logging and Run ID for this trial
    trial_id = trial.number
    original_prefix = trial_config["logging"]["name_prefix"]
    trial_config["train"]["run_id"] = f"optuna_trial_{trial_id}_{assigned_device.type}_{assigned_device.index}"
    trial_config["logging"]["name_prefix"] = f"{original_prefix}_trial_{trial_id}"
    
    
    # Overwrite the checkpoint callback with the OptunaPruningCallback
    # You may want to run both.
    pruning_callback = OptunaPruningCallback(
        trial=trial, 
        freq=trial_config["logging"]["checkpoint_save_freq"],
        n_steps=config["train"]["n_steps"],
        n_envs=trial_config["train"]["n_envs"]
    )
    
    # 5. Run Training
    final_reward = train(trial_config, assigned_device, seed, extra_callbacks=[pruning_callback])
    
    # Optuna minimizes the objective, so we return the negative reward.
    return -final_reward

def run_optuna_worker(worker_id: int, config: dict, gpus_to_use: list):
    """
    Worker function that runs independently in its own process.
    Each worker pulls trials from the shared database.
    """
    
    assigned_device = random.choice(gpus_to_use)
    print(f"Worker {worker_id} starting, assigned to GPU {assigned_device}")
    
    run_config = copy.deepcopy(config)
    original_run_id = config.get("train", {}).get("run_id", "0")
    original_prefix = run_config["logging"]["name_prefix"]
    run_config["train"]["run_id"] = f"{original_run_id}_optuna_worker_{worker_id}_on_gpu_{assigned_device.type}_{assigned_device.index}"
    run_config["logging"]["name_prefix"] = f"{original_prefix}_{original_run_id}_{worker_id}_run_on_gpu_{assigned_device.index}" 


    config_optuna = run_config["optuna"]
    storage = config_optuna["storage"]
    study_name = config_optuna["study_name"]
    n_trials_per_worker = config_optuna["n_trials_per_worker"]
    
    # Create/load study (all workers share the same study via database)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )
    
    # Each worker runs trials independently
    # Optuna handles synchronization via the database
    study.optimize(
        lambda trial: optuna_objective(trial, run_config, assigned_device=assigned_device, seed=None),
        n_trials=n_trials_per_worker,
        show_progress_bar=False,
    )
    
    print(f"Worker {worker_id} finished")



def train(config: dict, assigned_device: torch.device, seed: int, extra_callbacks=None):    
    num_envs = config["train"]["n_envs"]
    log_dir = config["logging"]["log_dir"]
    total_timesteps = config["train"]["total_timesteps"]
    name_prefix = config["logging"]["name_prefix"]
    save_freq = config["logging"]["checkpoint_save_freq"]
    
    env = make_configured_vec_env(num_envs=num_envs, config=config, seed=seed, training_mode=True)
    model = make_configured_algo_and_model(env=env, config=config, assigned_device=assigned_device, seed=seed)
    
    checkpoint_callback = OverwriteCheckpointCallback(
        save_freq=save_freq, save_path=log_dir, name_prefix=name_prefix  # This is now based on total timesteps
    )

    tb_path = os.path.join(log_dir, f"{model.__class__.__name__}_{config['train']['run_id']}")
    os.makedirs(tb_path, exist_ok=True)
    new_logger = configure(tb_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    log_hyper_parameters(logger=model.logger, config=config)

    callbacks = [checkpoint_callback]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    save_file = os.path.join(log_dir, name_prefix)
    model.save(save_file)
    env.close()
    
    
    n_eval_runs = 20
    final_score = evaluate_model(model=model, config=config, num_eval_runs=n_eval_runs)
    print(f"Evaluation with {n_eval_runs} runs, reward: {final_score}")
    return final_score


def vizualize(config: dict, model_path, scale_up=False):
    print("Starting visualization...")
    
    learning_algo = config["train"]["algo"]
    video_folder = config["visualize"]["video_folder"]
    deterministic_actions = config["visualize"]["deterministic"]
    
    stats_path = os.path.join(os.path.dirname(model_path), f"{os.path.basename(model_path).replace('.zip', '')}_env.pkl")
    env = make_configured_vec_env(num_envs=1, 
                                  config=config, 
                                  seed=None, 
                                  training_mode=False, 
                                  video_recording=True,
                                  stats_path=stats_path)
    model = None
    if learning_algo == "FRPPO":
        model = FRPPO.load(model_path, env=env)
    elif learning_algo == "PPO":
        model = PPO.load(model_path, env=env)
    else:
        print(f"Learning algorithm {learning_algo} may be in SB3 but not it's not been setup here.")
        return
    
    print(f"Model loaded from {model_path}.")
    
    # Run one episode
    obs = env.reset()
    done = False
    rewsum = 0
    step = 0
    MAX_STEPS = 10_000
    while not done:
        step += 1
        # Use deterministic actions for evaluation
        action, _ = model.predict(obs, deterministic=deterministic_actions)
        obs, rew, done, info = env.step(action)
        done = done[0]  # we only have one envirnoment
        rewsum += rew[0]
        if (step + 1) % 100 == 0:
            print(f"Step: {step+1}, reward so far: {rewsum:.2f}")
        if step > MAX_STEPS:
            break

    # The video is saved automatically when the environment is closed
    env.close()
    print(f"Visualization complete. Video saved in '{video_folder}' folder. Total steps is {step}. Total reward is {rewsum}.")

    if scale_up:
        # Find the most recently created video file in the folder
        list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        if not list_of_files:
            print("Error: No video file found to post-process.")
            return

        latest_file = max(list_of_files, key=os.path.getctime)
        output_file = os.path.join(video_folder, f"scaled_{os.path.basename(latest_file)}")

        # Scale the video
        post_process_video(latest_file, output_file, scale_factor=4)


if __name__ == "__main__":
    # try:
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass

    # --- Add command line argument parsing ---
    parser = argparse.ArgumentParser(description="Train or visualize a PPO or FRPPO agent for Atari or Mujoco gym.")
    parser.add_argument("--train", action="store_true", help="Run the training process.")
    parser.add_argument("--visualise", action="store_true", help="Run the visualization process.")
    parser.add_argument("--model", type=str, default="", help="Path to the trained model when running vis.")
    parser.add_argument("--vis_scaleup", action="store_true", help="Post process the video to scale up.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument(
        "--parallel_runs",
        type=int,
        default=1,
        help="Maximum number of parallel training runs. Will be limited by the number of free GPUs.",
    )
    # --- Add Optuna flag to argparse ---
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter optimization with Optuna.")
    
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        parser.print_help()
        exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not args.train and not args.visualise and not args.optuna:
        print("No action specified. Please use one of --train, --optuna, --visualise. Use -h, --help to see cmdline param help.")
        exit(1)
    elif args.train and args.optuna:
        print("")
        print("*** No point running train and optuna at the same time. Use -h, --help to see cmdline param help.***")
        print("")
        exit(1)
        
    else:
        if args.train:
            print(f"--- Preparing for up to {args.parallel_runs} parallel training run(s) ---")
            
            seed = config["train"].get("random_seed", None)
            force_cpu = config["train"].get("force_cpu", False)
            
            gpus_to_use = get_free_cuda_gpus(max_count=args.parallel_runs, force_cpu=force_cpu)
        
            print(
                f"Found {len(gpus_to_use)} free GPUs. Launching {args.parallel_runs} parallel runs."
            )
            processes = []
            for run_idx in range(0, args.parallel_runs):
                device_idx = run_idx % len(gpus_to_use) # int div remainder so e.g. with 8 gpus and 10 runs we'd cycle through 0, 1, .., 7, 0, 1
                device = gpus_to_use[device_idx]
                # copy and overwrite config
                run_config = copy.deepcopy(config)
                def_run_id = "0"
                original_run_id = config.get("train", {}).get("run_id", def_run_id)
                # original_run_id = run_config["train"]["run_id"]
                original_prefix = run_config["logging"]["name_prefix"]
                run_config["train"]["run_id"] = f"{original_run_id}_run_{run_idx}_{device_idx}_gpu_{device.type}_{device.index}"
                run_config["logging"][
                    "name_prefix"
                ] = f"{original_prefix}_{original_run_id}_{run_idx}_run_{device_idx}_gpu_{device.index}"  # this is where the train model gets saved

                # 4. Create and start the process
                p = mp.Process(target=train, args=(run_config, device, seed))
                processes.append(p)
                p.start()
                if seed is not None:
                    seed += 1

            # 5. Wait for all processes to finish
            for p in processes:
                p.join()

            

        if args.visualise:
            print("--- Running Visualization ---")
            vizualize(config=config, model_path=args.model, scale_up=args.vis_scaleup)

        if args.optuna:
            print("--- Running Optuna Optimization ---")
            
            optuna_config = config["optuna"]
            storage = optuna_config["storage"]
            study_name = optuna_config["study_name"]
            n_startup_trials = optuna_config["n_startup_trials"]
            sampler_name = optuna_config.get("sampler", "unknown")
            n_jobs = args.parallel_runs 
            
            sampler = None 
            if sampler_name == "TPESampler":
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=n_startup_trials,  # Random sampling for first n_jobs trials
                    multivariate=True,    # Better for parallel optimization
                    seed=42
                )
            elif sampler_name == "GPSampler":
                sampler = optuna.samplers.GPSampler(
                    n_startup_trials=n_startup_trials,
                    deterministic_objective=False,
                    seed=42
                )
            else:
                raise ValueError(f"Sampler is {sampler_name}, must be on of [GPSampler, TPESampler]")
            
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                storage=storage,
                sampler=sampler,
                load_if_exists=False,
            )

            seed = config["train"].get("random_seed", None)
            force_cpu = config["train"].get("force_cpu", False)
            
            gpus_to_use = get_free_cuda_gpus(max_count=args.parallel_runs, force_cpu=force_cpu)
        
            
            processes = []
            for worker_id in range(0, args.parallel_runs):
                p = mp.Process(target=run_optuna_worker, args=(worker_id, config, gpus_to_use))
                processes.append(p)
                p.start()
                if seed is not None:
                    seed += 1

            for p in processes:
                p.join()


            print("\n--- Optuna Optimization Finished ---")
            if study.best_trial:
                print(f"Best trial: {study.best_trial.value:.4f} (Negative Reward)")
                print("Best hyperparameters:")
                for key, value in study.best_trial.params.items():
                    print(f"  {key}: {value}")
            else:
                print("WARNING: No complete trials found in the study.")