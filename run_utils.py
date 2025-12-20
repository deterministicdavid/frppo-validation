import os
import collections
import pynvml
import torch
import numpy as np
import gymnasium as gym
import optuna
from optuna.exceptions import TrialPruned

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import unwrap_vec_normalize

import glob
from moviepy import VideoFileClip

# This callback lets Optuna stop unpromising trials early.
class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial: optuna.Trial, freq: int, n_envs: int, n_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.check_freq = freq
        self.n_envs = n_envs
        self.last_check_timesteps = 0
        # Track episode rewards for each parallel environment
        self.current_episode_rewards = [0.0] * n_envs 
        self.n_steps = n_steps # how many to consider for reward
    
        # A deque to store the final reward of completed episodes
        self.episode_rewards = collections.deque(maxlen=self.n_steps) # Track up to 20 recent episodes

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        
        for i in range(self.n_envs):
        # Accumulate reward for the current active episode in environment i
            self.current_episode_rewards[i] += rewards[i] 

            # Check for episode termination (only the 'done' signal is needed)
            if dones[i]:
                # Episode finished: record the total reward and reset the counter
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.current_episode_rewards[i] = 0.0

        if (self.num_timesteps - self.last_check_timesteps) >= self.check_freq:
            self.last_check_timesteps = self.num_timesteps

            if self.episode_rewards:
                # Calculate the mean reward over the tracked completed episodes
                # Using np.mean(self.episode_rewards) gives the mean over the maxlen (200) episodes
                mean_reward = np.mean(self.episode_rewards) 
                
                # Log to SB3 logger (Optional, but useful for TensorBoard)
                self.logger.record("optuna/ep_rew_mean_tracked", mean_reward)
                
                # Report intermediate score to Optuna, with a minus since we minimise
                self.trial.report(-mean_reward, step=self.num_timesteps)
                
                # Handle pruning
                if self.trial.should_prune():
                    print(f"Trial {self.trial.number} pruned at step {self.num_timesteps}.")
                    raise TrialPruned()
    
        return True # Continue training
    
    def get_final_mean_reward(self) -> float:
        if self.episode_rewards:
            return np.mean(self.episode_rewards)
        return -np.inf # Return a very low value if no episodes were completed


class ForceFloat32Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 1. Force the declared space to be float32 to satisfy SB3 check
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        # 2. Force the actual data to be float32
        return obs.astype(np.float32)

class OverwriteCheckpointCallback(BaseCallback):
    """
    Callback for saving a model periodically, overwriting the same file.

    :param save_freq: The frequency (in total timesteps) at which to save the model.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: The name of the file to save the model to.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "latest_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.num_saves = 0
        self.save_path = save_path
        # The full path for the save file, e.g., ./logs/latest_model.zip
        self.save_file = os.path.join(save_path, f"{name_prefix}.zip")
        self.save_file_stats = os.path.join(save_path, f"{name_prefix}_env.pkl")


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # self.num_timesteps is the total number of steps taken in the environment
        if self.num_timesteps > (self.num_saves + 1) * self.save_freq:
            self.num_saves += 1
            self.model.save(self.save_file)

            normalized_env = unwrap_vec_normalize(self.model.env)
            if normalized_env is not None:
                normalized_env.save(self.save_file_stats)
            
            if self.verbose > 0:
                print(f"Saving latest model to {self.save_file}")

        return True


def get_free_cuda_gpus(max_count: int, force_cpu=False, n_giga_needed=1):
    """
    Returns a list of torch devices for gpus with free memory
    """
    gpus_with_free_mem = []
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available() and not force_cpu:
            for _ in range(0, max_count):
                device = torch.device("mps")
                gpus_with_free_mem.append(device)
        else:
            for _ in range(0, max_count):
                device = torch.device("cpu")
                gpus_with_free_mem.append(device)
        return gpus_with_free_mem
    elif force_cpu:
        for _ in range(0, max_count):
            device = torch.device("cpu")
            gpus_with_free_mem.append(device)
        return gpus_with_free_mem        

    # Initialize NVML
    pynvml.nvmlInit()
    device_count = torch.cuda.device_count()
    gpu_indices = list(range(device_count))
    if len(gpu_indices) == 1: 
        # we have just one gpu
        device = torch.device(f"cuda:{0}")
        gpus_with_free_mem.append(device)
        return gpus_with_free_mem
    
    for i in gpu_indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        free_mem = int(meminfo.free)
        # a bit of heuristic but will pretend that
        # any gpu that's using less than 0.5 GiB is
        # free
        gigabyte = int(1024**3)
        if free_mem < n_giga_needed * gigabyte:
            device = torch.device(f"cuda:{i}")
            gpus_with_free_mem.append(device)
        if len(gpus_with_free_mem) >= max_count:
            break
    
    if len(gpus_with_free_mem) == 0:
        print("Warning no GPUs meet the free mem requirement. Trying all.")
        all_gpus = []
        for i in gpu_indices:
            device = torch.device(f"cuda:{i}")
            all_gpus.append(device)
        return all_gpus
    else:
        return gpus_with_free_mem


def select_free_gpu_or_fallback(never_mps=False):
    """
    Selects MPS on Arm Macs, on CUDA systems the GPU with the most free memory.
    Returns the device as a torch.device object.
    """
    device = torch.device("cpu")
    if torch.backends.mps.is_available() and not never_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        # we have just one gpu
        if device_count == 1: 
            device = torch.device(f"cuda:{0}")
            return device
    
        # okay more than one GPU
        pynvml.nvmlInit()

        device_count = torch.cuda.device_count()
        gpu_indices = list(range(device_count))
        # random.shuffle(gpu_indices)  # Randomize the order of GPU checks

        best_gpu = None
        max_free_mem = 0

        for i in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = int(meminfo.free)
            if free_mem > max_free_mem:
                best_gpu = i
                max_free_mem = free_mem

        if best_gpu is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{best_gpu}")

    print(f"Selected device: {device}")
    return device


def post_process_video(input_path: str, output_path: str, scale_factor: int = 4):
    """
    Loads a video, scales it up using nearest-neighbor (pixelated),
    and saves the result.
    """
    try:
        clip = VideoFileClip(input_path)

        # Scale the clip
        # interp="nearest" is crucial for the pixel-art look
        scaled_clip = clip.resized(width=clip.w * scale_factor)

        # Write the new video file
        scaled_clip.write_videofile(output_path, logger=None)

        clip.close()
        scaled_clip.close()
        print(f"Scaled video saved to {output_path}")

    except Exception as e:
        print(f"\nError during video post-processing: {e}")
        print("Please ensure 'moviepy' is installed (`pip install moviepy`)")
        print("And that 'ffmpeg' is available on your system.")


def flatten_dict(d, parent_key="", sep="."):
    """
    Flattens a nested dictionary, joining keys with a separator.
    e.g., {'train': {'algo': 'PPO'}} -> {'train.algo': 'PPO'}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_hyper_parameters(logger, config):

    # Flatten the config
    flat_config = flatten_dict(config)

    # Filter for simple types and convert to string for add_hparams
    for k, v in flat_config.items():
        logger.record(f"hyperparameters/{k}", v)

    logger.dump(step=0)
    print("Logged hyperparameters to TensorBoard HParams tab.")


def unwrap_wrapper_diy(env: gym.Env, wrapper_class: type[gym.Wrapper]):
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while env_tmp is not None:
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None