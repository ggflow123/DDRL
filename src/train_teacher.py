import gym
import torch
import datetime
import numpy as np
import os

from gymnasium.envs.registration import register
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv, VecTransposeImage, VecExtractDictObs, VecFrameStack, VecNormalize
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same')

    def forward(self, x):
        residual = x
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + residual

class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSequence, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')
        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.max_pool(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        return out

class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ImpalaCNN, self).__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]
        depths = [16, 32, 32]

        self.policy_conv_sequences = nn.ModuleList()
        self.value_conv_sequences = nn.ModuleList()
        current_channels = input_channels
        for depth in depths:
            self.policy_conv_sequences.append(ConvSequence(current_channels, depth))
            self.value_conv_sequences.append(ConvSequence(current_channels, depth))
            current_channels = depth

        # Assume the final spatial size is 11x11 based on the convolutions and pooling
        self.policy_flatten = nn.Flatten()
        self.value_flatten = nn.Flatten()
        self.policy_fc = nn.Linear(current_channels * 11 * 11, features_dim)
        self.value_fc = nn.Linear(current_channels * 11 * 11, features_dim)

    def forward(self, observations):
        out = observations.float() / 255.0  # Normalize the images
        #out = features
        for conv_seq in self.conv_sequences:
            out = conv_seq(out)
        out = self.flatten(out)
        return F.relu(self.fc(out))



#class CustomActorCriticPolicy(ActorCriticPolicy):
#    def _build_mlp_extractor(self):
#        self.mlp_extractor = ImpalaCNN(self.observation_space)

class EpisodeRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > len(self.episode_rewards):
            info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(info['r'])
            self.episode_lengths.append(info['l'])
        return True


CONFIG_PATH = "../configs"
CONFIG_NAME = "train_teacher"
HYDRA_FULL_ERROR = 1

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #==================================
    total_timesteps = cfg.total_timesteps # 200 Million for hard, 25 M for easy
    eval_freq = cfg.eval_freq
    save_freq = cfg.save_freq
    env_name = cfg.env.env_name
    start_level = cfg.env.start_level
    num_levels = cfg.env.num_levels
    distribution_mode = cfg.env.distribution_mode
    ckpts_dir = cfg.ckpts_dir # Path to save checkpoints
    plot_dir = cfg.plot_dir # Path to save reward plot
    model_dir = cfg.model_dir # Path to save models
    
    features_extractor = cfg.features_extractor # Which feature extractor to deploy, only IMPALA CNN available for custom features extractor
    policy_net = cfg.policy_net
    #==================================

    curr_save_dir = os.path.join(os.getcwd(), model_dir)
    print(f"Working Dir: {os.getcwd()}")
    dir_name = now + env_name + "_start_level_" + str(start_level) + "_#levels_" + str(num_levels) + "_mode_" + distribution_mode
    checkpoint_callback = CheckpointCallback(
      save_freq=save_freq,
      save_path=f"./{ckpts_dir}",
      name_prefix=f"ppo_{features_extractor}_{policy_net}_{env_name}_start-{start_level}_num-levels-{num_levels}_mode-{distribution_mode}_model",
      save_replay_buffer=True,
      save_vecnormalize=True,
    )
    # Initialize your custom callback
    reward_logging_callback = EpisodeRewardCallback()


    venv = gym.make(env_name, apply_api_compatibility=True, start_level=start_level, num_levels=num_levels, distribution_mode=distribution_mode)
    eval_venv = gym.make(env_name, apply_api_compatibility=True, start_level=start_level, num_levels=0, distribution_mode=distribution_mode)
    #eval_env = VecTransposeImage(DummyVecEnv([lambda: gym.make(env_name, apply_api_compatibility=True, start_level=start_level, num_levels=num_levels, distribution_mode=distribution_mode)]))
    #venv = ProcgenEnv(num_envs=64, env_name="bigfish", num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    #venv = VecExtractDictObs(env, "rgb")

    #venv = VecMonitor(
    #    venv=venv, filename=None
    #)

    #venv = VecNormalize(venv=venv)
    #venv = 
    eval_callback = EvalCallback(eval_venv, best_model_save_path=f"./{ckpts_dir}/best_model",
                                 log_path=f"./logs/", eval_freq=eval_freq, deterministic=False)
    #Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback, reward_logging_callback])

    # Create the PPO model
    if features_extractor == "Impala":
        policy_kwargs = dict(
    features_extractor_class=ImpalaCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
        model = PPO(policy_net, venv, policy_kwargs=policy_kwargs, verbose=1, batch_size=8, clip_range=0.2, clip_range_vf=0.2, ent_coef=0.01, gae_lambda=0.95, gamma=0.999, learning_rate=0.0005, n_epochs=3, n_steps=256, vf_coef=0.5, device=device)
    else:  
        model = PPO(policy_net, venv, verbose=1, batch_size=8, clip_range=0.2, clip_range_vf=0.2, ent_coef=0.01, gae_lambda=0.95, gamma=0.999, learning_rate=0.0005, n_epochs=3, n_steps=256, vf_coef=0.5, device=device)
    #Train the model
    if cfg.resume_path != None:
        print("Loading model: ", cfg.resume_path)
        model.load(cfg.resume_path, print_system_info=True, force_reset=False)
    
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"./{model_dir}/{dir_name}_PPO_{features_extractor}_{policy_net}_{env_name}_model"
    #model_replaybuffer_save_path = f"./{model_dir}/{dir_name}_PPO_{features_extractor}_{policy_net}_{env_name}_replaybuffer"
    model.save(model_save_path)
    #model.save_replay_buffer(model_replaybuffer_save_path)
    #print(np.arange(1, len(reward_logging_callback.episode_rewards)+1, 1, dtype=int))
    # Plotting
    eps = np.arange(1, len(reward_logging_callback.episode_rewards)+1, 1, dtype=int)

    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(eps, reward_logging_callback.episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards Over Episodes')
    # Save the plot to a file
    plt.savefig(f'./{plot_dir}/{dir_name}_episode_rewards_plot.png', dpi=300)

if __name__ == "__main__":
    main()

