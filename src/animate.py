import gym
import torch
import datetime
import numpy as np
import os

from gymnasium.envs.registration import register
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor  # , VecExtractDictObs
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

from students import Student, PolicyStudent

env = gym.make("procgen-heist-v0",
               apply_api_compatibility=True, start_level=2, num_levels=1, distribution_mode="easy", render=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyStudent()
model.load_model(
    "student-model__heist3__2023-12-07T13-43-37procgen-heist-v0_start_level_3_levels_1_mode_easy_ppo_heist_model__policy-student__mlp0__MAE__adam0__exp1.pt")
# model.learn(total_timesteps=100, progress_bar=True)

obs, _ = env.reset()
while True:
    action = model.step(obs)
    obs, rewards, dones, info, _ = env.step(action)
