import gym
import torch
import datetime

from gymnasium.envs.registration import register
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor  # , VecExtractDictObs
from stable_baselines3.common.env_checker import check_env

# venv = ProcgenEnv(num_envs=1, env_name="coinrun")

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
seed = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_name = now + "_starpilot_" + "_seed_" + str(seed)
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path=f"./logs/{dir_name}",
  name_prefix="ppo_heist_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

# env = ProcgenGym3Env(num=1, env_name="heist", start_level=seed, num_levels=1, distribution_mode="easy")

eval_env = gym.make("procgen-starpilot-v0",
               apply_api_compatibility=True, start_level=seed, num_levels=1, distribution_mode="easy")
eval_callback = EvalCallback(eval_env, best_model_save_path=f"./logs/{dir_name}/best_model",
                             log_path=f"./logs/{dir_name}/results", eval_freq=5000)
#Create the callback list
callback = CallbackList([checkpoint_callback, eval_callback])

env = gym.make("procgen-starpilot-v0",
               apply_api_compatibility=True, start_level=seed, num_levels=1, distribution_mode="easy")



# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1, batch_size=2048, clip_range=0.2, clip_range_vf=0.2,
            ent_coef=0.01, gae_lambda=0.95, gamma=0.999, learning_rate=0.0005, n_epochs=3, n_steps=256,
            vf_coef=0.5, device=device)

# Train the model
total_timesteps = 25000000 # 25 Million
model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

# Save the model
# model.save("ppo_starpilot")

# Test the trained model
# obs, _ = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info, _ = env.step(action)
#     print(rewards)
#     env.render()
