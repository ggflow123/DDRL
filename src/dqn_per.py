# Adapted from https://github.com/Howuhh/prioritized_experience_replay/blob/main/train.py

import gym
import torch
import datetime
import numpy as np
import os
from buffer import ReplayBuffer, PrioritizedReplayBuffer
from copy import deepcopy
import random

from gymnasium.envs.registration import register
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv

from stable_baselines3 import PPO
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
import torch.optim as optim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class BatchReshape(nn.Module):
    def __init__(self):
        super(BatchReshape, self).__init__()

    def forward(self, x):
        # Reshape the tensor to have the same batch demension but flatten the remaining dimensions
        return x.reshape(x.shape[0], -1)


class DQN:
    def __init__(self, state_size, action_size, gamma, tau, lr, device):
        self.device = device
        #layers.append(nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding='same', stride=2))
        #layers.append(nn.Conv2d(32, 64, kernel_size=kernel_size, padding='same', stride=2))
        #layers.append(nn.Conv2d(64, 32, kernel_size=kernel_size, padding='same', stride=2))
        #layers.append(nn.Conv2d(32, 1input_channels, kernel_size=kernel_size, padding='same', stride=2))
        #self.model = nn.Sequential(
        #    nn.Linear(np.prod(state_size), 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, action_size)
        #).to(self.device)
        # ============== Same CNN as the distillation part, harding coding for DQN only ============
        channel_multiplier = 4
        input_channels = state_size[2]
        input_grid_size = state_size[0]
        final_size = input_channels * input_grid_size*input_grid_size//(64*4)
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, input_channels * channel_multiplier, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(input_channels * channel_multiplier, input_channels * channel_multiplier * channel_multiplier, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(input_channels * channel_multiplier * channel_multiplier, input_channels * channel_multiplier, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(input_channels * channel_multiplier, input_channels, 3, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            BatchReshape(),
            nn.Linear(final_size, action_size)
        ).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        with torch.no_grad():
            #state = state.reshape(state.shape[0], np.prod(state.shape[1:])).to(self.device)
            state = state.permute(0, 3, 1, 2).to(self.device)
            action_probs = torch.softmax(self.model(state), dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze().item()
            #action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch

        #print(next_state.shape)
        #print(state.shape)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        if len(next_state.shape) == 3:
            next_state = next_state.unsqueeze(0)
        #state = torch.as_tensor(state, dtype=torch.float).reshape(state.shape[0], np.prod(state.shape[1:])).to(self.device)
        state = state.permute(0, 3, 1, 2).to(self.device)
        #next_state = torch.as_tensor(next_state, dtype=torch.float).reshape(next_state.shape[0], np.prod(next_state.shape[1:])).to(self.device)
        next_state = next_state.permute(0, 3, 1, 2).to(self.device)
        # ============= Q_next and Q_target ====================
        #print(self.target_model(next_state).shape)
        action_probs_qnext = torch.softmax(self.target_model(next_state), dim=-1)
        #print(torch.multinomial(action_probs_qnext, 1).squeeze().shape)
        #action_qnext = torch.multinomial(action_probs_qnext, 1).squeeze()
        Q_next = action_probs_qnext.max(dim=1).values
        Q_target = reward.to(self.device) + self.gamma * (1 - done.to(self.device)) * Q_next
        #print(self.model(state).shape)
        #print(action.shape)
        #print(action.argmax(dim=1).shape)
        batch_indices = torch.arange(action.shape[0]).unsqueeze(1).expand(-1, action.shape[1])  # Expand to match action_indices shape
        # ============ Q ==================
        action_probs_q = torch.softmax(self.model(state), dim=-1)
        #action_q = torch.multinomial(action_probs_q, 1).squeeze()
        Q = action_probs_q[torch.arange(action.shape[0]), action.to(torch.long).argmax(dim=1)]

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target.to(self.device))**2 * weights.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update(self.target_model, self.model)

        return loss.item(), td_error

    def save(self, model_save_path):
        torch.save(self.model, f"{model_save_path}.pkl")

def evaluate_policy(env, agent, episodes=5, seed=0):
    returns = []
    for ep in range(episodes):
        done, total_reward = False, 0
        state, _ = env.reset()

        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns), np.std(returns)

def train(env, model, buffer, num_episodes, batch_size=128, eps_max=0.1, eps_min=0.0, test_every=1000, seed=0, max_steps=None, model_save_path="dqn_per_agent"):
    print(f"Training started.")

    rewards_total, stds_total = [], []
    loss_count, total_loss = 0, 0

    episodes = 0
    best_reward = -np.inf

    timesteps = 0
    rewards = np.zeros(num_episodes)
    for episode in range(num_episodes):
        print(episode)
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        done = False
        steps = 0

        while not done:
            timesteps += 1
            eps = eps_max - (eps_max - eps_min) * timesteps / (num_episodes * 1000) # 1000 is the number of environment truncation 

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = model.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            rewards[episode] += reward
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True
            
            buffer.add((state, action, reward, next_state, int(done)))


            if timesteps > batch_size:
                if isinstance(buffer, ReplayBuffer):
                    batch = buffer.sample(batch_size)
                    loss, td_error = model.update(batch)
                elif isinstance(buffer, PrioritizedReplayBuffer):
                    batch, weights, tree_idxs = buffer.sample(batch_size)
                    loss, td_error = model.update(batch, weights=weights)

                    buffer.update_priorities(tree_idxs, td_error.cpu().numpy())
                else:
                    raise RuntimeError("Unknown buffer")

                total_loss += loss
                loss_count += 1

                if timesteps % test_every == 0:
                    mean, std = evaluate_policy(env, model, episodes=10)

                    print(f"Episode: {episode}, Step: {timesteps}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps}")

                    if mean > best_reward:
                        best_reward = mean
                        model.save(model_save_path)

                    rewards_total.append(mean)
                    stds_total.append(std)
                

    return rewards


CONFIG_PATH = "../configs"
CONFIG_NAME = "student_per"
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
    policy_net = cfg.policy_net

    distill_kwargs = cfg.distill.distill_kwargs
    #==================================

    curr_save_dir = os.path.join(os.getcwd(), model_dir)
    print(f"Working Dir: {os.getcwd()}")
    dir_name = now + env_name + "_start_level_" + str(start_level) + "_#levels_" + str(num_levels) + "_mode_" + distribution_mode


    venv = gym.make(env_name, apply_api_compatibility=True, start_level=start_level, num_levels=num_levels, distribution_mode=distribution_mode)
    eval_venv = gym.make(env_name, apply_api_compatibility=True, start_level=start_level, num_levels=0, distribution_mode=distribution_mode)
    #Create the callback list
    state_shape = venv.observation_space.shape
    action_size = venv.action_space.n
    

    buffer = PrioritizedReplayBuffer(state_size=state_shape, action_size=action_size, buffer_size=100000, alpha=0.7, beta=0.4)
    model = DQN(state_size=state_shape, action_size=action_size, gamma=0.99, tau=0.01, lr=1e-4, device=device)
    rewards = train(env=venv, model=model, buffer=buffer, num_episodes=cfg.distill.num_episodes, batch_size=256, eps_max=0.1, eps_min=0.0, test_every=5000, seed=0, max_steps=None) 
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = f"./{model_dir}/{dir_name}_dqn_{policy_net}_{env_name}_model"
    #model_replaybuffer_save_path = f"./{model_dir}/{dir_name}_PPO_{features_extractor}_{policy_net}_{env_name}_replaybuffer"

    model.save(model_save_path)
    #model.save_replay_buffer(model_replaybuffer_save_path)
    #print(np.arange(1, len(reward_logging_callback.episode_rewards)+1, 1, dtype=int))
    # Plotting
    #eps = np.arange(1, len(reward_logging_callback.episode_rewards)+1, 1, dtype=int)

    #os.makedirs(plot_dir, exist_ok=True)
    #plt.figure(figsize=(12, 6))
    #plt.plot(eps, reward_logging_callback.episode_rewards)
    #plt.xlabel('Episodes')
    #plt.ylabel('Rewards')
    #plt.title('Rewards Over Episodes')
    ## Save the plot to a file
    #plt.savefig(f'./{plot_dir}/{dir_name}_episode_rewards_plot.png', dpi=300)

if __name__ == "__main__":
    main()

