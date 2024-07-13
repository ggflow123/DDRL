from data_loader import *
import os
from ppo import PPO as ProcgenPPO
from policies import ImpalaCNN
from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
import numpy as np
import json
from teacher import *
from wrappers import Model
from file_management import get_class
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_trajectory(state_list:list, action_list:list, reward_list:list, info):
    trajectory = []
    episode_length = info['l']
    for i in range(len(state_list)):
        state = state_list[i]
        action = action_list[i]
        reward = reward_list[i]
        info_i = {}
        info_i['episode_return'] = info['r']
        info_i['episode_length'] = info['l']
        trajectory.append((state, action, reward, info_i))

    # loop backwards to calculate future return
    future_return = 0
    for i, item in enumerate(reversed(trajectory)):
        future_return += item[2]
        item[3]['future_return'] = future_return
        if i < episode_length - 1:
            item[3]['next_state'] = trajectory[i + 1][0]
        else:
            item[3]['next_state'] = None
    return trajectory


@hydra.main(config_path="../configs", config_name="offlinedata")
def main(cfg: DictConfig):

    original_cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(original_cwd, cfg.model_dir)
    data_path = os.path.join(original_cwd, cfg.data_dir)
    # Configurations
    env_name = cfg.env.env_name
    start_level = cfg.env.start_level
    num_levels = cfg.env.num_levels
    distribution_mode = cfg.env.distribution_mode
    # create and load model
    model_save_path = os.path.join(model_path, cfg.model_save_path) # model save path should be model_path + cfg.model_save_path
    model_class_path = cfg.model_class
    model_class: Model = get_class(model_class_path)
    model = model_class(device=device)
    model.load_model(model_save_path)
    # Configure Env
    env = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        num_threads=1,
    )
    env = VecExtractDictObs(env, "rgb")
    env = VecMonitor(venv=env, filename=None, keep_buf=100)
    env = VecNormalize(venv=env, ob=False)
    
    num_episodes = cfg.num_episodes 
    # Initialize the environment
    #num_episodes = 100  # Number of episodes for which to collect data
    
    data_loader = DataLoaderPickle(f"{data_path}/{env_name}-offlinedata.pkl")  # Initialize the data loader
 
    for eps in range(num_episodes):
        print("episode: ", eps)
        trajectory = []
        state = env.reset()  # Reset the environment to start a new episode
        done = False
        state_list = []
        action_list = []
        reward_list = []
        
        while not done:
            state = torch.tensor(state)
            action = model.step(state)  # Get action from the trained agent
            next_state, reward, done, info = env.step(action)  # Take action in the environment
            state_list.append(state[0])
            action_list.append(action[0])
            reward_list.append(reward[0])
            # Append current step data to trajectory
            #state_list.append(state)
            #action_list.append(action)
            #reward_list.append(reward)
            #trajectory.append((state, action, reward, info))  # Add empty dict if no additional info
            
            # Update state
            state = next_state
        
        # Append final step
        if info:
            info = info[0].get('episode')
        trajectory = parse_trajectory(state_list, action_list, reward_list, info) 
        #trajectory.append((state, None, 0.0, {}))  # Convention: last action is None, last reward is 0.0
        data_loader.add_trajectory(trajectory)  # Add trajectory to DataLoader
    data_loader.save_data()  # Save collected data to file

if __name__ == "__main__":
    main()
