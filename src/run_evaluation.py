from evaluation import Evaluator
from wrappers import Model
from file_management import get_class
import hydra
from omegaconf import DictConfig, OmegaConf
import gym
import numpy as np

from utils import setup_logging_environment

# load procgen
from gymnasium.envs.registration import register
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv
import os
import logging

import torch

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize


device = 'cuda' if torch.cuda.is_available() else 'cpu'


@hydra.main(config_path="../configs", config_name="evaluation1000")
def main(cfg: DictConfig):
    '''
    Evaluate the model specified in the configuration file on the environment specified in the configuration file.

    Saves the rewards per episode to the file specified in the configuration file.
    '''
    setup_logging_environment()

    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))
    hydra_working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Outputs for current run will be saved to {hydra_working_dir}")
    logger.info('Starting initialization')


    original_cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(original_cwd, cfg.model_dir)

    evaluator_class_path = cfg.evaluator_class
    evaluator_class: Evaluator = get_class(evaluator_class_path)
    evaluator = evaluator_class()

    # create and load model
    model_save_path = os.path.join(model_path, cfg.model_save_path) # model save path should be model_path + cfg.model_save_path
    model_class_path = cfg.model_class
    model_class: Model = get_class(model_class_path)
    model = model_class(device=device)
    model.load_model(model_save_path)
    print(f"Model Path: {model_save_path}")

    # evaluate model
    env_name = cfg.env.env_name
    start_level = cfg.env.start_level
    num_levels = cfg.env.num_levels
    distribution_mode = cfg.env.distribution_mode
    #env = gym.make(env_name, apply_api_compatibility=True, start_level=start_level, 
                   #num_levels=num_levels, distribution_mode=distribution_mode)
    venv = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        num_threads=1,
    )
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
 
    num_episodes = cfg.num_episodes
    rewards = evaluator.evaluate(model, venv, num_episodes=num_episodes)

    # create results directory if it does not exist
    results_dir = cfg.results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # save path should be last part of model_save_path  
    save_name = cfg.model_save_path.rsplit("/", 1)[-1]
    # save rewards to ./results_dir/{cfg.model_save_path}__{cfg.save_name }.pt as a csv
    rewards.tofile(f"./{results_dir}/{save_name}__{cfg.save_name}.csv", sep=",")
    print('reward mean: ', np.mean(rewards))
    print('reward std: ', np.std(rewards))


if __name__ == "__main__":
    main()
