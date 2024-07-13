from data_loader import DataLoaderPickle
import pickle
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from load_offline_data import data_distill_loader
import os
from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize

@hydra.main(config_path="../configs", config_name="offlinedata")
def main(cfg: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(original_cwd, cfg.model_dir)
    data_path = os.path.join(original_cwd, cfg.data_dir)
    env_name = cfg.env.env_name
    start_level = cfg.env.start_level
    num_levels = cfg.env.num_levels
    distribution_mode = cfg.env.distribution_mode
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
    data_loader = DataLoaderPickle(f"{data_path}/{env_name}-offlinedata.pkl")
    states_syn, teacher_knowledge_syn = data_distill_loader(f"{data_path}/{env_name}-offlinedata.pkl", cfg.data_size, env, device='cpu')
    save_states_name = os.path.join(f"{data_path}/{env_name}-{cfg.data_size}-distilled-states.pkl")
    save_action_name = os.path.join(f"{data_path}/{env_name}-{cfg.data_size}-distilled-actions.pkl")
    with open(save_states_name, 'wb') as f:
        pickle.dump(states_syn, f)
    with open(save_action_name, 'wb') as f:
        pickle.dump(teacher_knowledge_syn, f)

if __name__ == "__main__":
    main()

