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
from data_distillation_training_offline import train_data_episodic_offline

from teacher import Teacher
from students import Student
from loss_functions import LossFunction
from data_distillation_training_offline import train_student_data_distillation_episodic_offline
from evaluation import EpisodicEvaluator, VecEvaluator
import torch
from stable_baselines3.common.evaluation import evaluate_policy
import hydra
from omegaconf import DictConfig, OmegaConf
from file_management import get_class


CONFIG_PATH = "../configs"
CONFIG_NAME = "main_ds"
HYDRA_FULL_ERROR = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
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
    save_names = []
    env = VecExtractDictObs(env, "rgb")
    env = VecMonitor(venv=env, filename=None, keep_buf=100)
    env = VecNormalize(venv=env, ob=False)


    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    save_names.append(cfg.env.save_name)

    num_of_data = cfg.trainablebuffer.buffer_kwargs.synthetic_buffer_size
    data_loader = DataLoaderPickle(f"{data_path}/{env_name}-offlinedata.pkl")
    #states_path = os.path.join(data_path, f"{env_name}-{num_of_data}-distilled-states.pkl")
    states_path = f"{data_path}/{env_name}-{num_of_data}-distilled-states.pkl"
    #teacher_knowledge_path = os.path.join(data_path, f"{env_name}-{num_of_data}-distilled-actions.pkl")
    teacher_knowledge_path = f"{data_path}/{env_name}-{num_of_data}-distilled-actions.pkl"
    save_states_name = f"{data_path}/{env_name}-{num_of_data}-trained-distilled-states.pkl"
    save_action_name = f"{data_path}/{env_name}-{num_of_data}-trained-distilled-actions.pkl"

    data_path = os.path.join(data_path, f"{env_name}-offlinedata.pkl")
    #states_syn, teacher_knowledge_syn = data_distill_loader(data_path, total_size, env, device)
    with open(states_path, 'rb') as f:
        states_syn = pickle.load(f)

    with open(teacher_knowledge_path, 'rb') as f:
        teacher_knowledge_syn = pickle.load(f)
    states_syn = states_syn.to(device)
    teacher_knowledge_syn = teacher_knowledge_syn.to(device)
    #states_syn, teacher_knowledge_syn = data_distill_loader(f"{data_path}/{env_name}-offlinedata.pkl", cfg.data_size, env, device='cpu')

    # configure teacher
    teacher_path = os.path.join(model_path, cfg.distill.teacher_path) # teacher path should be model path + cfg.distill.teacher_path
    teacher_class_path = cfg.distill.teacher_class
    teacher_class = get_class(teacher_class_path)
    teacher: Teacher = teacher_class()
    teacher.load_model(path=teacher_path)
    teacher_save_name = teacher_path.rsplit("/", 1)[-1] # save name should be final part of teacher_path
    # configure student
    student_class_path = cfg.student.student_class
    student_class: Student = get_class(student_class_path)
    save_names.append(cfg.student.save_name)

    # configure base model
    student_base_model_class_path = cfg.student.base_model.model_class
    base_model_class = get_class(student_base_model_class_path)
    base_model_kwargs = cfg.student.base_model.model_kwargs
    save_names.append(cfg.student.base_model.save_name)

    base_model = base_model_class(input_shape=state_shape, output_shape=tuple([action_size]), **base_model_kwargs)
    student = student_class(state_shape=state_shape, action_size=action_size, base_model=base_model, device=device)
    if cfg.student_save_path != None:
        student_load_path = os.path.join(model_path, cfg.student_save_path)
        student.load_model_training(student_load_path)

    # configure loss function
    loss_function_class_path = cfg.loss.loss_class
    loss_function_class: LossFunction = get_class(loss_function_class_path)
    loss_function = loss_function_class()
    save_names.append(cfg.loss.save_name)

    # configure optimizer
    network = student.base_model
    optimizer_class_path = cfg.optimizer.optimizer_class
    optimizer_class = get_class(optimizer_class_path)
    optimizer_kwargs = cfg.optimizer.optimizer_kwargs
    optimizer = optimizer_class(network.parameters(), **optimizer_kwargs)
    save_names.append(cfg.optimizer.save_name)

    # configure rollout buffer
    rollout_buffer_class_path = cfg.buffer.buffer_class
    rollout_buffer_class = get_class(rollout_buffer_class_path)
    rollout_buffer_kwargs = cfg.buffer.buffer_kwargs
    knowledge_shape = (action_size, )
    rollout_buffer = rollout_buffer_class(state_dim=state_shape, knowledge_dim=knowledge_shape, device=device, **rollout_buffer_kwargs)
    save_names.append(cfg.buffer.save_name)

     # configure trainable synthetic buffer
    trainable_buffer_class_path = cfg.trainablebuffer.buffer_class
    trainable_buffer_class = get_class(trainable_buffer_class_path)
    trainable_buffer_kwargs = cfg.trainablebuffer.buffer_kwargs
    trainable_buffer = trainable_buffer_class(state_dim=state_shape, knowledge_dim=knowledge_shape, device=device, **trainable_buffer_kwargs)
    save_names.append(cfg.trainablebuffer.save_name)

    # configure evaluation
    evaluator = VecEvaluator()

    # configure distillation
    distill_kwargs = cfg.distill.distill_kwargs
    # Need to change the configuration
    states_syn_save, teacher_knowledge_syn_save = train_data_episodic_offline(student=student, teacher=teacher, env=env, loss_function=loss_function, data_path=data_path, states_syn=states_syn, teacher_knowledge_syn=teacher_knowledge_syn, optimizer=optimizer, rollout_buffer=rollout_buffer, synthetic_buffer=trainable_buffer, student_train_epochs=cfg.student_train_epochs, student_update_freq=cfg.student_update_freq, device=device, **distill_kwargs)
    save_names.append(cfg.distill.save_name)
    with open(save_states_name, 'wb') as f:
        pickle.dump(states_syn_save, f)
    with open(save_action_name, 'wb') as f:
        pickle.dump(teacher_knowledge_syn_save, f)

if __name__ == "__main__":
    main()



