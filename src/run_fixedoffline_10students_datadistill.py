import os

import gym

# load procgen
from gymnasium.envs.registration import register
from procgen import ProcgenGym3Env
from procgen import ProcgenEnv

from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize

from teacher import Teacher
from students import Student
from loss_functions import LossFunction
from data_distillation_training_offline import train_student_fixed_data_distillation_episodic_offline
from evaluation import EpisodicEvaluator, VecEvaluator
import torch
from stable_baselines3.common.evaluation import evaluate_policy
import hydra
from omegaconf import DictConfig, OmegaConf
from file_management import get_class
from load_offline_data import data_distill_loader
import pickle

CONFIG_PATH = "../configs"
CONFIG_NAME = "main_ds"
HYDRA_FULL_ERROR = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    '''
    Runs the distillation process using the configuration file specified in CONFIG_PATH/CONFIG_NAME, 
    and saves the results to the model_dir and results_dir specified in the configuration file.
    '''
    original_cwd = hydra.utils.get_original_cwd()
    model_path = os.path.join(original_cwd, cfg.model_dir)
    data_path = os.path.join(original_cwd, cfg.data_dir)
    
    print(f"Model Path: {model_path}") 
    print(f"Device: {device}")

    # configure environment
    env_name = cfg.env.env_name
    start_level = cfg.env.start_level
    num_levels = cfg.env.num_levels
    distribution_mode = cfg.env.distribution_mode
    #env = gym.make(env_name, apply_api_compatibility=True, start_level=start_level, 
    #               num_levels=num_levels, distribution_mode=distribution_mode)
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
    
    evalenv = ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=0,
        start_level=0,
        distribution_mode=distribution_mode,
        num_threads=1,
    )
    evalenv = VecExtractDictObs(evalenv, "rgb")
    evalenv = VecMonitor(venv=evalenv, filename=None, keep_buf=100)
    evalenv = VecNormalize(venv=evalenv, ob=False)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    states_path = os.path.join(data_path, f"{env_name}-distilled-states.pkl")
    teacher_knowledge_path = os.path.join(data_path, f"{env_name}-distilled-actions.pkl")

    data_path = os.path.join(data_path, f"{env_name}-offlinedata.pkl")
    #states_syn, teacher_knowledge_syn = data_distill_loader(data_path, total_size, env, device)
    with open(states_path, 'rb') as f:
        states_syn = pickle.load(f)

    with open(teacher_knowledge_path, 'rb') as f:
        teacher_knowledge_syn = pickle.load(f)
    # configure teacher
    states_syn = states_syn.to(device)
    teacher_knowledge_syn = teacher_knowledge_syn.to(device)

    teacher_path = os.path.join(model_path, cfg.distill.teacher_path) # teacher path should be model path + cfg.distill.teacher_path 
    teacher_class_path = cfg.distill.teacher_class
    teacher_class = get_class(teacher_class_path)
    teacher: Teacher = teacher_class()
    teacher.load_model(path=teacher_path)
    teacher_save_name = teacher_path.rsplit("/", 1)[-1] # save name should be final part of teacher_path

    save_names = []
    save_names.append(f"offline-distill-fixed-data")
    
    save_names.append(cfg.env.save_name)
    save_names.append(teacher_save_name)
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
    train_student_fixed_data_distillation_episodic_offline(student=student, teacher=teacher, env=env, loss_function=loss_function, data_path=data_path, states_syn=states_syn, teacher_knowledge_syn=teacher_knowledge_syn, optimizer=optimizer, rollout_buffer=rollout_buffer, synthetic_buffer=trainable_buffer, student_train_epochs=cfg.student_train_epochs, student_update_freq=cfg.student_update_freq, device=device, **distill_kwargs)
    save_names.append(cfg.distill.save_name)

    id_student_rewards = evaluator.evaluate(student, env, num_episodes=1000)
    id_student_mean = id_student_rewards.mean()
    id_student_std = id_student_rewards.std()
    print(f"ID Student Mean: {id_student_mean}")
    print(f"ID Student Std: {id_student_std}")


    ood_student_rewards = evaluator.evaluate(student, evalenv, num_episodes=1000)
    ood_student_mean = ood_student_rewards.mean()
    ood_student_std = ood_student_rewards.std()
    

    print(f"OOD Student Mean: {ood_student_mean}")
    print(f"OOD Student Std: {ood_student_std}")

    # teacher_rewards = evaluator.evaluate(teacher, env, num_episodes=10)
    # teacher_mean = teacher_rewards.mean()
    # teacher_std = teacher_rewards.std()
    #teacher_mean, teacher_std = evaluate_policy(teacher.model, env, n_eval_episodes=10, deterministic=False)
    #exit(0)
    teacher_rewards = evaluator.evaluate(teacher, env, num_episodes=10)
    teacher_mean = teacher_rewards.mean()
    teacher_std = teacher_rewards.std()
    print(f"Teacher Mean: {teacher_mean}")
    print(f"Teacher Std: {teacher_std}")
    

    # save results
    save_name = "__".join(save_names)

    model_dir = cfg.model_dir
    os.makedirs(model_dir, exist_ok=True)
    # save student model to ./model_dir/student-weights_save_name.pt
    # torch.save(student.base_model.state_dict(
    # ), f"{model_dir}/student-weights__{save_name}.pt")

    # save student model to ./model_dir/student-model_save_name.pt
    student.save_model(f"./{model_dir}/student-model__{save_name}.pt")

    results_dir = cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)
    # save distill_rewards to ./results_dir/student-distill-rewards_save_name.pt as a csv
    id_student_rewards.tofile(f"./{results_dir}/student-ID-distill-rewards__{save_name}.csv", sep=",", format="%s")
    ood_student_rewards.tofile(f"./{results_dir}/student-OOD-distill-rewards__{save_name}.csv", sep=",", format="%s")

if __name__ == "__main__":
    # print(os.path.abspath(CONFIG_PATH))
    main()
