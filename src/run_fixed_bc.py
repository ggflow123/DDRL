import os
import numpy as np
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
from behavioral_cloning import train_student_fixed_behavioral_cloning_episodic_offline
from evaluation import EpisodicEvaluator, VecEvaluator
import torch
from stable_baselines3.common.evaluation import evaluate_policy
import hydra
from omegaconf import DictConfig, OmegaConf
from file_management import get_class

from load_offline_data import data_distill_loader, entire_loader, bc_loader

CONFIG_PATH = "../configs"
CONFIG_NAME = "main_bc"
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

    percentage = cfg.percentage
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

    data_path = os.path.join(data_path, f"{env_name}-offlinedata.pkl")
    states_offline, actions_offline = bc_loader(data_path, percentage, env, device)
    num_samples = states_offline.size(0)  # Assuming the first dimension is the batch size
    sample_fraction = 0.20  # 20 percent as a decimal
    num_samples_to_select = int(num_samples * sample_fraction)  # Number of samples to select
    
    # Generate random indices
    indices = torch.randperm(num_samples)[:num_samples_to_select]
    
    # Index the tensors to get the sampled subset
    states_offline = states_offline[indices]
    actions_offline = actions_offline[indices]
    # configure teacher
    teacher_path = os.path.join(model_path, cfg.distill.teacher_path) # teacher path should be model path + cfg.distill.teacher_path 
    teacher_class_path = cfg.distill.teacher_class
    teacher_class = get_class(teacher_class_path)
    teacher: Teacher = teacher_class()
    teacher.load_model(path=teacher_path)
    teacher_save_name = teacher_path.rsplit("/", 1)[-1] # save name should be final part of teacher_path

    id_mean_list = []
    id_std_list = []
    id_rewards_list = []
    ood_mean_list = []
    ood_std_list = []
    ood_rewards_list = []
    for i in range(10):
        save_names = []
        save_names.append(f"bc-{percentage}-{i}-student")
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
        train_student_fixed_behavioral_cloning_episodic_offline(student=student, teacher=teacher, env=env, loss_function=loss_function, data_path=data_path, states_offline=states_offline, actions_offline=actions_offline, percentage=percentage, optimizer=optimizer, rollout_buffer=rollout_buffer, synthetic_buffer=trainable_buffer, student_train_epochs=cfg.student_train_epochs, student_update_freq=cfg.student_update_freq, device=device, **distill_kwargs)
        save_names.append(cfg.distill.save_name)

        id_student_rewards = evaluator.evaluate(student, env, num_episodes=100)
        id_student_mean = id_student_rewards.mean()
        id_student_std = id_student_rewards.std()
        print(f"ID Student Mean: {id_student_mean}")
        print(f"ID Student Std: {id_student_std}")


        ood_student_rewards = evaluator.evaluate(student, evalenv, num_episodes=100)
        ood_student_mean = ood_student_rewards.mean()
        ood_student_std = ood_student_rewards.std()


        print(f"OOD Student Mean: {ood_student_mean}")
        print(f"OOD Student Std: {ood_student_std}") 
        id_mean_list.append(id_student_mean)
        id_std_list.append(id_student_std)
        id_rewards_list.extend(id_student_rewards)
        ood_mean_list.append(ood_student_mean)
        ood_std_list.append(ood_student_std)
        ood_rewards_list.append(ood_student_rewards)
        



        #teacher_rewards = evaluator.evaluate(teacher, env, num_episodes=10)
        #teacher_mean = teacher_rewards.mean()
        #teacher_std = teacher_rewards.std()
        #print(f"Teacher Mean: {teacher_mean}")
        #print(f"Teacher Std: {teacher_std}")
        

        # save results
        save_name = "__".join(save_names)

        model_dir = cfg.model_dir
        os.makedirs(model_dir, exist_ok=True)
        # save student model to ./model_dir/student-weights_save_name.pt
        # torch.save(student.base_model.state_dict(
        # ), f"{model_dir}/student-weights__{save_name}.pt")

        # save student model to ./model_dir/student-model_save_name.pt
        student.save_model(f"./{model_dir}/student-model__{save_name}.pt")
        trainable_buffer.save(f"./{model_dir}/trainable-buffer-student__{save_name}.pt")
        rollout_buffer.save(f"./{model_dir}/rollout-buffer-student__{save_name}.pt")

        results_dir = cfg.results_dir
        os.makedirs(results_dir, exist_ok=True)
    # save distill_rewards to ./results_dir/student-distill-rewards_save_name.pt as a csv
# Convert lists to NumPy arrays
    id_mean_list = np.array(id_mean_list)
    id_std_list = np.array(id_std_list)
    id_rewards_list = np.array(id_rewards_list)
    ood_mean_list = np.array(ood_mean_list)
    ood_std_list = np.array(ood_std_list)
    ood_rewards_list = np.array(ood_rewards_list)
    print("Final student IID Mean: ", id_mean_list.mean())
    print("Final student IID STD: ", id_std_list.mean())
    print("Final student OOD Mean: ", ood_mean_list.mean())
    print("Final student OOD STD: ", ood_std_list.mean())
 
    id_rewards_list.tofile(f"./{results_dir}/ID-student-distill-rewards__{save_name}.csv", sep=",", format="%s")
    ood_rewards_list.tofile(f"./{results_dir}/OOD-student-distill-rewards__{save_name}.csv", sep=",", format="%s")

if __name__ == "__main__":
    # print(os.path.abspath(CONFIG_PATH))
    main()
