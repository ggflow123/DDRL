import gymnasium as gymnasium
import gym
from wrappers import Teacher, Student
from loss_functions import LossFunction
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from load_offline_data import data_distill_loader, entire_loader, bc_loader
from buffer import RolloutBufferBatch, TrainableSyntheticBufferBatch



# class TensorPairsDataset(Dataset):
#     def __init__(self, x_list, y_list):
#         self.x_list = x_list
#         self.y_list = y_list
    
#     def __len__(self):
#         return len(self.x_list)
    
#     def __getitem__(self, index):
#         return self.x_list[index], self.y_list[index]            

# def concat_state_and_teacher_knowledge(state, teacher_knowledge):
#     """
#     Helper function to convert state and teacher knowledge into tensors
#     NOTE: Used in buffer_to_tensor()
#     """
#     state = torch.tensor(state)
#     teacher_knowledge = torch.flatten(torch.tensor(teacher_knowledge))

#     return torch.cat((state, teacher_knowledge), dim=0)
def match_loss(gw_syn, gw_real, device):

    dis = torch.tensor(0.0).to(device)

    # only MSE implemented for now
    gw_real_vec = []
    gw_syn_vec = []
    for ig in range(len(gw_real)):
        gw_real_vec.append(gw_real[ig].reshape((-1)))
        gw_syn_vec.append(gw_syn[ig].reshape((-1)))
    gw_real_vec = torch.cat(gw_real_vec, dim=0)
    gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
    dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    return dis


def get_batch(states, actions, batch_size):
    indices = torch.randperm(states.size(0))[:batch_size]
    states_batch = states[indices]
    actions_batch = actions[indices]
    return states_batch, actions_batch

def train_student_behavioral_cloning_episodic_offline(student: Student, teacher: Teacher, env: gymnasium.Env or gym, loss_function: LossFunction, data_path, percentage=1.0, num_episodes=100, optimizer=None, online_network="student", max_steps=None, student_train_epochs=10, student_update_freq=16, rollout_buffer=None, synthetic_buffer=None, manual_init=True, rng=np.random.default_rng(), batch_size=1, device='cpu'):
    '''
    Trains the student model using the teacher model

    Args:
        student (Student): Student model
        teacher (Teacher): Teacher model
        env (gym.Env): Environment
        loss_function (LossFunction): Loss function
        num_episodes (int, optional): Number of episodes to train the student. Defaults to 1000.
        optimizer (torch.optim, optional): Optimizer to use for training. Defaults to None.
        online_network (int, optional): Whether to use an online network for training. If None, the student model is used as the online network. Defaults to None.

    Returns:
        rewards (np.array): Array of rewards for each episode during training
    '''

    if rollout_buffer is None:
        rollout_buffer = RolloutBufferBatch(1, rng=rng)
    
    if synthetic_buffer is None:
        synthetic_buffer = TrainableSyntheticBufferBatch(synthetic_buffer_size=100, synthetic_init_threshold_size=1000, rng=rng)

    student.base_model.train()
    student_params = list(student.base_model.parameters())
    rewards = np.zeros(num_episodes)
    episode = 0
    
    states_offline, actions_offline = bc_loader(data_path, percentage, env, device)
    total_size = synthetic_buffer.synthetic_buffer_size
    #synthetic_data = torch.nn.ParameterList([state_param, ])
    #optimizer_states_syn = torch.optim.Adam(synthetic_data, lr=0.01) # TODO: Try different optim.
    #optimizer_states_syn = torch.optim.SGD(synthetic_data, lr=0.1, momentum=0.5) # TODO: Try different optim.
    #optimizer_states_syn.zero_grad()
                    #synthetic_buffer.is_init_optimizer = False
    for episode in range(num_episodes):
        print("Running Episode ", episode)
        #state, _ = env.reset()
        #state = env.reset()
        #state = torch.tensor(state, dtype=torch.float32)

        #done = False
        #steps = 0
        # place state and teacher knowledge in buffer
        #print("steps: ", steps)
        #print(f"state: ", state.min(), state.max(), state.mean())
        #print(f"teacher knowledge: {teacher_knowledge}")

        states, actions = get_batch(states_offline, actions_offline, batch_size) 
        student.base_model.train()
        student_knowledge_training = student.get_knowledge(states)
        #print("syn student knowledge in training: " ,syn_student_knowledge_training.grad)
        training_loss = loss_function.loss(student_knowledge_training, actions)

        # optimize
        optimizer.zero_grad() # student optimizer
        training_loss.backward()
        optimizer.step()
                    #state, reward, term, trunc, _ = env.step(action)
    return


def train_student_fixed_behavioral_cloning_episodic_offline(student: Student, teacher: Teacher, env: gymnasium.Env or gym, loss_function: LossFunction, data_path, states_offline, actions_offline, percentage=1.0, num_episodes=100, optimizer=None, online_network="student", max_steps=None, student_train_epochs=10, student_update_freq=16, rollout_buffer=None, synthetic_buffer=None, manual_init=True, rng=np.random.default_rng(), batch_size=1, device='cpu'):
    '''
    Trains the student model using the teacher model

    Args:
        student (Student): Student model
        teacher (Teacher): Teacher model
        env (gym.Env): Environment
        loss_function (LossFunction): Loss function
        num_episodes (int, optional): Number of episodes to train the student. Defaults to 1000.
        optimizer (torch.optim, optional): Optimizer to use for training. Defaults to None.
        online_network (int, optional): Whether to use an online network for training. If None, the student model is used as the online network. Defaults to None.

    Returns:
        rewards (np.array): Array of rewards for each episode during training
    '''

    if rollout_buffer is None:
        rollout_buffer = RolloutBufferBatch(1, rng=rng)
    
    if synthetic_buffer is None:
        synthetic_buffer = TrainableSyntheticBufferBatch(synthetic_buffer_size=100, synthetic_init_threshold_size=1000, rng=rng)

    student.base_model.train()
    student_params = list(student.base_model.parameters())
    rewards = np.zeros(num_episodes)
    episode = 0
    
    #states_offline, actions_offline = bc_loader(data_path, percentage, env, device)
    total_size = synthetic_buffer.synthetic_buffer_size
    #synthetic_data = torch.nn.ParameterList([state_param, ])
    #optimizer_states_syn = torch.optim.Adam(synthetic_data, lr=0.01) # TODO: Try different optim.
    #optimizer_states_syn = torch.optim.SGD(synthetic_data, lr=0.1, momentum=0.5) # TODO: Try different optim.
    #optimizer_states_syn.zero_grad()
                    #synthetic_buffer.is_init_optimizer = False
    for episode in range(num_episodes):
        print("Running Episode ", episode)
        #state, _ = env.reset()
        #state = env.reset()
        #state = torch.tensor(state, dtype=torch.float32)

        #done = False
        #steps = 0
        # place state and teacher knowledge in buffer
        #print("steps: ", steps)
        #print(f"state: ", state.min(), state.max(), state.mean())
        #print(f"teacher knowledge: {teacher_knowledge}")

        states, actions = get_batch(states_offline, actions_offline, batch_size) 
        student.base_model.train()
        student_knowledge_training = student.get_knowledge(states)
        #print("syn student knowledge in training: " ,syn_student_knowledge_training.grad)
        training_loss = loss_function.loss(student_knowledge_training, actions)

        # optimize
        optimizer.zero_grad() # student optimizer
        training_loss.backward()
        optimizer.step()
                    #state, reward, term, trunc, _ = env.step(action)
    return
