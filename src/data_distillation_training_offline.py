import gymnasium as gymnasium
import gym
from wrappers import Teacher, Student
from loss_functions import LossFunction
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from load_offline_data import data_distill_loader, entire_loader
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



def buffer_to_tensor(synthetic_buffer:list):
    """
    Convert the synthetic buffer into torch tensor, to prepare for optimized by torch optimizer
    NOTE: Only works when the observation is in the form of (state, teacher_knowledge), and teacher_knowledge is a tensor
    Input:
        synthetic_buffer: a list, with (state, teacher_knowledge)
    Return: 
        A tensor of stacked observations, where each observation is [state, teacher knowledge], all flattened
    """

        # Separate the states and teacher_knowledge
    states = [item[0] for item in synthetic_buffer]
    teacher_knowledges = [item[1] for item in synthetic_buffer]

    states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
    teacher_knowledges_tensors = [torch.tensor(teacher_knowledge, dtype=torch.float32, requires_grad=True) for teacher_knowledge in teacher_knowledges]

    return states_tensors, teacher_knowledges_tensors



def tensor_to_buffer(states_tensors, teacher_knowledges_tensors):
    """
    Reverses the operation of buffer_to_tensor, reconstructing the original synthetic buffer list
    from the provided state and teacher knowledge tensors.
    
    Inputs:
        states_tensors: a list or tensor of state tensors
        teacher_knowledges_tensors: a list or tensor of teacher knowledge tensors
    
    Returns:
        synthetic_buffer: a list of tuples, with each tuple in the form of (state, teacher_knowledge)
    """
    # Ensure both inputs are lists, for consistency
    #if isinstance(states_tensors, torch.Tensor):
    #    states_tensors = states_tensors.unbind()
    #if isinstance(teacher_knowledges_tensors, torch.Tensor):
    #    teacher_knowledges_tensors = teacher_knowledges_tensors.unbind()
    
    # Pair each state tensor with its corresponding teacher knowledge tensor
    synthetic_buffer = [(state.clone().detach(), teacher_knowledge.clone().detach()) for state, teacher_knowledge in zip(states_tensors, teacher_knowledges_tensors)]
    
    return synthetic_buffer


def match_loss(gw_syn, gw_real, device):

    dis = torch.tensor(0.0).to(device)

    # only MSE implemented for now
    gw_real_vec = []
    gw_syn_vec = []
    for ig in range(len(gw_real)):
        if gw_real[ig] is not None:
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
    gw_real_vec = torch.cat(gw_real_vec, dim=0)
    gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
    dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    return dis

def distribute_evenly(n, buckets=15):
    # Calculate base amount per bucket
    base = n // buckets
    # Calculate remainder
    remainder = n % buckets
    
    # Initialize the distribution array
    distribution = [base] * buckets
    
    # Distribute the remainder evenly by adding 1 to each of the first 'remainder' buckets
    for i in range(remainder):
        distribution[i] += 1
    
    return distribution
# def state_to_tensor(state: list):
#     """
#     Convert the list of state into tensors
#     NOTE: Only works when the observation is in the form of (state, teacher_knowledge), and teacher_knowledge is a tensor
#     Input:
#         buffer: a Gradient Matching Buffer
#     Return: 
#         A tensor of stacked observations, where each observation is [state, teacher knowledge], all flattened
#     """

#     states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
#     teacher_knowledges_tensors = [torch.tensor(teacher_knowledge, dtype=torch.float32, requires_grad=True) for teacher_knowledge in teacher_knowledges]

#     return states_tensors, teacher_knowledges_tensors


def get_batch(states_list, actions_list, action, batch_size):
    states = states_list[action]
    actions = actions_list[action]
    indices = torch.randperm(states.size(0))[:batch_size]
    states_batch = states[indices]
    actions_batch = actions[indices]
    return states_batch, actions_batch

def train_student_data_distillation_episodic_offline(student: Student, teacher: Teacher, env: gymnasium.Env or gym, loss_function: LossFunction, data_path, num_episodes=100, optimizer=None, online_network="student", max_steps=None, student_train_epochs=10, student_update_freq=16, rollout_buffer=None, synthetic_buffer=None, manual_init=True, rng=np.random.default_rng(), batch_size=1, device='cpu'):
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
    
    states_offline_list, actions_offline_list = entire_loader(data_path, env, device)
    total_size = synthetic_buffer.synthetic_buffer_size
    states_syn, teacher_knowledge_syn = data_distill_loader(data_path, total_size, env, device) 
    state_param = torch.nn.Parameter(states_syn).to(device)
    teacher_knowledge_param = teacher_knowledge_syn # Only for Naming 
    state_param.requires_grad = True
    synthetic_data = torch.nn.ParameterList([state_param, ])
    #optimizer_states_syn = torch.optim.Adam(synthetic_data, lr=0.01) # TODO: Try different optim.
    optimizer_states_syn = torch.optim.SGD(synthetic_data, lr=0.1, momentum=0.5) # TODO: Try different optim.
    optimizer_states_syn.zero_grad()
                    #synthetic_buffer.is_init_optimizer = False
    for episode in range(num_episodes):
        print("Running Episode ", episode)
        #state, _ = env.reset()
        #state = env.reset()
        #state = torch.tensor(state, dtype=torch.float32)

        done = False
        steps = 0
       
        num_action_list = distribute_evenly(synthetic_buffer.synthetic_buffer_size)
        loss = torch.tensor(0.0).to(device)
            #num_each_action = synthetic_buffer.synthetic_buffer_size // 15 #TODO: make it configured by hydra.
        start_idx = 0 
        for action in range(len(num_action_list)): # Do classification as Gradient Matching did
            bucket_size = num_action_list[action] 
            #real_batch = rollout_buffer.sample(batch_size)
            states_batch, actions_batch = get_batch(states_offline_list, actions_offline_list, action, batch_size)
            real_student_knowledge = student.get_knowledge(states_batch)
            real_loss = loss_function.loss(real_student_knowledge, actions_batch)

            syn_student_knowledge = student.get_knowledge(synthetic_data[0][start_idx:start_idx+bucket_size])
            #print(syn_student_knowledge.grad)
            syn_loss = loss_function.loss(syn_student_knowledge, teacher_knowledge_param[start_idx:start_idx+bucket_size])
            gw_real = torch.autograd.grad(real_loss, student_params)
            gw_real = list((_.detach().clone() for _ in gw_real))

            #print(syn_outer_batch)           
            #syn_student_knowledge = student.get_knowledge(syn_outer_batch[0][0])
            #syn_loss = loss_function.loss(syn_student_knowledge, syn_outer_batch[0][1])

            gw_syn = torch.autograd.grad(syn_loss, student_params, create_graph=True)
            #print(states_syn)
            #print(teacher_knowledge_syn)
            #print("Gradient of teacher knowledge param:", teacher_knowledge_param.grad)
            loss += match_loss(gw_syn=gw_syn, gw_real=gw_real, device=device)
            start_idx += bucket_size

            #prev_buffer = synthetic_data[0].clone().detach().cpu().numpy()
            #prev_teacher = synthetic_data[1].clone().detach().cpu().numpy()
            # Update synthetic buffer tensor by torch optimizer
            #print("Gradient of teacher knowledge param:", teacher_knowledge_param.grad)
        optimizer_states_syn.zero_grad()
        loss.backward()
        #print("Gradient of state param:", state_param.grad)
        optimizer_states_syn.step()

        syn_batch = [copy.deepcopy(synthetic_data[0].clone().detach().to(device)), copy.deepcopy(teacher_knowledge_param.clone().detach().to(device))]
        if steps % student_update_freq == 0:
            #print(f"step: {steps}, updating student")
            student.base_model.train()
            for epoch in range(student_train_epochs):
                #syn_batch = synthetic_buffer.sample(batch_size)
                syn_student_knowledge_training = student.get_knowledge(syn_batch[0])
                #print("syn student knowledge in training: " ,syn_student_knowledge_training.grad)
                syn_training_loss = loss_function.loss(syn_student_knowledge_training, syn_batch[1])

                # optimize
                optimizer.zero_grad() # student optimizer
                syn_training_loss.backward()
                optimizer.step()
                    #state, reward, term, trunc, _ = env.step(action)
    return

def train_student_fixed_data_distillation_episodic_offline(student: Student, teacher: Teacher, env: gymnasium.Env or gym, loss_function: LossFunction, data_path, states_syn, teacher_knowledge_syn, num_episodes=100, optimizer=None, online_network="student", max_steps=None, student_train_epochs=10, student_update_freq=16, rollout_buffer=None, synthetic_buffer=None, manual_init=True, rng=np.random.default_rng(), batch_size=1, device='cpu'):
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
    
    states_offline_list, actions_offline_list = entire_loader(data_path, env, device)
    total_size = synthetic_buffer.synthetic_buffer_size
    #states_syn, teacher_knowledge_syn = data_distill_loader(data_path, total_size, env, device) 
    state_param = torch.nn.Parameter(states_syn).to(device)
    teacher_knowledge_param = teacher_knowledge_syn # Only for Naming 
    state_param.requires_grad = True
    synthetic_data = torch.nn.ParameterList([state_param, ])
    #optimizer_states_syn = torch.optim.Adam(synthetic_data, lr=0.01) # TODO: Try different optim.
    optimizer_states_syn = torch.optim.SGD(synthetic_data, lr=0.1, momentum=0.5) # TODO: Try different optim.
    optimizer_states_syn.zero_grad()
                    #synthetic_buffer.is_init_optimizer = False
    for episode in range(num_episodes):
        print("Running Episode ", episode)
        #state, _ = env.reset()
        #state = env.reset()
        #state = torch.tensor(state, dtype=torch.float32)

        done = False
        steps = 0
       
        num_action_list = distribute_evenly(synthetic_buffer.synthetic_buffer_size)
        loss = torch.tensor(0.0).to(device)
            #num_each_action = synthetic_buffer.synthetic_buffer_size // 15 #TODO: make it configured by hydra.
        start_idx = 0 
        for action in range(len(num_action_list)): # Do classification as Gradient Matching did
            bucket_size = num_action_list[action] 
            #real_batch = rollout_buffer.sample(batch_size)
            states_batch, actions_batch = get_batch(states_offline_list, actions_offline_list, action, batch_size)
            real_student_knowledge = student.get_knowledge(states_batch)
            real_loss = loss_function.loss(real_student_knowledge, actions_batch)

            syn_student_knowledge = student.get_knowledge(synthetic_data[0][start_idx:start_idx+bucket_size])
            #print(syn_student_knowledge.grad)
            syn_loss = loss_function.loss(syn_student_knowledge, teacher_knowledge_param[start_idx:start_idx+bucket_size])
            gw_real = torch.autograd.grad(real_loss, student_params)
            gw_real = list((_.detach().clone() for _ in gw_real))

            #print(syn_outer_batch)           
            #syn_student_knowledge = student.get_knowledge(syn_outer_batch[0][0])
            #syn_loss = loss_function.loss(syn_student_knowledge, syn_outer_batch[0][1])

            gw_syn = torch.autograd.grad(syn_loss, student_params, create_graph=True)
            #print(states_syn)
            #print(teacher_knowledge_syn)
            #print("Gradient of teacher knowledge param:", teacher_knowledge_param.grad)
            loss += match_loss(gw_syn=gw_syn, gw_real=gw_real, device=device)
            start_idx += bucket_size

            #prev_buffer = synthetic_data[0].clone().detach().cpu().numpy()
            #prev_teacher = synthetic_data[1].clone().detach().cpu().numpy()
            # Update synthetic buffer tensor by torch optimizer
            #print("Gradient of teacher knowledge param:", teacher_knowledge_param.grad)
        optimizer_states_syn.zero_grad()
        loss.backward()
        #print("Gradient of state param:", state_param.grad)
        optimizer_states_syn.step()

        syn_batch = [copy.deepcopy(synthetic_data[0].clone().detach().to(device)), copy.deepcopy(teacher_knowledge_param.clone().detach().to(device))]
        if steps % student_update_freq == 0:
            #print(f"step: {steps}, updating student")
            student.base_model.train()
            for epoch in range(student_train_epochs):
                #syn_batch = synthetic_buffer.sample(batch_size)
                syn_student_knowledge_training = student.get_knowledge(syn_batch[0])
                #print("syn student knowledge in training: " ,syn_student_knowledge_training.grad)
                syn_training_loss = loss_function.loss(syn_student_knowledge_training, syn_batch[1])

                # optimize
                optimizer.zero_grad() # student optimizer
                syn_training_loss.backward()
                optimizer.step()
                    #state, reward, term, trunc, _ = env.step(action)
    return


def train_data_episodic_offline(student: Student, teacher: Teacher, env: gymnasium.Env or gym, loss_function: LossFunction, data_path, states_syn, teacher_knowledge_syn, num_episodes=100, optimizer=None, online_network="student", max_steps=None, student_train_epochs=10, student_update_freq=16, rollout_buffer=None, synthetic_buffer=None, manual_init=True, rng=np.random.default_rng(), batch_size=1, device='cpu'):
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
    teacher_params = list(teacher.model.model.parameters())
    # Ensuring requires_grad is set

    rewards = np.zeros(num_episodes)
    episode = 0
    
    states_offline_list, actions_offline_list = entire_loader(data_path, env, device)
    states_offline_list = [s.clone().detach().requires_grad_() for s in states_offline_list]

    total_size = synthetic_buffer.synthetic_buffer_size
    #states_syn, teacher_knowledge_syn = data_distill_loader(data_path, total_size, env, device) 
    state_param = torch.nn.Parameter(states_syn).to(device)
    teacher_knowledge_param = teacher_knowledge_syn # Only for Naming 
    state_param.requires_grad = True
    synthetic_data = torch.nn.ParameterList([state_param, ])
    #optimizer_states_syn = torch.optim.AdamW(synthetic_data, lr=1e-3) # TODO: Try different optim.
    optimizer_states_syn = torch.optim.SGD(synthetic_data, lr=0.1, momentum=0.5) # TODO: Try different optim.
    optimizer_states_syn.zero_grad()
                    #synthetic_buffer.is_init_optimizer = False
    for episode in range(num_episodes):
        print("Running Episode ", episode)
        #state, _ = env.reset()
        #state = env.reset()
        #state = torch.tensor(state, dtype=torch.float32)

        done = False
        steps = 0
       
        num_action_list = distribute_evenly(synthetic_buffer.synthetic_buffer_size)
        loss = torch.tensor(0.0).to(device)
            #num_each_action = synthetic_buffer.synthetic_buffer_size // 15 #TODO: make it configured by hydra.
        start_idx = 0 
        for action in range(len(num_action_list)): # Do classification as Gradient Matching did
            bucket_size = num_action_list[action] 
            #real_batch = rollout_buffer.sample(batch_size)
            states_batch, actions_batch = get_batch(states_offline_list, actions_offline_list, action, batch_size)
            
            real_teacher_knowledge = teacher.get_knowledge_distill_train(states_batch)
            real_loss = loss_function.loss(real_teacher_knowledge.to(device), actions_batch)
            syn_teacher_knowledge = teacher.get_knowledge_distill_train(synthetic_data[0][start_idx:start_idx+bucket_size])
            syn_loss = loss_function.loss(syn_teacher_knowledge.to(device), teacher_knowledge_param[start_idx:start_idx+bucket_size])
            # Perform Gradient Matching
            gw_real = torch.autograd.grad(real_loss, teacher_params, allow_unused=True)
            gw_real = list((_.detach().clone() if _ is not None else None for _ in gw_real))

            gw_syn = torch.autograd.grad(syn_loss, teacher_params, create_graph=True, allow_unused=True)
            #print(states_syn)
            #print(teacher_knowledge_syn)
            #print("Gradient of teacher knowledge param:", teacher_knowledge_param.grad)
            loss += match_loss(gw_syn=gw_syn, gw_real=gw_real, device=device)
            start_idx += bucket_size

            #prev_buffer = synthetic_data[0].clone().detach().cpu().numpy()
            #prev_teacher = synthetic_data[1].clone().detach().cpu().numpy()
            # Update synthetic buffer tensor by torch optimizer
            #print("Gradient of teacher knowledge param:", teacher_knowledge_param.grad)
        optimizer_states_syn.zero_grad()
        loss.backward()
        #print("Gradient of state param:", state_param.grad)
        optimizer_states_syn.step()

        syn_batch = [copy.deepcopy(synthetic_data[0].clone().detach().to(device)), copy.deepcopy(teacher_knowledge_param.clone().detach().to(device))]
    return syn_batch[0], syn_batch[1] 



