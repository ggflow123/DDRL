import gymnasium as gymnasium
import gym
from wrappers import Teacher, Student
from loss_functions import LossFunction
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import copy
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

def compare_lists_of_tensor_pairs(list1, list2, rtol=1e-05, atol=1e-08):
    """
    Compare two lists of tuples of tensors.

    Parameters:
    - list1, list2: Lists of tuples, where each tuple contains two tensors.
    - rtol (float): Relative tolerance for tensor comparison (default 1e-5).
    - atol (float): Absolute tolerance for tensor comparison (default 1e-8).

    Returns:
    - bool: True if both lists are identical in terms of tensor values, shapes, and types.
    """
    # Check if both lists are the same length
    if len(list1) != len(list2):
        return False
    
    # Check each pair of tuples
    for (tensor1a, tensor1b), (tensor2a, tensor2b) in zip(list1, list2):
        # Check each pair of tensors within the tuples
                # Convert tensors to the same type (float) if necessary
        if tensor1a.dtype != tensor2a.dtype:
            tensor1a = tensor1a.to(torch.float32)
            tensor2a = tensor2a.to(torch.float32)
        if tensor1b.dtype != tensor2b.dtype:
            tensor1b = tensor1b.to(torch.float32)
            tensor2b = tensor2b.to(torch.float32)
        if not (torch.allclose(tensor1a, tensor2a, rtol=rtol, atol=atol) and 
                torch.allclose(tensor1b, tensor2b, rtol=rtol, atol=atol)):
            return False
        # Optionally check for the same shape and dtype
        if (tensor1a.shape != tensor2a.shape or tensor1a.dtype != tensor2a.dtype or
            tensor1b.shape != tensor2b.shape or tensor1b.dtype != tensor2b.dtype):
            return False

    return True



def train_student_data_distillation_episodic(student: Student, teacher: Teacher, env: gymnasium.Env or gym, loss_function: LossFunction, num_episodes=100, optimizer=None, online_network="student", max_steps=None, student_train_epochs=50, student_update_freq=16, rollout_buffer=None, synthetic_buffer=None, manual_init=True, rng=np.random.default_rng(), batch_size=1, device='cpu'):
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
    for episode in range(num_episodes):
        print("Running Episode ", episode)
        #state, _ = env.reset()
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        done = False
        steps = 0
        while not done:
            # place state and teacher knowledge in buffer
            #print("steps: ", steps)
            teacher_knowledge = teacher.get_knowledge(state)
            #print(f"state: ", state.min(), state.max(), state.mean())
            #print(f"teacher knowledge: {teacher_knowledge}")
            rollout_buffer.add((state, teacher_knowledge))
            synthetic_buffer.add((state, teacher_knowledge))

            # size of synthetic buffer does not reach the threshold size, can not do gradient matching loss.
            # fill out the synthetic buffer first
            #print(len(synthetic_buffer.synthetic_buffer), synthetic_buffer.synthetic_init_threshold_size)
            #if len(synthetic_buffer.synthetic_buffer) < synthetic_buffer.synthetic_init_threshold_size:
            if not synthetic_buffer.is_synthetic_initialized:
                action = teacher.step(state)
                #print(f"step: {action}")
                #print("not activated")
            else:
                # sample from buffer 
                if synthetic_buffer.is_init_optimizer: # Make this is_initialized() by synthetic buffer.
                    if not manual_init:                    
                        #synthetic_buffer_list = synthetic_buffer.synthetic_buffer
                        states_syn = copy.deepcopy(synthetic_buffer.synthetic_buffer)
                        teacher_knowledge_syn = copy.deepcopy(synthetic_buffer.synthetic_knowledge_buffer.float())
                    else:
                        num_action_list = distribute_evenly(synthetic_buffer.synthetic_buffer_size)
                        total_size = synthetic_buffer.synthetic_buffer_size
                        #num_each_action = synthetic_buffer.synthetic_buffer_size // 15 #TODO: make it configured by hydra.
                        states_syn = torch.empty((total_size, *env.observation_space.shape), dtype=torch.float32, device=device)
                        teacher_knowledge_syn = torch.empty((total_size, *(env.action_space.n, )), dtype=torch.float32, device=device)
                        index = 0
                        for action in range(len(num_action_list)): # actions
                            bucket_size = num_action_list[action]
                            action_tensor = torch.zeros(15, dtype=torch.float32, device=device)
                            for num in range(bucket_size):
                                #state, reward, term, trunc, _ = env.step(action)
                                state, reward, term, infos = env.step(np.expand_dims(action, axis=0))
                                state = torch.tensor(state, dtype=torch.float32, device=device)
                                states_syn[index+num] = state
                                action_tensor[action] = 1.0
                                teacher_knowledge_syn[index+num] = action_tensor
                            index += bucket_size
                    #states_syn, teacher_knowledge_syn = buffer_to_tensor(synthetic_buffer_list)
                    #print(states_syn.shape)
                    #print(teacher_knowledge_syn.shape)
                    #states_syn.requires_grad_()
                    #teacher_knowledge_syn.requires_grad_()
                    state_param = torch.nn.Parameter(states_syn).to(device)
                    #teacher_knowledge_param = torch.nn.Parameter(teacher_knowledge_syn).to(device)
                    teacher_knowledge_param = teacher_knowledge_syn # Only for Naming 
                    state_param.requires_grad = True
                    #teacher_knowledge_param.requires_grad = True
                    #print("teacher knowledge param: " ,teacher_knowledge_param)
                    synthetic_data = torch.nn.ParameterList([state_param, ])
                    #synthetic_teacher = [teacher_knowledge_param, ]
                    #optimizer_states_syn = torch.optim.Adam(synthetic_data, lr=0.01) # TODO: Try different optim.
                    optimizer_states_syn = torch.optim.SGD(synthetic_data, lr=0.1, momentum=0.5) # TODO: Try different optim.
                    optimizer_states_syn.zero_grad()

                    synthetic_buffer.is_init_optimizer = False

                ### UPDATE SYNTHETIC DATA ### 
                #real_batch = rollout_buffer.sample(batch_size)
                #real_student_knowledge = student.get_knowledge(real_batch[0])
                #real_loss = loss_function.loss(real_student_knowledge, real_batch[1])

                loss = torch.tensor(0.0).to(device)
                #num_each_action = synthetic_buffer.synthetic_buffer_size // 15 #TODO: make it configured by hydra.
                start_idx = 0 
                for action in range(len(num_action_list)): # Do classification as Gradient Matching did
                    bucket_size = num_action_list[action] 
                    real_batch = rollout_buffer.sample(batch_size)
                    real_student_knowledge = student.get_knowledge(real_batch[0])
                    real_loss = loss_function.loss(real_student_knowledge, real_batch[1])

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

                #for name, param in student.base_model.named_parameters():
                    #print(f"{name} gradient: {param.grad}")
                #print("Gradient of state param:", state_param.grad)
                #print("=====After Loss backward=====")
                #print(states_syn)
                #print(teacher_knowledge_syn)
                # Update Synthetic Buffer
                #prev_buffer = synthetic_buffer.synthetic_buffer
                #print(prev_buffer)
                #synthetic_buffer.synthetic_buffer = tensor_to_buffer(states_syn, teacher_knowledge_syn)
                synthetic_buffer.synthetic_buffer = synthetic_data[0].clone().detach().to(device)
                synthetic_buffer.synthetic_knowledge_buffer = teacher_knowledge_param.clone().detach().to(device)
                #print(synthetic_buffer.synthetic_knowledge_buffer)
                #print(compare_lists_of_tensor_pairs(prev_buffer, synthetic_buffer.synthetic_buffer))
                #curr_buffer = synthetic_data[0].clone().detach().cpu().numpy()
                #`curr_teacher = synthetic_data[1].clone().detach().cpu().numpy()
                #print("If synthetic buffers are equal: ", np.equal(prev_buffer, curr_buffer))
                #print("If synthetic teachers are equal: ", np.equal(prev_teacher, curr_teacher))
                # Train on synthetic data for student
                #print("training on synthetic data.")
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
                # generate next state
                if online_network == "student":
                    action = student.step(state)
                elif online_network == "teacher":
                    action = teacher.step(state)
                else:
                    action = online_network.step(state)

            #state, reward, term, trunc, _ = env.step(action)
            if np.isscalar(action):
                action = np.expand_dims(action, axis=0)  # Add an axis to make it a single-element array
            state, reward, term, infos = env.step(action)
            #done = term or trunc
            done = term
            done = done[0]
            info = infos[0]
            maybe_epinfo = info.get('episode')
            rewards[episode] += reward
            state = torch.tensor(state, dtype=torch.float32)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True
        if maybe_epinfo:
            rewards[episode] = maybe_epinfo['r']
    return rewards
