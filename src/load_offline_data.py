from data_loader import DataLoaderPickle
import pickle
import numpy as np
import torch
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


def data_distill_loader(data_path, data_size, env, device):
    # Get the data with respect with each action as evenly as possible.
    dataloader = DataLoaderPickle(data_path)
    num_action_list = distribute_evenly(data_size)
    data = []
    print("Loading data with size: ", data_size)
    for action in range(len(num_action_list)):
        print(action)
        bucket_size = num_action_list[action]
        data_curr_action = dataloader.get_flatted_dataset_filtered_by_action(action)
        np.random.shuffle(data_curr_action)
        data.extend(data_curr_action[:bucket_size])
    states = torch.empty((data_size, *env.observation_space.shape), dtype=torch.float32, device=device)
    actions = torch.empty((data_size, *(env.action_space.n, )), dtype=torch.int64, device=device)
    print("Convert offline data into states and actions tensors.")
    for i in range(len(data)):
        item = data[i]
        state, action = item[0], item[1]
        state = state.to(device)
        states[i] = state
        action_tensor = torch.zeros(15, device=device)
        action_tensor[action] = 1
        actions[i] = action_tensor
    return states, actions

def entire_loader(data_path, env, device):
    # Get the data with respect with each action as evenly as possible.
    data = []
    dataloader = DataLoaderPickle(data_path)
    print("Loading entire offline data.")
    data_size = len(dataloader.data)
    for action in range(15):
        print(action)
        data_curr_action = dataloader.get_flatted_dataset_filtered_by_action(action)
        data.append(data_curr_action)
    print("Convert offline data into states and actions tensors.")
    print("15 lists to 15 actions: ", len(data))
    states_list = []
    actions_list = []
    for action in range(len(data)):
        data_curr_action = data[action]
        states = torch.empty((len(data_curr_action), *env.observation_space.shape), dtype=torch.float32, device=device)
        actions = torch.empty((len(data_curr_action), *(env.action_space.n, )), dtype=torch.int64, device=device)
        for i in range(len(data_curr_action)):
            item = data_curr_action[i]
            state, action = item[0], item[1]
            state = state.to(device)
            states[i] = state
            action_tensor = torch.zeros(15, device=device)
            action_tensor[action] = 1
            actions[i] = action_tensor
        states_list.append(states)
        actions_list.append(actions)
    return states_list, actions_list

def bc_loader(data_path, percentage, env, device):
    dataloader = DataLoaderPickle(data_path)
    data = []
    dataloader.filter_top_trajectories_fraction(percentage)
    data_filtered = dataloader.data
    data_size = 0
    for k in range(len(data_filtered)):
        data_size += len(data_filtered[k])
    print(f"Behavioral Cloning data size with {percentage}: {data_size}")
    states = torch.empty((data_size, *env.observation_space.shape), dtype=torch.float32, device=device)
    actions = torch.empty((data_size, *(env.action_space.n, )), dtype=torch.int64, device=device)
    for j in range(len(data_filtered)):
        curr_episode = data_filtered[j]
        for i in range(len(curr_episode)):
            item = curr_episode[i]
            state, action = item[0], item[1]
            state = state.to(device)
            states[i] = state
            action_tensor = torch.zeros(15, device=device)
            action_tensor[action] = 1
            actions[i] = action_tensor
    return states, actions

#data_path = "/gpfs/u/home/KDRL/KDRLynze/scratch/KDRL/data/bigfish-offlinedata.pkl"
#data_distill_loader(data_path, 1500)
