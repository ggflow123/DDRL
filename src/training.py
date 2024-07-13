import gymnasium as gymnasium
import gym
from wrappers import Teacher, Student
from loss_functions import LossFunction
import torch
from tqdm import tqdm
import numpy as np
from buffer import RolloutBuffer
    

def train_student_episodic(student: Student, teacher: Teacher, env: gymnasium.Env or gym, 
                           loss_function: LossFunction, num_episodes=100, optimizer=None, 
                           online_network="student", max_steps=None, rollout_buffer=None, rng=np.random.default_rng(), batch_size=1):
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
        rollout_buffer = RolloutBuffer(1, rng=rng)

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
            teacher_knowledge = teacher.get_knowledge(state)
            #print(state.shape, teacher_knowledge.shape)
            rollout_buffer.add((state, teacher_knowledge))

            # sample from buffer 
            #if batch_size > 1:
                #raise NotImplementedError("Batch size > 1 not yet supported")
            batch = rollout_buffer.sample(batch_size)

            # calculate loss. TODO: add support for multiple batch sizes
            # Only works for batch size 1
            if batch_size == 1:
                student_knowledge = student.get_knowledge(batch[0][0])
                loss = loss_function.loss(student_knowledge, batch[0][1])
            else:
            # Batch size > 1
                student_knowledge = student.get_knowledge(batch[0])
                loss = loss_function.loss(student_knowledge, batch[1])

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # generate next state
            if online_network == "student":
                action = student.step(state)
            elif online_network == "teacher":
                action = teacher.step(state)
            else:
                action = online_network.step(state)
            #state, reward, term, trunc, _ = env.step(action)
            state, reward, term, infos = env.step(action)
            #done = term or trunc
            done = term
            info = infos[0]
            maybe_epinfo  = info.get('episode')
            rewards[episode] += reward
            state = torch.tensor(state, dtype=torch.float32)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True
        if maybe_epinfo:
            rewards[episode] = maybe_epinfo['r']
    return rewards

# def train_student_episodic(student: Student, teacher: Teacher, env: gym.Env, loss_function: LossFunction, num_episodes=100, optimizer=None, online_network=None):
#     '''
#     Trains the student model using the teacher model

#     Args:
#         student (Student): Student model
#         teacher (Teacher): Teacher model
#         env (gym.Env): Environment
#         loss_function (LossFunction): Loss function
#         num_episodes (int, optional): Number of episodes to train the student. Defaults to 1000.
#         optimizer (torch.optim, optional): Optimizer to use for training. Defaults to None.
#         online_network (int, optional): Whether to use an online network for training. If None, the student model is used as the online network. Defaults to None.
#     '''
#     for episode in tqdm(range(num_episodes)):
#         state = env.reset()
#         state = torch.tensor(state, dtype=torch.float32)

#         done = False
#         while not done:
#             student_knowledge = student.get_knowledge(state)
#             teacher_knowledge = teacher.get_knowledge(state)
#             loss = loss_function.loss(student_knowledge, teacher_knowledge)

#             # optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # generate next state
#             if online_network is None:
#                 action = student.step(state)
#             else:
#                 action = online_network.step(state)
#             state, reward, done, _ = env.step(action)
#             state = torch.tensor(state, dtype=torch.float32)
