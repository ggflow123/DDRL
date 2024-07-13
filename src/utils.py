# Replay buffer for standard gym tasks
import cv2
import gymnasium as gym
import numpy as np
import torch
import logging
import os
import datetime
from abc import ABC

class AbstractLogged(ABC):
    _instance_counter = 0

    def __init__(self):
        # Increment the instance counter and set it as part of the logger name
        type(self)._instance_counter += 1
        self.instance_id = type(self)._instance_counter
        
        # Create a logger with a name based on the class name and instance counter
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.instance_id}")

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)


class ReplayBuffer():
	def __init__(self, state_dim, batch_size, buffer_size, device):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, 1))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

	def add(self, state, action, next_state, reward, done):

		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self):
		ind = np.random.randint(0, self.size, size=self.batch_size)

		batch = (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

		return batch
	
def setup_logging_environment(log_level=logging.DEBUG):
    """
    Sets up the logging environment by ensuring the log directory exists and configuring
    the root logger to use a FileHandler with a unique filename. This setup affects all
    loggers created in the application.

    :param log_directory: The directory where log files will be stored.
    :param log_level: The logging level for the handler.
    """

    # Ensure the logs directory exists
    # os.makedirs(log_directory, exist_ok=True)

    # Create a unique filename with the current date and time
    # filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log')
    # full_log_path = os.path.join(log_directory, filename)

    # Configure the root logger to use a FileHandler with the unique filename
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')