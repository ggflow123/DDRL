from wrappers import Student
from base_models import *
import torch
import numpy as np
from typing import Optional
from buffer import RolloutBufferBatch
# from sklearn


class PolicyStudent(Student):

    base_model: torch.nn.Module

    def __init__(self, base_model=Optional[torch.nn.Module], model_name="Policy-Student", state_shape=None, action_size=None, device='cpu', normalised_knowledge=True):
        '''
        
        Args:
            base_model (torch.nn.Module): Base model for the student
            model_name (str): Name of the student model
            state_shape (tuple): Shape of the state
            action_size (int): Number of actions
            normalised_knowledge (bool): Whether the knowledge is normalised. Defaults to True.
        '''
        super().__init__(model_name)
        # # if base_model is None, use the default MLP model
        if base_model is None:
            base_model = MLP(self.state_shape, 128, tuple([self.action_size]))

        try:
            self.base_model = base_model.to(device)
        except:
            self.base_model = base_model

        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.normalised_knowledge = normalised_knowledge

    def step(self, state: torch.Tensor):

        # convert state to tensor
        #state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True)
        state = state.to(dtype=torch.float32)
        # softmax over the output of the base model, then sample an action from the distribution
        action_probs = torch.softmax(self.base_model.forward(state.to(self.device)), dim=-1)
        action = torch.multinomial(action_probs, 1).squeeze().item()
        return action

    def get_knowledge(self, state: torch.Tensor):
        # convert state to tensor
        #state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True)
        state = state.to(dtype=torch.float32)
        if self.normalised_knowledge:
            return torch.softmax(self.base_model.forward(state.to(self.device)), dim=-1)
        else:
            return self.base_model.forward(state.to(self.device))
    
    def save_model(self, path):
        '''
        Torch save the full base model to path, not just the model weights
        '''
        torch.save(self.base_model, path)

    def load_model(self, path):
        '''
        The path should be to a torch model file for the base model
        '''
        self.base_model = torch.load(path).to(self.device)
        self.base_model.eval()
        self.state_shape = self.base_model.input_shape
        self.action_size = self.base_model.output_shape[0]


    def load_model_training(self, path):
        '''
        The path should be to a torch model file for the base model
        '''
        self.base_model = torch.load(path).to(self.device)
        self.base_model.train()
        self.state_shape = self.base_model.input_shape
        self.action_size = self.base_model.output_shape[0]

class BuffedStudent(PolicyStudent):
    '''
    Student with synthetic buffer attached so that they can learn from it
    '''

    def __init__(self, buffer: RolloutBufferBatch, base_model:Optional[torch.nn.Module], optimizer, model_name="Buffed-Student", state_shape=None, action_size=None, device='cpu', normalised_knowledge = True):
        super().__init__(base_model, model_name, state_shape, action_size, device, normalised_knowledge)
        self.buffer = buffer
        self.recall_batch_size = 4
        self.optimizer = optimizer

    def step(self, state: torch.Tensor):

        # convert state to tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device, requires_grad=True)

        # recall train
        self.recall_train(state, self.recall_batch_size)

        # softmax over the output of the base model, then sample an action from the distribution
        action_probs = torch.softmax(self.base_model.forward(state), dim=-1)
        action = torch.multinomial(action_probs, 1).squeeze().item()
        return action
    
    def recall_train(self, state: torch.Tensor, batch_size=8):
        '''
        Finds the batch_size most similar states in the synthetic buffer to the state and trains the student on them
        '''

        # find the most similar states in the buffer
        state = state.unsqueeze(0)
        state = state.to(self.device)
        state_batch = self.buffer.state_buffer
        state_batch = state_batch.to(self.device)
        distances = torch.cdist(state, state_batch) # calculates euclidean distance between state and all states in buffer
        _, indices = torch.topk(distances, batch_size, largest=False)

        # get the knowledge for the most similar states
        knowledge_batch = self.buffer.knowledge_buffer[indices]

        # get the knowledge for the most similar states from the student
        student_knowledge = self.get_knowledge(state_batch[indices])

        # calculate the loss
        loss = torch.nn.functional.cross_entropy(student_knowledge, knowledge_batch)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()






