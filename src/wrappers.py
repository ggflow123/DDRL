import torch
from abc import ABC, abstractmethod 

class Model(ABC):
    '''
    Wrapper for model
    '''
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def step(self, state: torch.Tensor):
        '''
        Returns the next action given the current state
        '''
        pass
    
    @abstractmethod
    def save_model(self, path):
        '''
        Save the model to a file
        '''
        pass

    @abstractmethod
    def load_model(self, path):
        '''
        Load the model from a file
        '''
        pass


class Teacher(Model):
    '''
    Wrapper for teacher model
    '''
    def __init__(self, model_name):
        self.model_name = model_name # Name of the model used to train the student

    @abstractmethod
    def get_knowledge(self, state: torch.Tensor)->torch.Tensor:
        '''
        Returns the knowledge of the teacher that is used to train the student
        '''
        pass

    def get_batch_knowledge(self, states: torch.Tensor)->torch.Tensor:
        '''
        Returns the knowledge of the teacher that is used to train the student

        TODO: this is a hack, should be parallelized
        '''
        knowledge = []
        for state in states:
            knowledge.append(self.get_knowledge(state))
        return torch.stack(knowledge)
    
    @abstractmethod
    def step(self, state: torch.Tensor):
        '''
        Returns the next action given the current state
        '''
        pass
    
class Student(Model):
    '''
    Wrapper for student model
    '''
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def step(self, state: torch.Tensor):
        '''
        Returns the next action given the current state
        '''
        pass
    
    @abstractmethod
    def get_knowledge(self, state: torch.Tensor)->torch.Tensor:
        '''
        Returns the knowledge of the student
        '''
        pass
    
    def get_batch_knowledge(self, states: torch.Tensor)->torch.Tensor:
        '''
        Returns the knowledge of the student

        TODO: this is a hack, should be parallelized
        '''
        knowledge = []
        for state in states:
            knowledge.append(self.get_knowledge(state))
        return torch.stack(knowledge)

    