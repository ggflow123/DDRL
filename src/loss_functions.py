import torch

class LossFunction:
    '''
    Abstract class for loss functions for torch models
    '''

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def loss(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:
        '''
        Computes the loss given the true and predicted values
        '''
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    '''
    Mean squared error loss function
    '''

    def __init__(self):
        super().__init__("Mean Squared Error")

    def loss(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:
        return torch.mean((y_true - y_pred)**2)
    
class MeanAbsoluteError(LossFunction):
    '''
    Mean absolute error loss function
    '''

    def __init__(self):
        super().__init__("Mean Absolute Error")

    def loss(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:
        return torch.mean(torch.abs(y_true - y_pred))