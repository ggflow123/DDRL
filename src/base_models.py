import torch
import torch.nn as nn
import numpy as np
from file_management import get_class

# Define abstract class for base models
class BaseModel(nn.Module):
    '''
    Abstract class for base models
    '''

    def __init__(self, name, input_shape, output_shape):
        super().__init__()
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __str__(self):
        return self.name

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        Args:
            x (torch.tensor): Input tensor
        '''
        raise NotImplementedError

# Define the policy network
class MLP(BaseModel):
    def __init__(self, input_shape, hidden_size, output_shape, num_layers=10, activation='torch.nn.ReLU', dropout=0.0):
        '''
        Args:
            input_shape (int): Input shape of the network
            hidden_size (int): Hidden size of the network
            output_shape (int): Output shape of the network
            num_layers (int, optional): Number of hidden layers in the network. Defaults to 10.
        '''
        super().__init__("MLP", input_shape, output_shape)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        activation_layer = get_class(activation)

        self.input_size = np.prod(input_shape) if isinstance(input_shape, tuple) else input_shape
        self.output_size = np.prod(output_shape) if isinstance(output_shape, tuple) else output_shape
        
        # create num_layers layers of linear layers followed by ReLU
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(activation_layer())
        for _ in range(self.num_layers-1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(activation_layer())
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> torch.tensor:
        '''
        Args:
            x (torch.tensor): Input tensor
        '''
        if len(x.shape) == 3:
            x = x.reshape(-1, self.input_size)
            out = self.model(x)
            # reshape the output to the desired output shape
            out = out.reshape(-1, *self.output_shape)
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
            out = self.model(x)
            out = out.reshape(x.shape[0], *self.output_shape)

        return out
    
class CNN(BaseModel):
    def __init__(self, input_shape, output_shape, kernel_size=3, lps = 2, channel_multiplier=1, activation='torch.nn.ReLU', pool='torch.nn.AvgPool2d'):
        '''
        Args:
            input_shape (tuple): Input shape of the network
            output_shape (int): Output shape of the network
            kernel_size (int): Size of the convolutional kernel
            lps (int): Number of layers per stage
            channel_multiplier (int): Multiplier for the number of channels
            activation (str): Activation function to use
            pool (str): Pooling function to use
        '''
        super().__init__("CNN", input_shape, output_shape)
        self.kernel_size = kernel_size
        self.output_size = np.prod(output_shape) if isinstance(output_shape, tuple) else output_shape
        input_channels = input_shape[2]
        input_grid_size = input_shape[0]

        #self.conv1 = nn.Conv2d(input_channels, input_channels * channel_multiplier, kernel_size, padding=1, stride=1)
        #self.conv2 = nn.Conv2d(input_channels, input_channels * channel_multiplier * channel_multiplier, kernel_size, padding=1, stride=1)

        layers = []
        #layers.append(nn.Conv2d(input_channels, 32, kernel_size=kernel_size, padding=1, stride=2))
        #layers.append(nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1, stride=2))
        #layers.append(nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1, stride=2))
        #layers.append(nn.Conv2d(32, input_channels, kernel_size=kernel_size, padding=1, stride=2))
        #=================== Old layers ============================
        #layers.append(nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=1))
        layers.append(nn.Conv2d(input_channels, input_channels * channel_multiplier, 3, 1, 1))

        #input_channels = input_channels * channel_multiplier

        layers.append(nn.Conv2d(input_channels * channel_multiplier, input_channels * channel_multiplier * channel_multiplier, 3, 1, 1))
        #input_channels = input_channels * channel_multiplier
        layers.append(nn.Conv2d(input_channels * channel_multiplier * channel_multiplier, input_channels * channel_multiplier, 3, 1, 1))
        layers.append(nn.Conv2d(input_channels * channel_multiplier, input_channels, 3, 1, 1))
        #=================== Old layers =============================
        self.layers = nn.ModuleList(layers) 
        #self.conv1 = nn.ModuleList([nn.Conv2d(input_channels, input_channels, 3, 1, 1)])
        #self.conv1.extend(nn.ModuleList([nn.Conv2d(input_channels, input_channels, 3, 1, 1) for i in range(lps-1)]))


        #self.conv2 = nn.ModuleList([nn.Conv2d(input_channels, input_channels * channel_multiplier, 3, 1, 1)])
        #self.conv2.extend(nn.ModuleList([nn.Conv2d(input_channels * channel_multiplier, input_channels * channel_multiplier, 3, 1, 1) for i in range(lps-1)]))

        #input_channels = input_channels * channel_multiplier

        #self.conv3 = nn.ModuleList([nn.Conv2d(input_channels, input_channels * channel_multiplier, 3, 1, 1)])
        #self.conv3.extend(nn.ModuleList([nn.Conv2d(input_channels * channel_multiplier, input_channels * channel_multiplier, 3, 1, 1) for i in range(lps-1)]))

        #input_channels = input_channels * channel_multiplier
        
        self.pool = get_class(pool)(2)
        self.activation = get_class(activation)()
        final_size = input_channels * input_grid_size*input_grid_size//(64*4)
        #final_size = input_channels * input_grid_size*input_grid_size//4
        self.fc1 = nn.Linear(final_size,  self.output_size)


    def forward(self, x):
        '''
        Args:
            x (torch.tensor): Input tensor
        '''

        # # reorder the dimensions fro (batch, height, width, channels) to (batch, channels, height, width)
        #print(x.shape)
        if len(x.shape) == 4: # with batch dimension during training
            x = x.permute(0,3,1,2)
        elif len(x.shape) == 3: # during step with single state
            x = x.permute(2,0,1)
        else:
            print("Wrong Dimension: either the input of the network has 3 or 4 dimension.")

        # reorder the dimensions fro (height, width, channels) to (channels, height, width)
        #x = x.permute(2,0,1)

        #for conv in self.conv1:
        #    x = conv(x)
        #    x = self.activation(x)

        #x = self.pool(x)

        #for conv in self.conv2:
        #    x = conv(x)
        #    x = self.activation(x)

        #x = self.pool(x)

        #for conv in self.conv3:
        #    x = conv(x)
        #    x = self.activation(x)

        #x = self.pool(x)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            x = self.activation(x)
            x = self.pool(x)
            #if i % 2 == 1:
                #x = self.pool(x)
        #x = self.pool(x)
        # flatten the output
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], np.prod(x.shape[1:]))
            x = self.fc1(x)
            x = x.reshape(x.shape[0], *self.output_shape)
        elif len(x.shape) == 3:
            x = x.reshape(-1, np.prod(x.shape[0:]))
            x = self.fc1(x)
            x = x.reshape(-1, *self.output_shape)
        return x

