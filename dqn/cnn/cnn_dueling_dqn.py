from torch.nn import Module, Sequential, Conv2d, Linear, ReLU

from .utils import output_dim_conv2d

class CnnDuelingDQN(Module):
    """A Dueling Deep Q-Networks which employs 2D convolution layers."""

    def __init__(self, action_size, obs_size=(4, 84, 84), conv_layers=[(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)], fc_dim=512, fc_num=1, advantage_type='mean'):
        """Create a new Dueling DQN which uses 2D convolution layers.
        
        Parameters:
        --------------------
        action_size: int
            action size.
            
        obs_size: tuple, optional
            observation size. The tuple must be as (ch, h, w) where 
            w is the image width, h is image height and ch is number of channels.
            
        conv_layers: list, optional
            list of 2D convolutional layer parameters. The list must contains tuples as (n_ch, k, s, p) where
            n_ch is number of channels, k is kernel size, s is stride and p is padding.
            
        fc_dim: int, optional
            number of nodes for a fully connected layer.
            
        fc_num: int, optional
            number of fully connected layers to use.
            
        advantage_type: str, optional
            type of operation to use for advantage. The values allowed are 'mean' and 'max'."""
        
        super().__init__()

        if action_size <= 0:
            raise ValueError("There must be at least one action which can be performed.")

        if len(conv_layers) <= 0:
            raise ValueError("conv_layers must contain at least one 2D convolutional layer parameter.")
        
        if fc_dim <= 0:
            raise ValueError("fc_dim isn't positive integer.")

        if fc_num <= 0:
            raise ValueError("fc_num isn't positive integer.")
        
        if advantage_type != "mean" and advantage_type != "max":
            raise ValueError("advantage operation type not allowed")

        #2D convolution layers.
        self._conv_layers = Sequential()
        for i in range(len(conv_layers)):
            n_channels, kernel_size, stride, padding = conv_layers[i]

            if i == 0:
                #First 2D convolution layer.
                self._conv_layers.append(Conv2d(obs_size[0], n_channels, kernel_size, stride, padding))
            else:
                #i-th 2D convolution layer.
                self._conv_layers.append(Conv2d(conv_layers[i - 1][0], n_channels, kernel_size, stride, padding))
        
            self._conv_layers.append(ReLU())

        #Advantage layers.
        self._advantage_type = advantage_type
        self._advantage_layers = Sequential()

        for i in range(fc_num + 1):
            if i == 0:
                #First fully connected layer.
                _, height, width = obs_size
                for (_, k, s, p) in conv_layers:
                    width, height = output_dim_conv2d((width, height), k, s, p)
                
                self._advantage_layers.append(Linear(conv_layers[-1][0] * width * height, fc_dim))
            elif i == fc_num:   #<-- i == (fc_num - 1) + 1
                #Last fully connected layer.
                self._advantage_layers.append(Linear(fc_dim, action_size))
            else:
                #i-th fully connected layer.
                self._advantage_layers.append(Linear(fc_dim, fc_dim))
            
            if i != fc_num :    #<-- i != (fc_num - 1) + 1
                self._advantage_layers.append(ReLU())

        #Value layers.
        self._value_layers = Sequential()

        for i in range(fc_num + 1):
            if i == 0:
                #First fully connected layer.
                _, height, width = obs_size
                for (_, k, s, p) in conv_layers:
                    width, height = output_dim_conv2d((width, height), k, s, p)
                
                self._value_layers.append(Linear(conv_layers[-1][0] * width * height, fc_dim))
            elif i == fc_num:   #<-- i == (fc_num - 1) + 1
                #Last fully connected layer.
                self._value_layers.append(Linear(fc_dim, 1))
            else:
                #i-th fully connected layer.
                self._value_layers.append(Linear(fc_dim, fc_dim))
                
            if i != fc_num: #<-- i != (fc_num - 1) + 1
                self._value_layers.append(ReLU())
        

    def forward(self, x):
        """Process x.
        
        Parameter
        --------------------
        x: tc.Tensor
            a tensor
            
        Return
        --------------------
        y: tc.Tensor
            x processed by CnnDuelingDQN"""
                
        x = self._conv_layers(x)
        x = x.view(x.size(0), -1)
        value = self._value_layers(x)
        advantage = self._advantage_layers(x)
        advantage_operation = advantage.mean(dim=1, keepdim=True) if self._advantage_type == "mean" else advantage.amax(dim=1, keepdim=True)

        return value + advantage - advantage_operation