from torch.nn import Module, Sequential, Conv2d, Linear, ReLU

from .utils import output_dim_conv2d

class CnnDQN(Module):
    """A Deep Q-Networks which employs 2D convolution layers."""

    def __init__(self, action_size, obs_size=(4, 84, 84), conv_layers=[(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)], fc_dim=512, fc_num=1):
        """Create a new DQN which uses 2D convolution layers.
        
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
            number of fully connected layers to use."""
        
        super().__init__()

        if action_size <= 0:
            raise ValueError("There must be at least one action which can be performed.")

        if len(conv_layers) <= 0:
            raise ValueError("conv_layers must contain at least one 2D convolutional layer parameter.")
        
        if fc_dim <= 0:
            raise ValueError("fc_dim isn't positive integer.")

        if fc_num <= 0:
            raise ValueError("fc_num isn't positive integer.")

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

        #Fully connected layers.
        self._fc_layers = Sequential()
        for i in range(fc_num):
            if i == 0:
                #First fully connected layer.
                _, height, width = obs_size
                for (_, k, s, p) in conv_layers:
                    width, height = output_dim_conv2d((width, height), k, s, p)
                
                self._fc_layers.append(Linear(conv_layers[-1][0] * width * height, fc_dim))
            else:
                #i-th fully connected layer.
                self._fc_layers.append(Linear(fc_dim, fc_dim))

            self._fc_layers.append(ReLU())

        #Output layer.
        self._out = Linear(fc_dim, action_size)

    def forward(self, x):
        """Process x.
        
        Parameter
        --------------------
        x: tc.Tensor
            a tensor
            
        Return
        --------------------
        y: tc.Tensor
            x processed by CnnDQN"""
                
        x = self._conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self._fc_layers(x)
        return self._out(x)