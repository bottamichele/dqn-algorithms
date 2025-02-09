from torch.nn import Module, Sequential, Linear, ReLU

class DQN(Module):
    """A Deep Q-Networks (DQN) that employs only fully connected."""

    def __init__(self, obs_size, action_size, fc_dim, fc_num):
        """Create new DQN.
        
        Parameters
        --------------------
        obs_size: int
            observation size
        
        action_size: int
            action size

        fc_dim: int
            number of neurons of each hidden layer

        fc_num: int
            number of DQN's hidden layers"""
        
        super(DQN, self).__init__()

        #Check parameters.
        if obs_size <= 0:
            raise ValueError("obs_size isn't positive integer")
        
        if action_size <= 0:
            raise ValueError("action_size isn't positive integer")

        if fc_dim <= 0:
            raise ValueError("fc_dim isn't positive integer")

        if fc_num <= 0:
            raise ValueError("fc_num isn't positive integer")

        #Hidden layers.
        self._hidden = Sequential()
        for i in range(fc_num):
            if i == 0:
                #First hidden layer.
                self._hidden.append(Linear(obs_size, fc_dim))
                self._hidden.append(ReLU())
            else:
                #i-th hidden layer.
                self._hidden.append(Linear(fc_dim, fc_dim))
                self._hidden.append(ReLU())

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
            x processed by DQN"""
        
        x = self._hidden(x)
        return self._out(x)