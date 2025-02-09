from torch.nn import Module, Sequential, Linear, ReLU

class DuelingDQN(Module):
    """A Dueling Deep Q-Networks (Dueling DQN) that employs only fully connected."""

    def __init__(self, obs_size, action_size, fc_dim, fc_num, fc_vl_num, fc_adv_num, adv_type='mean'):
        """Create new dueling DQN.
        
        Parameters
        --------------------
        obs_size: int
            observation size
        
        action_size: int
            action size

        fc_dim: int
            number of neurons of each hidden layer

        fc_num: int
            number of first initial hidden layers
            
        fc_vl_num: int
            number of value hidden layers

        fc_adv_num: int
            number of advantage hidden layers
            
        adv_type: str, optional
            type of operation to use for advantage. Values allowed are 'mean' and 'max'."""
        
        super(DuelingDQN, self).__init__()

        #Check parameters.
        if obs_size <= 0:
            raise ValueError("obs_size isn't positive integer")
        
        if action_size <= 0:
            raise ValueError("action_size isn't positive integer")

        if fc_dim <= 0:
            raise ValueError("fc_dim isn't positive integer")

        if fc_num <= 0:
            raise ValueError("fc_num isn't positive integer")
        
        if fc_vl_num <= 0:
            raise ValueError("fc_vl_num isn't positive integer")
        
        if fc_adv_num <= 0:
            raise ValueError("fc_num isn't positive integer")
        
        if adv_type != "mean" and adv_type != "max":
            raise ValueError("advantage operation type not allowed")

        #Initial hidden layers.
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

        #Advantage hidden layers.
        self._adv_type = adv_type
        self._adv = Sequential()
        for i in range(fc_adv_num):
            if i != fc_adv_num - 1:
                self._adv.append(Linear(fc_dim, fc_dim))
                self._adv.append(ReLU())
            else:
                self._adv.append(Linear(fc_dim, action_size))

        #Value hidden layers.
        self._value = Sequential()
        for i in range(fc_vl_num):
            if i != fc_vl_num - 1:
                self._value.append(Linear(fc_dim, fc_dim))
                self._value.append(ReLU())
            else:
                self._value.append(Linear(fc_dim, 1))

    def forward(self, x):
        """Process x.
        
        Parameter
        --------------------
        x: tc.Tensor
            a tensor
            
        Return
        --------------------
        y: tc.Tensor
            x processed by dueling DQN"""
        
        x = self._hidden(x)
        v = self._value(x)
        adv = self._adv(x)
        adv_op = adv.mean(dim=1, keepdim=True) if self._adv_type == "mean" else adv.amax(dim=1, keepdim=True)
        
        return v + adv - adv_op