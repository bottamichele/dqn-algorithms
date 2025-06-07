import torch as tc

from ._n_step_transiction_collector import NStepTransictionCollector

from ._uniform_memory import UniformMemory
from ._prop_prio_mem import PropPriorMemory

# ========================================
# ========= N-STEP UNIFORM MEMORY ========
# ========================================

class NStepUniformMemory(UniformMemory):
    """A memory replay which randomly samples minibatches and uses N-step leanirng."""

    def __init__(self, max_size, n_step, gamma, obs_size, obs_dtype=tc.float32, act_dtype=tc.int8, rew_dtype=tc.float32, device=tc.device("cpu")):
        """Create new memory replay.
        
        Parameters
        --------------------
        max_size: int
            max size of memory replay

        n_step: int
            number of steps to use for N-step learning

        gamma: float
            discount factor for N-step leaning
            
        obs_size: int or tuple
            observation size
            
        obs_dtype: tc.dtype, optional
            observation's data type
            
        act_dtype: tc.dtype, optional
            action's data type
            
        rew_dtype: tc.dtype, optional
            reward's data type
            
        device: tc.device, optional
            device where the memory replay is on"""
        
        #Check if the parameters are correct.
        if n_step <= 0:
            raise ValueError("n_step must be a positive integer.")
        
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        
        super().__init__(max_size, obs_size, obs_dtype, act_dtype, rew_dtype, device)
        self._n_step_buffer = NStepTransictionCollector(n_step)
        self._gamma = gamma

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        self._n_step_buffer.collect(obs, action, reward, next_obs, next_obs_done)

        if len(self._n_step_buffer) == self._n_step_buffer.n_step:
            transiction = self._n_step_buffer.get_transiction(self._gamma)
            super().store_transiction(transiction[0], transiction[1], transiction[2], transiction[3], transiction[4])

# ==========================================
# = N-STEP PROPORTIONAL PRIORITIZED MEMORY =
# ==========================================

class NStepPropPriorMemory(PropPriorMemory):
    """A memory replay which samples minibatches based on transictions' priority and uses N-step learning."""

    def __init__(self, max_size, n_step, gamma, obs_size, alpha=0.6, beta=0.4, eps=10**-5, obs_dtype=tc.float32, act_dtype=tc.int8, rew_dtype=tc.float32, device=tc.device("cpu")):
        """Create new memory replay.
        
        Parameters
        --------------------
        max_size: int
            max size of memory replay

        n_step: int
            number of steps to use for N-step learning

        gamma: float
            discount factor for N-step leaning
            
        obs_size: int or tuple
            observation size
            
        alpha: float, optional
            priority factor
            
        beta: float, optional
            importance sample factor
            
        eps: float, optional
            small value to avoid division by zero
            
        obs_dtype: tc.dtype, optional
            observation data type
            
        act_dtype: tc.dtype, optional
            action data type
            
        rew_dtype: tc.dtype, optional
            reward data type
            
        device: tc.device, optional
            device where the memory replay is on"""

        #Check if the parameters are correct.
        if n_step <= 0:
            raise ValueError("n_step must be a positive integer.")
        
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        
        super().__init__(max_size, obs_size, alpha, beta, eps, obs_dtype, act_dtype, rew_dtype, device)
        self._n_step_buffer = NStepTransictionCollector(n_step)
        self._gamma = gamma

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        self._n_step_buffer.collect(obs, action, reward, next_obs, next_obs_done)

        if len(self._n_step_buffer) == self._n_step_buffer.n_step:
            transiction = self._n_step_buffer.get_transiction(self._gamma)
            super().store_transiction(transiction[0], transiction[1], transiction[2], transiction[3], transiction[4])