import torch as tc

from abc import ABC, abstractmethod

class Memory(ABC):
    """Base class of memory replay."""

    def __init__(self, max_size, obs_size, obs_dtype=tc.float32, act_dtype=tc.int8, rew_dtype=tc.float32, device=tc.device("cpu")):
        """Create new memory replay.
        
        Parameters
        --------------------
        max_size: int
            max size of memory replay
            
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
        
        #Check parameters.
        if max_size <= 0:
            raise ValueError("max size of memory replay must be at least 1 element")
        
        if isinstance(obs_size, int) and obs_size <= 0:
            raise ValueError("observation size must be positive integer")
        
        if act_dtype != tc.int8 and act_dtype != tc.int16 and act_dtype != tc.int32:
            raise ValueError("act_dtype can be either tc.int8, tc.int16 or tc.int32")

        # --------------------
        self._current_idx = 0                   #Current index this memory replay points to.
        self._current_size = 0                  #Current size of memory replay.
        self._max_size = max_size
        self._device = device

        #Memory replay.
        if isinstance(obs_size, tuple): 
            self._obss      = tc.zeros((max_size, *obs_size), dtype=obs_dtype).to(device)
            self._next_obss = tc.zeros((max_size, *obs_size), dtype=obs_dtype).to(device)
        else: 
            self._obss      = tc.zeros((max_size, obs_size), dtype=obs_dtype).to(device)
            self._next_obss = tc.zeros((max_size, obs_size), dtype=obs_dtype).to(device)
        self._actions        = tc.zeros(max_size, dtype=act_dtype).to(device)
        self._rewards        = tc.zeros(max_size, dtype=rew_dtype).to(device)
        self._next_obss_done = tc.zeros(max_size, dtype=tc.bool).to(device)

    def __len__(self):
        return self._current_size
    
    @property
    def max_size(self):
        return self._max_size

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        """Store transiction on memory replay.
        
        Parameters
        --------------------
        obs: tc.Tensor
            an observation
            
        action: int
            action choosen to perform on obs
            
        reward: float
            reward obtained to perform action
            
        next_obs: tc.Tensor
            next observation obtained performing action
            
        next_obs_done: bool
            True if next_obs is a terminal state, False otherwise"""
        
        #Store transiction into memory replay.
        self._obss[self._current_idx]           = obs
        self._actions[self._current_idx]        = action
        self._rewards[self._current_idx]        = reward
        self._next_obss[self._current_idx]      = next_obs
        self._next_obss_done[self._current_idx] = next_obs_done

        #Update current infos.
        self._current_idx = (self._current_idx + 1) % self._max_size
        if self._current_size < self._max_size:
            self._current_size += 1

    def _sample_batch_idxs(self, idxs_batch):
        """Sample batch from indices specified.
        
        Parameter
        --------------------
        idxs_batch: tc.Tensor
            indices to sample from memory replay
            
        Returns
        --------------------
        obs_batch: tc.Tensor
            observation batch
            
        action_batch: tc.Tensor
            action batch
            
        reward_batch: tc.Tensor
            reward batch
            
        next_batch: tc.Tensor
            next observations
            
        next_obs_done_batch: tc.Tensor
            next observations done batch"""
        
        assert idxs_batch.dtype == tc.int32 or idxs_batch.dtype == tc.int64, "idx_batch must be a tc.int32 or tc.int64 data type."

        return self._obss[idxs_batch], self._actions[idxs_batch], self._rewards[idxs_batch], self._next_obss[idxs_batch], self._next_obss_done[idxs_batch]

    @abstractmethod
    def sample_batch(self, batch_size):
        """Sample a batch from memory replay.
        
        Parameter
        --------------------
        batch_size: int
            batch size to sample
            
        Returns
        --------------------
        obs_batch: tc.Tensor
            observation batch sampled from memory replay
            
        action_batch: tc.Tensor
            action batch sampled from memory replay
            
        reward_batch: tc.Tensor
            reward batch sampled from memory replay
            
        next_batch: tc.Tensor
            next observations batch sampled from memory replay
            
        next_obs_done_batch: tc.Tensor
            next observations done batch sampled from memory replay"""
        
        pass