import numpy as np

from abc import ABC, abstractmethod

class Memory(ABC):
    """Base class of memory replay."""

    def __init__(self, max_size, obs_size, obs_dtype=np.float32, act_dtype=np.int8, rew_dtype=np.float32):
        """Create new memory replay.
        
        Parameters
        --------------------
        max_size: int
            max size of memory replay
            
        obs_size: int
            observation size
            
        obs_dtype: np.dtype, optional
            observation data type
            
        act_dtype: np.dtype, optional
            action data type
            
        rew_dtype: np.dtype, optional
            reward data type"""
        
        #Check parameters.
        if max_size <= 0:
            raise ValueError("max size of memory replay must be at least 1 element")
        
        if obs_size <= 0:
            raise ValueError("observation size must be positive integer")
        
        if act_dtype != np.int8 and act_dtype != np.int16 and act_dtype != np.int32:
            raise ValueError("act_dtype can be either np.int8, np.int16 or np.int32")

        # --------------------
        self._current_idx = 0                   #Current index this memory replay points to.
        self._current_size = 0                  #Current size of memory replay.
        self._max_size = max_size

        #Memory replay.
        self._obss           = np.zeros((max_size, obs_size), dtype=obs_dtype)
        self._actions        = np.zeros(max_size, dtype=act_dtype)
        self._rewards        = np.zeros(max_size, dtype=rew_dtype)
        self._next_obss      = np.zeros((max_size, obs_size), dtype=obs_dtype)
        self._next_obss_done = np.zeros(max_size, dtype=bool)

    def __len__(self):
        return self._current_size
    
    @property
    def max_size(self):
        return self._max_size

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        """Store transiction on memory replay.
        
        Parameters
        --------------------
        obs: np.ndarray
            an observation
            
        action: int
            action choosen to perform on obs
            
        reward: float
            reward obtained to perform action
            
        next_obs: np.ndarray
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
        idxs_batch: np.ndarray
            indices to sample from memory replay
            
        Returns
        --------------------
        obs_batch: np.ndarray
            observation batch
            
        action_batch: np.ndarray
            action batch
            
        reward_batch: np.ndarray
            reward batch
            
        next_batch: np.ndarray
            next observations
            
        next_obs_done_batch: np.ndarray
            next observations done batch"""
        
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
        obs_batch: np.ndarray
            observation batch sampled from memory replay
            
        action_batch: np.ndarray
            action batch sampled from memory replay
            
        reward_batch: np.ndarray
            reward batch sampled from memory replay
            
        next_batch: np.ndarray
            next observations batch sampled from memory replay
            
        next_obs_done_batch: np.ndarray
            next observations done batch sampled from memory replay"""
        
        pass