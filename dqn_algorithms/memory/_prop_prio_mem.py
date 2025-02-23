import torch as tc

from dqn_algorithms._c_modules.sum_tree import SumTree

from ._memory import Memory

class PropPriorMemory(Memory):
    """A proportional prioritized memory replay. It samples a batch in order to transiction's priority."""

    def __init__(self, max_size, obs_size, alpha=0.6, beta=0.4, eps=10**-5, obs_dtype=tc.float32, act_dtype=tc.int8, rew_dtype=tc.float32, device=tc.device("cpu")):
        """Create new memory replay.
        
        Parameters
        --------------------
        max_size: int
            max size of memory replay
            
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
        
        super().__init__(max_size, obs_size, obs_dtype, act_dtype, rew_dtype, device)

        self._priorities = tc.zeros(max_size, dtype=tc.float32, device=device)      #Priority for each transiction.
        self._cum_prios = SumTree(max_size)                                         #Cumulative priorities.
        self._alpha = alpha
        self.beta = beta
        self._epsilon = eps                                                         #Small value epsilon.
        self._idxs_sampled = None                                                   #Last sample of batch indices.

    def __reduce__(self):
        reduce_state = super().__reduce__()

        if "_cum_prios" in reduce_state[2].keys():
            dict_state = reduce_state[2].copy()
            dict_state.pop("_cum_prios")
            return reduce_state[0], reduce_state[1], dict_state
        else:
            return reduce_state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._cum_prios = SumTree(self._max_size)
        for i in range(self._current_size):
            self._cum_prios.set_priority(i, self._priorities[i] ** self._alpha)

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        #Set priority for current transiction.
        prio = self._priorities.max().item() if self._current_size > 0 else 1.0
        
        self._priorities[self._current_idx] = prio
        self._cum_prios.set_priority(self._current_idx, prio**self._alpha)

        #Store transiction on memory replay.
        super().store_transiction(obs, action, reward, next_obs, next_obs_done)

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
            next observations done batch sampled from memory replay
            
        weights_batch: tc.Tensor
            weights batch sampled from memory replay"""
        
        #Sample index of transictions and probabilties
        idxs = self._cum_prios.sample_batch(batch_size)
        probs = self._cum_prios.get_batch_probability(idxs)
        
        idxs = tc.Tensor(idxs).to(dtype=tc.int32, device=self._device)
        probs = tc.Tensor(probs).to(device=self._device)
        self._idxs_sampled = idxs

        #Compute weights.
        weights = (self._max_size * probs)**(-self.beta)
        weights /= weights.max().item()

        obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = self._sample_batch_idxs(idxs)
        return obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, weights
    
    def update_priorities(self, td_errors):
        """Update priorities of current batch sampled.
        
        Parameter
        --------------------
        td_errors: tc.Tensor
            temporal difference errors"""
        
        assert self._idxs_sampled.shape == td_errors.shape

        with tc.no_grad():
            self._priorities[self._idxs_sampled] = tc.abs(td_errors) + self._epsilon

        # for i in range(self._idxs_sampled.shape[0]):
        #     idx = self._idxs_sampled[i].item()
        #     self._priorities[idx] = abs(td_errors[i].item()) + self._epsilon