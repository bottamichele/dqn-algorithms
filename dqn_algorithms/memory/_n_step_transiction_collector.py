from collections import deque

class NStepTransictionCollector:
    """A collector which stores transictions for N-step learning."""

    def __init__(self, n_step):
        """Create new collector of transictions.
        
        Parameter
        --------------------
        n_step: int
             of N-step learning"""
        
        self._buffer = deque(maxlen=n_step)
        self._n_step = n_step

    @property
    def n_step(self):
        return self._n_step
    
    def __len__(self):
        return len(self._buffer)
    
    def collect(self, obs, action, reward, next_obs, next_obs_done):
        """Collect one transiction.
        
        Parameters
        --------------------
        obs: tc.Tensor
            an observation
            
        action: int
            an action
            
        reward: float
            a reward scored by performing action
            
        next_obs: tc.Tensor
            an observation obtained by performing action in obs
            
        next_obs_done: bool
            True if next_obs is a terminal state, False otherwise"""
        
        self._buffer.append((obs, action, reward, next_obs, next_obs_done))

    def get_transiction(self, gamma):
        """Return the first transiction from the collector for N-step learning.
        
        Parameter
        --------------------
        gamma: float
            discount factor
            
        Returns
        --------------------
        obs: tc.Tensor
            an observation
            
        action: int
            chosen action from obs
            
        reward: float
            cumulative reward from obs to next_obs
            
        next_obs: tc.Tensor
            next observation
            
        next_obs_done: bool
            whether next_obs is a termina state"""
        
        assert len(self) == self._n_step
        assert gamma > 0 and gamma <= 1.0

        _, _, reward, next_obs, next_obs_done = self._buffer[-1]
        for transition in reversed(list(self._buffer)[:-1]):
            _, _, r, n_o, d = transition

            reward = r + gamma * reward * (1 - int(d))
            if d:
                next_obs, next_obs_done = (n_o, d)

        return self._buffer[0][0], self._buffer[0][1], reward, next_obs, next_obs_done