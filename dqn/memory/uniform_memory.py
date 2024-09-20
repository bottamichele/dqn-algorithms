import numpy as np

from .memory import Memory

class UniformMemory(Memory):
    """A memory replay that ramdomly samples a batch from."""

    def __init__(self, max_size, obs_size, obs_dtype=np.float32, act_dtype=np.int8, rew_dtype=np.float32):
        super().__init__(max_size, obs_size, obs_dtype, act_dtype, rew_dtype)
        self._rng = np.random.default_rng()

    def sample_batch(self, batch_size):
        indices_batch = self._rng.choice(self._current_size, batch_size, False)
        return self._sample_batch_idxs(indices_batch)