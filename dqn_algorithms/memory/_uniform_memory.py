import torch as tc

from ._memory import Memory

class UniformMemory(Memory):
    """A memory replay that ramdomly samples a batch from."""

    def sample_batch(self, batch_size):
        indices_batch = tc.randperm(self._current_size, device=self._device)[:batch_size]
        return self._sample_batch_idxs(indices_batch)