"""A circular replay buffer of self-play positions (cf. raccoon's ReplayBuffer).

Holds the most recent ``max_size`` training examples and serves random batches
as PyTorch tensors. Old games fall out the back as new ones arrive -- the buffer
size therefore controls how "fresh" the training data is (a knob explored in B3).
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from azl.selfplay import TrainingExample


class ReplayBuffer:
    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add_game(self, examples: list[TrainingExample]) -> None:
        self._buffer.extend(examples)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(batch_size, len(self._buffer))
        samples = random.sample(list(self._buffer), n)
        obs = torch.from_numpy(np.stack([s.observation for s in samples])).float()
        policy = torch.from_numpy(np.stack([s.policy_target for s in samples])).float()
        value = torch.from_numpy(np.array([s.value_target for s in samples], dtype=np.float32))
        return obs, policy, value
