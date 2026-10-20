"""Opt-in diagnostic: normalize PPO advantages within each explicit FSM task.

Returns, value targets, rewards and the PPO update stay unchanged. This class is
specific to the teacher's raw scalar or legacy one-hot actor observation layout; it must
not receive the actor network's normalized observations.
"""

from typing import Any

import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .tasks.universal.observation_layout import teacher_task_ids


def normalize_task_advantages(
    raw_advantages: torch.Tensor, observations: torch.Tensor,
) -> torch.Tensor:
    """Center and population-standardize [T, N, 1] advantages per FSM task.

    All four group reductions remain on the input device. Empty groups have no
    output entries, while constant and singleton groups produce finite zeros.
    """
    if raw_advantages.ndim != 3 or raw_advantages.shape[-1] != 1:
        raise ValueError("Expected advantages with shape [T, N, 1]")
    if (
        observations.ndim != 3 or observations.shape[:2] != raw_advantages.shape[:2]
        or observations.shape[-1] not in (56, 59)
    ):
        raise ValueError("Expected matching raw teacher observations [T, N, 56 or 59]")
    if observations.device != raw_advantages.device:
        raise ValueError("Advantages and observations must share a device")

    tasks = teacher_task_ids(observations).reshape(-1)
    values = raw_advantages.reshape(-1)
    counts = values.new_zeros(4).scatter_add_(0, tasks, torch.ones_like(values))
    sums = values.new_zeros(4).scatter_add_(0, tasks, values)
    means = sums / counts.clamp_min(1)
    centered = values - means[tasks]
    variance = values.new_zeros(4).scatter_add_(0, tasks, centered.square())
    variance = variance / counts.clamp_min(1)
    return (centered / (variance[tasks].sqrt() + 1e-8)).reshape_as(raw_advantages)


class TaskBalancedPPO(PPO):
    """Optional four-task advantage normalization, independent of default PPO."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._check_normalization()

    def _check_normalization(self) -> None:
        if self.normalize_advantage_per_mini_batch:
            raise ValueError("TaskBalancedPPO requires per-mini-batch normalization disabled")

    def compute_returns(self, obs: TensorDict) -> None:
        self._check_normalization()
        super().compute_returns(obs)
        self.storage.advantages = normalize_task_advantages(
            self.storage.returns - self.storage.values,
            self.storage.observations["actor"],
        )
