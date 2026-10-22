"""History student using the universal teacher's task and action contracts.

MJLab flattens each observation term's chronological history independently. The
student therefore requires one complete 56-feature frame term, oldest first.
"""

from collections.abc import Sequence
from typing import Any, cast

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from torch import nn

from .gaussian_action_models import GaussianTaskDistribution, _PackGaussianHead
from .task_models import LocalFrameTaskEncoding, LocalFrameTaskNormalization


class HistoryTaskNormalization(nn.Module):
    """Reuse per-task moments across historical frames without mixing tasks."""

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.history_length = history_length
        self.frame_normalizer = LocalFrameTaskNormalization(56)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        frames = history.reshape(-1, 56)
        return self.frame_normalizer(frames).reshape(history.shape[0], self.history_length * 56)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, history: torch.Tensor) -> None:
        self.frame_normalizer.update(history.reshape(-1, 56))


class HistoryTaskEncoding(nn.Module):
    """Encode each frame's task category and heading before trainable layers."""

    def __init__(self, history_length: int) -> None:
        super().__init__()
        self.history_length = history_length
        self.frame_encoding = LocalFrameTaskEncoding()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.frame_encoding(history.reshape(-1, 56)).reshape(history.shape[0], self.history_length * 61)


class HistoryTaskRoutedNetwork(nn.Module):
    """Route using the newest frame, including when history spans a transition."""

    def __init__(
        self, history_length: int, hidden_dims: Sequence[int], activation: str,
    ) -> None:
        super().__init__()
        self.task_index = (history_length - 1) * 56 + 39
        self.heads = nn.ModuleList([
            nn.Sequential(
                HistoryTaskEncoding(history_length),
                MLP(history_length * 61, width, list(hidden_dims), activation),
                _PackGaussianHead(task),
            )
            for task, width in enumerate((6, 3, 6, 3))
        ])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([head(latent) for head in self.heads], dim=1)
        tasks = latent[:, self.task_index].round().long()
        index = tasks.reshape(-1, 1, 1, 1).expand(-1, 1, 2, 6)
        return outputs.gather(1, index).squeeze(1)


class HistoryGaussianStudentMLP(MLPModel):
    """RSL-compatible history actor with independent Gaussian 6/3/6/3 heads."""

    native_biped_actions = True
    mirrored_transform_actions = True
    all_gaussian_actions = True

    def __init__(
        self, obs: Any, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = 'elu', obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims,
            activation=activation, obs_normalization=obs_normalization,
            distribution_cfg=None,
        )
        if self.obs_dim < 56 or self.obs_dim % 56 or len(self.obs_groups) != 1:
            raise ValueError('Student requires a single flattened history group of complete 56-feature frames')
        if not obs_normalization:
            raise ValueError('Student requires per-task observation normalization')
        self.history_length = self.obs_dim // 56
        self.obs_normalizer = HistoryTaskNormalization(self.history_length)
        config = dict(distribution_cfg or {})
        config.pop('class_name', None)
        self.distribution = GaussianTaskDistribution(output_dim, **config)
        self.mlp = HistoryTaskRoutedNetwork(self.history_length, hidden_dims, activation)

    def clear_inactive_distribution_gradients(self, active_tasks: set[int]) -> None:
        cast(GaussianTaskDistribution, self.distribution).clear_inactive_distribution_gradients(active_tasks)
