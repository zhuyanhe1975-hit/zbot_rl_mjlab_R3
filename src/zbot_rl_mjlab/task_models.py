"""Task-routed actor and critic models for the 56-feature universal teacher.

Ported without numerical changes from the bounded routed-task training diagnostic.
Each task has its own actor parameters and input moments; PPO's optimizer,
learning-rate adaptation and gradient clipping remain shared.
"""

from typing import Any, cast

import torch
from rsl_rl.models import MLPModel
from torch import nn

from .task_ppo import IndependentTaskCritics


class EmpiricalTaskNormalization(nn.Module):
    """Population moments per raw FSM task, updated once after each rollout."""

    _mean: torch.Tensor
    _var: torch.Tensor
    _std: torch.Tensor
    count: torch.Tensor

    def __init__(self, shape: int = 56, eps: float = .01) -> None:
        super().__init__()
        if shape != 56:
            raise ValueError('Task normalization requires 56 raw features')
        self.eps = eps
        self.register_buffer('_mean', torch.zeros(4, shape))
        self.register_buffer('_var', torch.ones(4, shape))
        self.register_buffer('_std', torch.ones(4, shape))
        self.register_buffer('count', torch.zeros(4, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tasks = x[:, 39].round().long()
        normalized = (x - self._mean[tasks]) / (self._std[tasks] + self.eps)
        # Constant codes within a task would vanish if centered/standardized.
        return torch.cat((normalized[:, :38], x[:, 38:40], normalized[:, 40:]), dim=-1)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.training:
            return
        tasks = x[:, 39].round().long()
        index = tasks[:, None].expand(-1, x.shape[-1])
        batch_count = torch.zeros_like(self.count).scatter_add_(0, tasks, torch.ones_like(tasks))
        batch_sum = torch.zeros_like(self._mean).scatter_add_(0, index, x)
        batch_mean = batch_sum / batch_count[:, None].clamp_min(1)
        centered = x - batch_mean[tasks]
        batch_var = torch.zeros_like(self._var).scatter_add_(0, index, centered.square())
        batch_var /= batch_count[:, None].clamp_min(1)
        next_count = self.count + batch_count
        rate = batch_count[:, None] / next_count[:, None].clamp_min(1)
        delta = batch_mean - self._mean
        next_mean = self._mean + rate * delta
        next_var = self._var + rate * (batch_var - self._var + delta * (batch_mean - next_mean))
        # With no samples rate==0, retaining both initial and existing moments.
        self._mean.copy_(next_mean)
        self._var.copy_(next_var.clamp_min(0))
        self._std.copy_(self._var.sqrt())
        self.count.copy_(next_count)


class TaskNormalizedMLP(MLPModel):
    """Ordinary Beta actor with four independent input-moment tables only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.obs_dim != 56 or not self.obs_normalization:
            raise ValueError('TaskNormalizedMLP requires obs_normalization=True and 56 raw features')
        self.obs_normalizer = EmpiricalTaskNormalization(self.obs_dim)


class TaskRoutedNetwork(nn.Module):
    """Independent initialized actor heads selected by the preserved raw task code."""

    def __init__(self, initial: nn.Module) -> None:
        super().__init__()
        # Copy initialization, never a previously trained policy checkpoint.
        import copy

        self.heads = nn.ModuleList([copy.deepcopy(initial) for _ in range(4)])

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([head(latent) for head in self.heads], dim=1)
        tasks = latent[:, 39].round().long()
        index = tasks.reshape(-1, 1, 1).expand(-1, 1, outputs.shape[-1])
        # Distribution heads can have [N, 2, actions] structured output.
        if outputs.dim() == 4:
            index = tasks.reshape(-1, 1, 1, 1).expand(-1, 1, outputs.shape[-2], outputs.shape[-1])
        return outputs.gather(1, index).squeeze(1)


class RoutedTaskNormalizedMLP(TaskNormalizedMLP):
    """Task-routed parameters with one existing Beta distribution interface.

    Shared optimizer, LR adaptation and global gradient clipping still couple the
    heads. This is not equivalent to four independently trained PPO algorithms.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = TaskRoutedNetwork(self.mlp)


class TaskNormalizedCritics(IndependentTaskCritics):
    """Independent task critics with task-separated input moments in every head."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.obs_dim != 56:
            raise ValueError('TaskNormalizedCritics requires 56 raw features')
        for child in self.critics:
            model = cast(MLPModel, child)
            if not model.obs_normalization:
                raise ValueError('TaskNormalizedCritics requires obs_normalization=True')
            model.add_module("obs_normalizer", EmpiricalTaskNormalization(56))


class OneHotTaskEncoding(nn.Module):
    """Replace the raw task code by four categories and a recovery residual.

    Normalization and FSM routing still consume the 56-feature observation. Only
    the trainable MLP consumes these 60 features; morphology remains unchanged.
    The existing ascent recovery code 3.25 retains its observable .25 residual.
    """

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        code = raw[:, 39:40]
        task = code.round()
        categorical = torch.nn.functional.one_hot(task.squeeze(-1).long(), 4).to(raw.dtype)
        return torch.cat((raw[:, :39], categorical, code - task, raw[:, 40:]), dim=-1)


class OneHotRoutedTaskNormalizedMLP(RoutedTaskNormalizedMLP):
    """Independent actor heads with categorical task features before every MLP."""

    def _get_latent_dim(self) -> int:
        # MLPModel constructs and initializes the appropriately sized RSL-RL MLP.
        return 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        routed = cast(TaskRoutedNetwork, self.mlp)
        routed.heads = nn.ModuleList([
            nn.Sequential(OneHotTaskEncoding(), head) for head in routed.heads
        ])


class _OneHotTaskNormalizedMLP(TaskNormalizedMLP):
    """RSL-RL scalar critic with raw moments and encoded trainable inputs."""

    def _get_latent_dim(self) -> int:
        return 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(OneHotTaskEncoding(), self.mlp)


class OneHotTaskNormalizedCritics(TaskNormalizedCritics):
    """Independent task critics using the same categorical features as the actor."""

    def __init__(
        self, obs: Any, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, **kwargs: Any,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.critics = nn.ModuleList([
            _OneHotTaskNormalizedMLP(obs, obs_groups, obs_set, output_dim, **kwargs)
            for _ in self.critics
        ])


class LocalFrameTaskNormalization(EmpiricalTaskNormalization):
    """Keep the circular heading raw until its sine/cosine encoding.

    Moment updates retain the historical 56-feature, per-task layout. Heading
    moments are recorded for compatibility but never applied to the angle.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tasks = x[:, 39].round().long()
        normalized = (x - self._mean[tasks]) / (self._std[tasks] + self.eps)
        return torch.cat((
            normalized[:, :38], x[:, 38:40], normalized[:, 40:46],
            x[:, 46:47], normalized[:, 47:],
        ), dim=-1)


class LocalFrameTaskEncoding(OneHotTaskEncoding):
    """Categorical task features plus a continuous circular heading pair."""

    def __init__(self) -> None:
        super().__init__()
        self.task_encoding = OneHotTaskEncoding()

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        encoded = self.task_encoding(raw)
        heading = raw[:, 46:47]
        return torch.cat((encoded[:, :50], heading.sin(), heading.cos(), encoded[:, 51:]), dim=-1)


class LocalFrameRoutedTaskNormalizedMLP(RoutedTaskNormalizedMLP):
    """Versioned local-frame actor with raw56 input and 61 trainable features."""

    def _get_latent_dim(self) -> int:
        return 61

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.obs_normalizer = LocalFrameTaskNormalization(self.obs_dim)
        routed = cast(TaskRoutedNetwork, self.mlp)
        routed.heads = nn.ModuleList([
            nn.Sequential(LocalFrameTaskEncoding(), head) for head in routed.heads
        ])


class _LocalFrameTaskNormalizedMLP(TaskNormalizedMLP):
    """Scalar critic with raw heading preservation and circular encoding."""

    def _get_latent_dim(self) -> int:
        return 61

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.obs_normalizer = LocalFrameTaskNormalization(self.obs_dim)
        self.mlp = nn.Sequential(LocalFrameTaskEncoding(), self.mlp)


class LocalFrameTaskNormalizedCritics(TaskNormalizedCritics):
    """Independent task critics matching the local-frame actor encoding."""

    def __init__(
        self, obs: Any, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, **kwargs: Any,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.critics = nn.ModuleList([
            _LocalFrameTaskNormalizedMLP(obs, obs_groups, obs_set, output_dim, **kwargs)
            for _ in self.critics
        ])
