"""Task routing around the archived teachers' native RSL-RL distributions.

Task zero uses unbounded Gaussian position actions; the remaining tasks retain
bounded Beta velocity actions. Distribution mathematics stays in RSL-RL.
"""

from typing import Any, cast

import torch
from rsl_rl.modules import MLP, BetaDistribution, GaussianDistribution
from rsl_rl.modules.distribution import Distribution
from torch import nn

from .task_models import (
    LocalFrameRoutedTaskNormalizedMLP,
    LocalFrameTaskEncoding,
    TaskRoutedNetwork,
)


class _PackNativeHead(nn.Module):
    def __init__(self, gaussian: bool) -> None:
        super().__init__()
        self.gaussian = gaussian

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if self.gaussian:
            return torch.stack((output, torch.zeros_like(output), torch.ones_like(output)), dim=-2)
        return torch.cat((output, torch.zeros_like(output[..., :1, :])), dim=-2)


class _NativeDeterministicOutput(nn.Module):
    def __init__(self, beta: BetaDistribution) -> None:
        super().__init__()
        self.beta = beta.as_deterministic_output_module()

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return torch.where(packed[..., 2, :] > .5, packed[..., 0, :], self.beta(packed[..., :2, :]))


class NativeTaskDistribution(Distribution):
    """Select native Gaussian/Beta outputs using an explicit packed task flag."""

    def __init__(
        self, output_dim: int, init_std: float = 1.0, std_type: str = 'scalar',
        action_range: tuple[float, float] = (-1., 1.),
    ) -> None:
        super().__init__(output_dim)
        self.gaussian = GaussianDistribution(output_dim, init_std=init_std, std_type=std_type)
        self.beta = BetaDistribution(output_dim, action_range=action_range)

    @property
    def input_dim(self) -> list[int]:
        return [3, self.output_dim]

    def update(self, mlp_output: torch.Tensor) -> None:
        self._gaussian_mask = mlp_output[..., 2, :1] > .5
        self.gaussian.update(mlp_output[..., 0, :])
        self.beta.update(mlp_output[..., :2, :])

    def sample(self) -> torch.Tensor:
        return torch.where(self._gaussian_mask, self.gaussian.sample(), self.beta.sample())

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.where(
            mlp_output[..., 2, :] > .5, mlp_output[..., 0, :],
            self.beta.deterministic_output(mlp_output[..., :2, :]),
        )

    def as_deterministic_output_module(self) -> nn.Module:
        return _NativeDeterministicOutput(self.beta)

    @property
    def mean(self) -> torch.Tensor:
        return torch.where(self._gaussian_mask, self.gaussian.mean, self.beta.mean)

    @property
    def std(self) -> torch.Tensor:
        return torch.where(self._gaussian_mask, self.gaussian.std, self.beta.std)

    @property
    def entropy(self) -> torch.Tensor:
        return torch.where(self._gaussian_mask.squeeze(-1), self.gaussian.entropy, self.beta.entropy)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (*self.gaussian.params, *self.beta.params, self._gaussian_mask)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        # Unbounded Gaussian samples must never enter the bounded likelihood.
        beta_outputs = torch.where(self._gaussian_mask, torch.zeros_like(outputs), outputs)
        return torch.where(
            self._gaussian_mask.squeeze(-1), self.gaussian.log_prob(outputs), self.beta.log_prob(beta_outputs),
        )

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return torch.where(
            old_params[4].squeeze(-1).bool(),
            self.gaussian.kl_divergence(old_params[:2], new_params[:2]),
            self.beta.kl_divergence(old_params[2:4], new_params[2:4]),
        )


class NativeActionRoutedTaskNormalizedMLP(LocalFrameRoutedTaskNormalizedMLP):
    """Keep raw56/local61/task-separated actors with native action interfaces."""

    native_biped_actions = True

    def __init__(
        self, obs: Any, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = 'elu', obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        native_cfg = dict(distribution_cfg or {})
        native_cfg.pop('class_name', None)
        # The parent constructs the archived Beta heads and applies RSL's own
        # initialization before they are wrapped in the common packed shape.
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims,
            activation=activation, obs_normalization=obs_normalization,
            distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)},
        )
        router = cast(TaskRoutedNetwork, self.mlp)
        router.heads[0] = nn.Sequential(
            LocalFrameTaskEncoding(), MLP(61, output_dim, hidden_dims, activation), _PackNativeHead(True),
        )
        for index in range(1, 4):
            cast(nn.Sequential, router.heads[index]).append(_PackNativeHead(False))
        self.distribution = NativeTaskDistribution(output_dim, **native_cfg)
