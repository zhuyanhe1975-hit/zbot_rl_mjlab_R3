"""Task-native control dimensions with Gaussian exploration for every skill."""

from collections.abc import Sequence
from typing import Any, cast

import torch
from rsl_rl.modules import MLP, GaussianDistribution
from rsl_rl.modules.distribution import Distribution
from torch import nn

from .task_models import (
    LocalFrameRoutedTaskNormalizedMLP,
    LocalFrameTaskEncoding,
    TaskRoutedNetwork,
)


def expand_mirror(actions: torch.Tensor) -> torch.Tensor:
    """Expand three independent transform controls in the archived joint order."""
    return torch.cat((actions, -actions.flip(-1)), dim=-1)


class _PackGaussianHead(nn.Module):
    def __init__(self, task: int) -> None:
        super().__init__()
        self.task = task
        self.mirrored = task == 1 or task == 3

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        mean = expand_mirror(output) if self.mirrored else output
        return torch.stack((mean, torch.full_like(mean, float(self.task))), dim=-2)


class _GaussianDeterministicOutput(nn.Module):
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed[..., 0, :]


class GaussianTaskDistribution(Distribution):
    """Four independent RSL Gaussian distributions with intrinsic 6/3/6/3 axes.

    External actions always contain six joint controls. The mirror manifold's
    likelihood, entropy and KL use only its three independent controls.
    """

    def __init__(
        self, output_dim: int, init_std: float = 1., std_type: str = 'scalar',
        std_range: tuple[float, float] = (1e-6, 1e6), learn_std: bool = True,
        rate_init_std: float = .5,
    ) -> None:
        if output_dim != 6:
            raise ValueError('Gaussian task actions require six joint outputs.')
        super().__init__(output_dim)
        self.gaussians = cast(Sequence[GaussianDistribution], nn.ModuleList([
            GaussianDistribution(dim, init_std if task == 0 else rate_init_std, std_range, std_type, learn_std)
            for task, dim in enumerate((6, 3, 6, 3))
        ]))

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    def update(self, mlp_output: torch.Tensor) -> None:
        self._tasks = mlp_output[..., 1, :1]
        for task, gaussian in enumerate(self.gaussians):
            gaussian.update(mlp_output[..., 0, :3] if task in (1, 3) else mlp_output[..., 0, :])

    def _select(self, values: list[torch.Tensor]) -> torch.Tensor:
        result = values[0]
        for task in range(1, 4):
            mask = self._tasks.squeeze(-1) == task if result.dim() == 1 else self._tasks == task
            result = torch.where(mask, values[task], result)
        return result

    def sample(self) -> torch.Tensor:
        return self._select([
            expand_mirror(gaussian.sample()) if task in (1, 3) else gaussian.sample()
            for task, gaussian in enumerate(self.gaussians)
        ])

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        return _GaussianDeterministicOutput()

    @property
    def mean(self) -> torch.Tensor:
        return self._select([
            expand_mirror(gaussian.mean) if task in (1, 3) else gaussian.mean
            for task, gaussian in enumerate(self.gaussians)
        ])

    @property
    def std(self) -> torch.Tensor:
        return self._select([
            torch.cat((gaussian.std, gaussian.std.flip(-1)), dim=-1) if task in (1, 3) else gaussian.std
            for task, gaussian in enumerate(self.gaussians)
        ])

    @property
    def entropy(self) -> torch.Tensor:
        return self._select([gaussian.entropy for gaussian in self.gaussians])

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return self.mean, self.std, self._tasks

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        on_manifold = (outputs == expand_mirror(outputs[..., :3])).all(dim=-1)
        values = []
        for task, gaussian in enumerate(self.gaussians):
            if task in (1, 3):
                value = gaussian.log_prob(outputs[..., :3])
                values.append(torch.where(on_manifold, value, -torch.inf))
            else:
                values.append(gaussian.log_prob(outputs))
        return self._select(values)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # Select using the stored task identity, independent of the latest update.
        result = self.gaussians[0].kl_divergence(old_params[:2], new_params[:2])
        for task in range(1, 4):
            width = 3 if task in (1, 3) else 6
            old = tuple(value[..., :width] for value in old_params[:2])
            new = tuple(value[..., :width] for value in new_params[:2])
            value = self.gaussians[task].kl_divergence(old, new)
            result = torch.where(old_params[2].squeeze(-1) == task, value, result)
        return result

    def clear_inactive_distribution_gradients(self, active_tasks: set[int]) -> None:
        for task, gaussian in enumerate(self.gaussians):
            if task not in active_tasks:
                for parameter in gaussian.parameters():
                    parameter.grad = None


class GaussianActionRoutedTaskNormalizedMLP(LocalFrameRoutedTaskNormalizedMLP):
    """Versioned raw56/local61 actor with Gaussian heads of widths 6/3/6/3."""

    native_biped_actions = True
    mirrored_transform_actions = True
    all_gaussian_actions = True

    def __init__(
        self, obs: Any, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = 'elu', obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        config = dict(distribution_cfg or {})
        config.pop('class_name', None)
        config.pop('action_range', None)  # Legacy generic actor factories may pass Beta's range.
        # The parent initializes normalization/routing; all temporary heads and
        # its distribution are replaced below by native Gaussian controls.
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims=hidden_dims,
            activation=activation, obs_normalization=obs_normalization,
            distribution_cfg={'class_name': 'GaussianDistribution'},
        )
        self.distribution = GaussianTaskDistribution(output_dim, **config)
        router = cast(TaskRoutedNetwork, self.mlp)
        router.heads = nn.ModuleList([
            nn.Sequential(LocalFrameTaskEncoding(), MLP(61, width, hidden_dims, activation), _PackGaussianHead(task))
            for task, width in enumerate((6, 3, 6, 3))
        ])

    def clear_inactive_distribution_gradients(self, active_tasks: set[int]) -> None:
        cast(GaussianTaskDistribution, self.distribution).clear_inactive_distribution_gradients(active_tasks)
