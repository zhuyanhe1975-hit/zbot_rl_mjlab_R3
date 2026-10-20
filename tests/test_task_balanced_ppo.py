"""Bounded checks for the optional task-balanced diagnostic algorithm."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from zbot_rl_mjlab.balanced_ppo import TaskBalancedPPO, normalize_task_advantages


def observations(tasks: torch.Tensor) -> torch.Tensor:
    obs = torch.zeros(*tasks.shape, 59)
    obs[..., 39:43] = torch.nn.functional.one_hot(tasks, 4).float()
    return obs


def test_matches_hand_computed_task_means_and_population_std():
    tasks = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    raw = torch.tensor([[1., 10., -5., 7.], [3., 20., -3., 7.], [5., 30., -1., 7.]])[..., None]
    result = normalize_task_advantages(raw, observations(tasks))
    expected = torch.tensor([-1., 0., 1.]) / (torch.tensor(2. / 3.).sqrt())
    for task in range(3):
        torch.testing.assert_close(result[:, task, 0], expected)
        torch.testing.assert_close(result[:, task, 0].mean(), torch.tensor(0.))
        torch.testing.assert_close(result[:, task, 0].std(correction=0), torch.tensor(1.))
    assert torch.equal(result[:, 3], torch.zeros(3, 1))


def test_independent_task_scales_and_offsets_do_not_suppress_small_group():
    tasks = torch.tensor([[0, 1], [0, 1], [0, 1]])
    raw = torch.tensor([[.01, 1.], [.02, 2.], [.03, 3.]])[..., None]
    changed = raw.clone()
    changed[:, 1] = changed[:, 1] * 100 + 30
    torch.testing.assert_close(
        normalize_task_advantages(raw, observations(tasks)),
        normalize_task_advantages(changed, observations(tasks)),
    )


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1)])
def test_empty_constant_and_singleton_groups_are_finite_zero(shape):
    tasks = torch.full(shape, 2, dtype=torch.long)
    result = normalize_task_advantages(torch.full((*shape, 1), 12.), observations(tasks))
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("raw_shape,obs_shape", [
    ((2, 3), (2, 3, 59)), ((2, 3, 2), (2, 3, 59)),
    ((2, 3, 1), (2, 3, 58)), ((2, 3, 1), (3, 2, 59)),
])
def test_rejects_wrong_shapes(raw_shape, obs_shape):
    with pytest.raises(ValueError):
        normalize_task_advantages(torch.zeros(raw_shape), torch.zeros(obs_shape))


def test_compute_returns_keeps_super_gae_value_targets_and_uses_raw_advantages(monkeypatch):
    tasks = torch.tensor([[0, 1], [0, 1]])
    obs = TensorDict({"actor": observations(tasks)}, batch_size=[2, 2])
    values = torch.tensor([[[2.], [3.]], [[4.], [5.]]])
    returns = values + torch.tensor([[[.01], [10.]], [[.03], [30.]]])
    storage = SimpleNamespace(values=values.clone(), returns=torch.zeros_like(returns), observations=obs)
    algorithm = object.__new__(TaskBalancedPPO)
    algorithm.storage = cast(Any, storage)
    algorithm.normalize_advantage_per_mini_batch = False
    calls = []

    def super_compute(self, final_obs):
        calls.append(final_obs)
        self.storage.returns.copy_(returns)
        self.storage.advantages = torch.full_like(returns, 123.)

    monkeypatch.setattr(PPO, "compute_returns", super_compute)
    final_obs = TensorDict({"actor": torch.zeros(2, 59)}, batch_size=[2])
    algorithm.compute_returns(final_obs)
    assert calls == [final_obs]
    torch.testing.assert_close(storage.returns, returns)
    torch.testing.assert_close(storage.values, values)
    torch.testing.assert_close(storage.advantages, normalize_task_advantages(returns - values, obs["actor"]))


def test_rejects_minibatch_normalization_at_construction_and_compute(monkeypatch):
    def base_init(self, *args, **kwargs):
        self.normalize_advantage_per_mini_batch = True

    monkeypatch.setattr(PPO, "__init__", base_init)
    with pytest.raises(ValueError, match="per-mini-batch"):
        TaskBalancedPPO()
    algorithm = object.__new__(TaskBalancedPPO)
    algorithm.normalize_advantage_per_mini_batch = True
    with pytest.raises(ValueError, match="per-mini-batch"):
        algorithm.compute_returns(TensorDict({}, batch_size=[]))


def test_scalar_layout_matches_legacy_task_balancing():
    tasks = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    legacy = observations(tasks)
    compact = torch.cat((legacy[..., :39], tasks[..., None], legacy[..., 43:]), dim=-1)
    raw = torch.tensor([[1., 10., 4., 2.], [3., 20., 8., 6.]])[..., None]
    torch.testing.assert_close(
        normalize_task_advantages(raw, compact), normalize_task_advantages(raw, legacy),
    )
