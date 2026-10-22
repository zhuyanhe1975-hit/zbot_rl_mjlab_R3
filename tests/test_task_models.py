"""Numerical and deployment regressions for the validated routed-task models."""

import copy
import io
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from zbot_rl_mjlab.task_models import (
    EmpiricalTaskNormalization,
    RoutedTaskNormalizedMLP,
    TaskNormalizedCritics,
    TaskNormalizedMLP,
    TaskRoutedNetwork,
)


def raw(tasks: tuple[int, ...] = (0, 1, 2, 3) * 8) -> torch.Tensor:
    ids = torch.tensor(tasks)
    x = torch.randn(len(tasks), 56) * (ids[:, None] + 1) + ids[:, None] * 10
    x[:, 39] = ids
    x[:, 38] = torch.tensor([1. if t == 0 else -1. if t == 2 else 0. for t in tasks])
    return x


def observations(x: torch.Tensor) -> TensorDict:
    return TensorDict({'actor': x.clone(), 'critic': x.clone()}, batch_size=[x.shape[0]])


def actor(cls: type[TaskNormalizedMLP], obs: TensorDict) -> TaskNormalizedMLP:
    return cls(obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
               distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)})


def critics(obs: TensorDict) -> TaskNormalizedCritics:
    return TaskNormalizedCritics(obs, {'critic': ['critic']}, 'critic', 1,
                                 hidden_dims=(16,), obs_normalization=True)


def heads(model: TaskNormalizedMLP) -> torch.nn.ModuleList:
    return cast(TaskRoutedNetwork, model.mlp).heads


def normalizer(model: MLPModel | TaskNormalizedMLP) -> EmpiricalTaskNormalization:
    return cast(EmpiricalTaskNormalization, model.obs_normalizer)


def test_population_moments_per_task():
    x = raw()
    norm = EmpiricalTaskNormalization()
    norm.update(x)
    for task in range(4):
        subset = x[x[:, 39] == task]
        assert torch.allclose(norm._mean[task], subset.mean(0), atol=1e-5)
        assert torch.allclose(norm._var[task], subset.var(0, unbiased=False), atol=1e-5)
    assert torch.equal(norm.count, torch.tensor([8, 8, 8, 8]))


def test_codes_preserved_exactly():
    x = raw()
    norm = EmpiricalTaskNormalization()
    norm.update(x)
    assert torch.equal(norm(x)[:, 38:40], x[:, 38:40])


def test_other_task_extremes_do_not_change_biped_statistics():
    x = raw()
    biped = x[x[:, 39] == 0]
    other = x[x[:, 39] != 0].clone()
    other[:, :38] += 1e6
    other[:, 40:] -= 1e6
    solo, mixed = EmpiricalTaskNormalization(), EmpiricalTaskNormalization()
    solo.update(biped)
    mixed.update(torch.cat((biped, other)))
    assert torch.equal(solo._mean[0], mixed._mean[0])
    assert torch.equal(solo._var[0], mixed._var[0])
    assert torch.equal(solo(biped), mixed(biped))


def test_empty_tasks_retain_initial_moments():
    norm = EmpiricalTaskNormalization()
    norm.update(raw((0,) * 8))
    assert torch.equal(norm.count, torch.tensor([8, 0, 0, 0]))
    assert torch.equal(norm._mean[1:], torch.zeros(3, 56))
    assert torch.equal(norm._var[1:], torch.ones(3, 56))


def test_empty_tasks_retain_existing_moments():
    norm = EmpiricalTaskNormalization()
    norm.update(raw())
    before = copy.deepcopy(norm.state_dict())
    norm.update(raw((0,) * 8))
    assert all(torch.equal(value[1:], norm.state_dict()[key][1:]) for key, value in before.items())


def test_incremental_population_merge_matches_full_batch():
    x = raw()
    seq, full = EmpiricalTaskNormalization(), EmpiricalTaskNormalization()
    seq.update(x[:16])
    seq.update(x[16:])
    full.update(x)
    assert torch.allclose(seq._mean, full._mean, atol=1e-5)
    assert torch.allclose(seq._var, full._var, atol=1e-5)


def test_eval_freezes_update():
    norm = EmpiricalTaskNormalization()
    norm.update(raw())
    before = copy.deepcopy(norm.state_dict())
    norm.eval()
    norm.update(raw() * 100)
    assert all(torch.equal(value, norm.state_dict()[key]) for key, value in before.items())


def test_forward_never_updates_normalization():
    obs = observations(raw())
    model = actor(TaskNormalizedMLP, obs)
    for _ in range(3):
        assert model(obs).shape == (32, 6)
    assert torch.equal(normalizer(model).count, torch.zeros(4, dtype=torch.long))
    model.update_normalization(obs)
    assert torch.equal(normalizer(model).count, torch.tensor([8, 8, 8, 8]))


def test_actor_state_serialization_roundtrip():
    obs = observations(raw())
    model = actor(TaskNormalizedMLP, obs)
    model.update_normalization(obs)
    restored = actor(TaskNormalizedMLP, obs)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(model(obs), restored(obs))


@pytest.mark.parametrize('cls', [TaskNormalizedMLP, RoutedTaskNormalizedMLP])
def test_jit_deterministic_export_parity(cls):
    x = raw()
    obs = observations(x)
    model = actor(cls, obs)
    model.update_normalization(obs)
    scripted = torch.jit.script(model.as_jit())
    assert torch.equal(scripted(x), model(obs))


def test_pure_task0_matches_unrouted_outputs_and_policy_gradients():
    obs = observations(raw((0,) * 8))
    torch.manual_seed(42)
    plain = actor(TaskNormalizedMLP, obs)
    torch.manual_seed(42)
    routed = actor(RoutedTaskNormalizedMLP, obs)
    for model in (plain, routed):
        model.update_normalization(obs)
    assert torch.equal(plain(obs), routed(obs))
    for model in (plain, routed):
        torch.manual_seed(44)
        sampled = model(obs, stochastic_output=True)
        (-model.get_output_log_prob(sampled.detach()).mean()).backward()
    for left, right in zip(plain.mlp.parameters(), heads(routed)[0].parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.equal(left.grad, right.grad)
    for head in list(heads(routed))[1:]:
        assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in head.parameters())


def test_task2_sgd_leaves_task0_parameters_unchanged():
    obs = observations(raw((2,) * 8))
    model = actor(RoutedTaskNormalizedMLP, obs)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    before = copy.deepcopy(heads(model)[0].state_dict())
    optimizer.zero_grad()
    actions = model(obs, stochastic_output=True)
    (-model.get_output_log_prob(actions.detach()).mean()).backward()
    optimizer.step()
    assert all(torch.equal(value, heads(model)[0].state_dict()[key]) for key, value in before.items())


def test_raw_task_selects_corresponding_branch():
    x = raw((0, 1, 2, 3))
    model = actor(RoutedTaskNormalizedMLP, observations(x))
    for task, head in enumerate(heads(model)):
        for parameter in head.parameters():
            parameter.data.zero_()
        last = next(layer for layer in reversed(list(head.modules())) if isinstance(layer, torch.nn.Linear))
        assert last.bias is not None
        last.bias.data.fill_(task + 1.)
    selected = model.mlp(model.obs_normalizer(x))
    assert selected.shape == (4, 2, 6)
    for task in range(4):
        assert torch.equal(selected[task], torch.full((2, 6), task + 1.))


def test_routed_heads_start_with_identical_independent_weights():
    model = actor(RoutedTaskNormalizedMLP, observations(raw()))
    first = next(heads(model)[0].parameters())
    for head in list(heads(model))[1:]:
        weight = next(head.parameters())
        assert torch.equal(first, weight)
        assert first.data_ptr() != weight.data_ptr()


def test_critic_normalization_isolated_from_other_task_extremes():
    x = raw()
    obs = observations(x)
    base = critics(obs)
    base.update_normalization(obs)
    extreme, control = copy.deepcopy(base), copy.deepcopy(base)
    changed = x.clone()
    changed[x[:, 39] == 2, :38] += 1e6
    changed[x[:, 39] == 2, 40:] -= 1e6
    extreme.update_normalization(observations(changed))
    control.update_normalization(obs)
    for left, right in zip(extreme.critics, control.critics, strict=True):
        biped = x[x[:, 39] == 0]
        assert torch.equal(normalizer(cast(MLPModel, left))(biped), normalizer(cast(MLPModel, right))(biped))


def test_inactive_critic_values_are_zero():
    x = raw()
    values = critics(observations(x))(observations(x))
    mask = torch.nn.functional.one_hot(x[:, 39].long(), 4)
    assert values.shape == (32, 4)
    assert torch.equal(values[mask == 0], torch.zeros_like(values[mask == 0]))


def factory(obs: TensorDict) -> PPO:
    cfg = {
        'algorithm': {'class_name': 'zbot_rl_mjlab.task_ppo:TaskPPO', 'num_learning_epochs': 1, 'num_mini_batches': 1},
        'actor': {'class_name': 'zbot_rl_mjlab.task_models:RoutedTaskNormalizedMLP', 'hidden_dims': (16,),
                  'obs_normalization': True, 'distribution_cfg': {'class_name': 'BetaDistribution', 'action_range': (-1., 1.)}},
        'critic': {'class_name': 'zbot_rl_mjlab.task_models:TaskNormalizedCritics', 'hidden_dims': (16,), 'obs_normalization': True},
        'obs_groups': {'actor': ['actor'], 'critic': ['critic']}, 'num_steps_per_env': 3, 'multi_gpu': None,
    }
    return PPO.construct_algorithm(obs, cast(VecEnv, SimpleNamespace(num_actions=6, num_envs=obs.batch_size[0])), cfg, 'cpu')


def test_full_factory_beta_update_and_save_load():
    obs = observations(raw())
    alg = factory(obs)
    for _ in range(3):
        with torch.inference_mode():
            assert alg.act(obs).shape == (32, 6)
            reward = torch.randn(32)
            alg.process_env_step(obs, reward, torch.zeros(32), {})
            assert torch.allclose(alg.storage.rewards[alg.storage.step - 1, :, 0], reward)
    with torch.inference_mode():
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    critic = cast(TaskNormalizedCritics, alg.critic)
    assert all(torch.equal(normalizer(cast(MLPModel, child)).count, torch.tensor([24] * 4)) for child in critic.critics)
    restored = factory(obs)
    serialized = io.BytesIO()
    torch.save(alg.save(), serialized)
    serialized.seek(0)
    assert restored.load(torch.load(serialized, weights_only=False), None, True)
    assert torch.equal(alg.actor(obs), restored.actor(obs))
    assert torch.equal(alg.critic(obs), restored.critic(obs))


@pytest.mark.parametrize('cls', [TaskNormalizedMLP, RoutedTaskNormalizedMLP])
def test_unsupported_actor_configuration_rejected(cls):
    obs = observations(raw())
    with pytest.raises(ValueError, match='obs_normalization'):
        cls(obs, {'actor': ['actor']}, 'actor', 6, obs_normalization=False)


def test_unsupported_feature_dimension_rejected():
    with pytest.raises(ValueError, match='56 raw features'):
        EmpiricalTaskNormalization(59)
