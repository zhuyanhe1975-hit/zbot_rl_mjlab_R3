"""Local-frame policy contracts without changing historical model interfaces."""

import copy
import math
from typing import cast

import pytest
import torch
from tensordict import TensorDict

from zbot_rl_mjlab.task_models import (
    EmpiricalTaskNormalization,
    LocalFrameRoutedTaskNormalizedMLP,
    LocalFrameTaskEncoding,
    LocalFrameTaskNormalization,
    LocalFrameTaskNormalizedCritics,
    OneHotTaskEncoding,
)


def observations(tasks=(0., 1., 2., 3., 3.25)):
    x = torch.randn(len(tasks), 56)
    x[:, 38] = -1.
    x[:, 39] = torch.tensor(tasks)
    x[:, 46] = torch.linspace(-math.pi, math.pi, len(tasks))
    return TensorDict({'actor': x.clone(), 'critic': x.clone()}, batch_size=[len(tasks)])


def actor(obs):
    return LocalFrameRoutedTaskNormalizedMLP(
        obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
        distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)},
    )


def critics(obs):
    return LocalFrameTaskNormalizedCritics(
        obs, {'critic': ['critic']}, 'critic', 1, hidden_dims=(16,), obs_normalization=True,
    )


def first_linear(module):
    return next(child for child in module.modules() if isinstance(child, torch.nn.Linear))


def test_encoding_only_replaces_task_and_heading():
    x = observations()['actor']
    previous = OneHotTaskEncoding()(x)
    features = LocalFrameTaskEncoding()(x)
    assert features.shape == (5, 61)
    assert torch.equal(features[:, :50], previous[:, :50])
    assert torch.equal(features[:, 52:], previous[:, 51:])
    assert torch.equal(features[:, 50], x[:, 46].sin())
    assert torch.equal(features[:, 51], x[:, 46].cos())


def test_normalization_preserves_raw_heading_and_all_historical_moments():
    x = observations((0., 1., 2., 3., 3.25) * 5)['actor']
    legacy, norm = EmpiricalTaskNormalization(), LocalFrameTaskNormalization()
    for batch in (x[:10], x[10:]):
        legacy.update(batch)
        norm.update(batch)
    assert all(torch.equal(value, norm.state_dict()[key]) for key, value in legacy.state_dict().items())
    normalized = norm(x)
    assert torch.equal(normalized[:, 38:40], x[:, 38:40])
    assert torch.equal(normalized[:, 46], x[:, 46])
    assert torch.equal(normalized[:, :46], legacy(x)[:, :46])
    assert torch.equal(normalized[:, 47:], legacy(x)[:, 47:])
    before = copy.deepcopy(norm.state_dict())
    norm.eval()
    norm.update(x)
    assert all(torch.equal(value, norm.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize('factory', [actor, critics])
def test_actual_linear_input_preserves_angle_before_circular_encoding(factory):
    obs = observations()
    model = factory(obs)
    model.update_normalization(obs)
    heads = model.mlp.heads if factory is actor else model.critics
    norms = [model.obs_normalizer] if factory is actor else [head.obs_normalizer for head in heads]
    # Deliberately extreme historical angle moments must not affect sin/cos.
    for norm in norms:
        norm._mean[:, 46] = 50.
        norm._std[:, 46] = .001
    captured = []
    handles = [first_linear(head).register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    ) for head in heads]
    try:
        model(obs)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == 4
    for features in captured:
        assert features.shape == (5, 61)
        assert torch.equal(features[:, 39:43], torch.eye(4)[[0, 1, 2, 3, 3]])
        assert torch.equal(features[:, 43], torch.tensor([0., 0., 0., 0., .25]))
        assert torch.equal(features[:, 50], obs['actor'][:, 46].sin())
        assert torch.equal(features[:, 51], obs['actor'][:, 46].cos())
    for norm in norms:
        assert norm._mean.shape == (4, 56)
        assert torch.equal(norm.count, torch.tensor([1, 1, 1, 2]))


@pytest.mark.parametrize('factory', [actor, critics])
def test_wrap_boundary_is_continuous_at_actual_linear_input(factory):
    obs = observations((0., 0.))
    obs['actor'][1] = obs['actor'][0]
    obs['actor'][:, 46] = torch.tensor([-math.pi + 1e-5, math.pi - 1e-5])
    obs['critic'] = obs['actor'].clone()
    model = factory(obs)
    model.update_normalization(obs)
    head = model.mlp.heads[0] if factory is actor else model.critics[0]
    captured = []
    handle = first_linear(head).register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    try:
        model(obs)
    finally:
        handle.remove()
    assert torch.allclose(captured[0][0], captured[0][1], atol=3e-5, rtol=0)


@pytest.mark.parametrize('factory', [actor, critics])
def test_model_serialization_has_raw56_interface(factory):
    obs = observations()
    model = factory(obs)
    model.update_normalization(obs)
    restored = factory(obs)
    restored.load_state_dict(model.state_dict())
    assert model.obs_dim == 56
    assert torch.equal(restored(obs), model(obs))


def test_all_tasks_and_recovery_route_to_independent_actor_heads():
    obs = observations()
    model = actor(obs)
    for task, head in enumerate(model.mlp.heads):
        for parameter in head.parameters():
            parameter.data.zero_()
        last = [layer for layer in head.modules() if isinstance(layer, torch.nn.Linear)][-1]
        last.bias.data.fill_(task + 1.)
    output = model.mlp(model.obs_normalizer(obs['actor']))
    for row, task in enumerate((0, 1, 2, 3, 3)):
        assert torch.equal(output[row], torch.full((2, 6), task + 1.))
    first = first_linear(model.mlp.heads[0]).weight
    assert all(first.data_ptr() != first_linear(head).weight.data_ptr() for head in model.mlp.heads[1:])


def test_critics_keep_per_task_values_and_normalization_isolated():
    obs = observations()
    model = critics(obs)
    model.update_normalization(obs)
    values = model(obs)
    mask = torch.nn.functional.one_hot(obs['actor'][:, 39].round().long(), 4)
    assert values.shape == (5, 4)
    assert torch.equal(values[mask == 0], torch.zeros_like(values[mask == 0]))
    changed = observations()
    changed['critic'][2, :38] = 1e6
    norms = [cast(LocalFrameTaskNormalization, head.obs_normalizer) for head in model.critics]
    before = [copy.deepcopy(norm.state_dict()) for norm in norms]
    # Only task 2 is observed in this update; all other moment tables stay fixed.
    model.update_normalization(changed[2:3])
    for old, norm in zip(before, norms, strict=True):
        for key, value in old.items():
            assert torch.equal(value[[0, 1, 3]], norm.state_dict()[key][[0, 1, 3]])


def test_jit_export_encodes_raw56_and_has_no_noise_or_history(tmp_path):
    obs = observations()
    model = actor(obs)
    model.update_normalization(obs)
    model.eval()
    scripted = torch.jit.script(model.as_jit())
    path = tmp_path / 'local_frame_actor.pt'
    scripted.save(str(path))
    deployed = torch.jit.load(str(path))
    expected = model(obs)
    for _ in range(3):
        assert torch.equal(deployed(obs['actor']), expected)
    assert first_linear(model.mlp.heads[0]).in_features == 61
    before = copy.deepcopy(model.state_dict())
    model(observations())
    assert torch.equal(model(obs), expected)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
