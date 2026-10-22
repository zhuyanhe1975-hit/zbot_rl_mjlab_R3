"""Categorical model input, independent branches, PPO and export contracts."""

import copy
import io
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from zbot_rl_mjlab.task_models import (
    OneHotRoutedTaskNormalizedMLP,
    OneHotTaskEncoding,
    OneHotTaskNormalizedCritics,
)


def observations(tasks=(0., 1., 2., 3., 3.25)):
    x = torch.randn(len(tasks), 56) * .1
    x[:, 38] = -1.
    x[:, 39] = torch.tensor(tasks)
    return TensorDict({'actor': x.clone(), 'critic': x.clone()}, batch_size=[len(tasks)])


def actor(obs):
    return OneHotRoutedTaskNormalizedMLP(
        obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
        distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)},
    )


def critics(obs):
    return OneHotTaskNormalizedCritics(
        obs, {'critic': ['critic']}, 'critic', 1, hidden_dims=(16,), obs_normalization=True,
    )


def first_linear(module):
    return next(child for child in module.modules() if isinstance(child, torch.nn.Linear))


def test_encoding_preserves_physics_morphology_and_recovery():
    x = observations()['actor']
    features = OneHotTaskEncoding()(x)
    assert features.shape == (5, 60)
    assert torch.equal(features[:, :39], x[:, :39])
    assert torch.equal(features[:, 44:], x[:, 40:])
    assert torch.equal(features[:, 39:43], torch.eye(4)[[0, 1, 2, 3, 3]])
    assert torch.equal(features[:, 43], torch.tensor([0., 0., 0., 0., .25]))


@pytest.mark.parametrize('factory', [actor, critics])
def test_actual_linear_inputs_have_only_categorical_task_features(factory):
    obs = observations()
    model = factory(obs)
    model.update_normalization(obs)
    captured = []
    handles = [first_linear(head).register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    ) for head in (model.mlp.heads if factory is actor else model.critics)]
    try:
        model(obs)
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == 4
    for features in captured:
        assert features.shape == (5, 60)
        assert torch.equal(features[:, 38], obs['actor'][:, 38])
        assert torch.equal(features[:, 39:43], torch.eye(4)[[0, 1, 2, 3, 3]])
        assert torch.equal(features[:, 43], torch.tensor([0., 0., 0., 0., .25]))
    normalizers = [model.obs_normalizer] if factory is actor else [m.obs_normalizer for m in model.critics]
    for norm in normalizers:
        assert norm._mean.shape == (4, 56)
        assert torch.equal(norm.count, torch.tensor([1, 1, 1, 2]))


def test_ascent_and_recovery_use_same_independent_branch():
    obs = observations()
    model = actor(obs)
    for task, head in enumerate(model.mlp.heads):
        for parameter in head.parameters():
            parameter.data.zero_()
        last = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.bias.data.fill_(task + 1.)
    output = model.mlp(model.obs_normalizer(obs['actor']))
    for row, task in enumerate((0, 1, 2, 3, 3)):
        assert torch.equal(output[row], torch.full((2, 6), task + 1.))
    first = first_linear(model.mlp.heads[0]).weight
    assert all(first.data_ptr() != first_linear(head).weight.data_ptr() for head in model.mlp.heads[1:])


def test_training_one_task_leaves_other_actor_branches_unchanged():
    obs = observations((2.,) * 8)
    model = actor(obs)
    before = copy.deepcopy(model.mlp.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    actions = model(obs, stochastic_output=True)
    (-model.get_output_log_prob(actions.detach()).mean()).backward()
    optimizer.step()
    after = model.mlp.state_dict()
    assert all(torch.equal(value, after[key]) for key, value in before.items() if not key.startswith('heads.2.'))
    assert any(not torch.equal(value, after[key]) for key, value in before.items() if key.startswith('heads.2.'))


@pytest.mark.parametrize('factory', [actor, critics])
def test_model_state_roundtrip(factory):
    obs = observations()
    model = factory(obs)
    model.update_normalization(obs)
    restored = factory(obs)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(restored(obs), model(obs))


def test_jit_includes_encoding_and_raw_normalization(tmp_path):
    obs = observations()
    model = actor(obs)
    model.update_normalization(obs)
    model.eval()
    exported = torch.jit.script(model.as_jit())
    path = tmp_path / 'actor.pt'
    exported.save(str(path))
    deployed = torch.jit.load(str(path))
    assert torch.equal(deployed(obs['actor']), model(obs))
    assert first_linear(model.mlp.heads[0]).in_features == 60
    assert model.obs_dim == 56


def algorithm(obs, guided) -> Any:
    cfg = {
        'algorithm': {
            'class_name': 'zbot_rl_mjlab.guided_ppo:GuidedTaskPPO' if guided else 'zbot_rl_mjlab.task_ppo:TaskPPO',
            'num_learning_epochs': 1, 'num_mini_batches': 1,
        },
        'actor': {'class_name': 'zbot_rl_mjlab.task_models:OneHotRoutedTaskNormalizedMLP',
                  'hidden_dims': (16,), 'obs_normalization': True,
                  'distribution_cfg': {'class_name': 'BetaDistribution', 'action_range': (-1., 1.)}},
        'critic': {'class_name': 'zbot_rl_mjlab.task_models:OneHotTaskNormalizedCritics',
                   'hidden_dims': (16,), 'obs_normalization': True},
        'obs_groups': {'actor': ['actor'], 'critic': ['critic']},
        'num_steps_per_env': 3, 'multi_gpu': None,
    }
    if guided:
        cfg['algorithm'].update(guidance_tasks=(0, 1, 2, 3), guidance_updates=1)
    return PPO.construct_algorithm(obs, cast(VecEnv, SimpleNamespace(num_actions=6, num_envs=len(obs))), cfg, 'cpu')


@pytest.mark.parametrize('guided', [False, True])
def test_real_ppo_and_guidance_updates_with_categorical_models(guided):
    obs = observations((0., 1., 2., 3., 3.25) * 8)
    alg = algorithm(obs, guided)
    if guided:
        targets = SimpleNamespace(actions=torch.full((len(obs), 6), .6), weights=torch.ones(len(obs)), counts={})
        alg.teachers = SimpleNamespace(query=lambda: targets)
    before_actor = copy.deepcopy(alg.actor.state_dict())
    before_critic = copy.deepcopy(alg.critic.state_dict())
    for _ in range(3):
        with torch.inference_mode():
            alg.act(obs)
            alg.process_env_step(obs, torch.randn(len(obs)), torch.zeros(len(obs)), {})
    with torch.inference_mode():
        alg.compute_returns(obs)
    metrics = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(v, alg.actor.state_dict()[k]) for k, v in before_actor.items())
    assert any(not torch.equal(v, alg.critic.state_dict()[k]) for k, v in before_critic.items())
    if guided:
        assert metrics['guidance'] > 0
    restored = algorithm(obs, guided)
    restored.load(alg.save(), None, True)
    assert torch.equal(alg.actor(obs), restored.actor(obs))
    assert torch.equal(alg.critic(obs), restored.critic(obs))


def test_guidance_freezes_excluded_heads_and_existing_adam_state():
    from zbot_rl_mjlab.task_ppo import TaskPPO

    obs = observations((0., 1., 2., 3.) * 8)
    alg = algorithm(obs, True)
    for _ in range(3):
        with torch.inference_mode():
            alg.act(obs)
            alg.process_env_step(obs, torch.randn(len(obs)), torch.zeros(len(obs)), {})
    with torch.inference_mode():
        alg.compute_returns(obs)
    TaskPPO.update(alg)
    heads = alg.actor.mlp.heads
    assert all(any(alg.optimizer.state[p]['exp_avg'].abs().sum() > 0 for p in head.parameters())
               for head in heads)
    parameters = [copy.deepcopy(head.state_dict()) for head in heads]
    states = [{p: copy.deepcopy(alg.optimizer.state[p]) for p in head.parameters()} for head in heads]
    subset = observations((1.,) * 20)
    alg._cache_guidance(subset, torch.full((20, 6), .6), torch.ones(20))
    assert alg._guidance_update()['guidance'] > 0
    for task in (0, 2, 3):
        assert all(torch.equal(v, heads[task].state_dict()[k]) for k, v in parameters[task].items())
        for parameter in heads[task].parameters():
            assert parameter.grad is None
            for key, value in states[task][parameter].items():
                actual = alg.optimizer.state[parameter][key]
                assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
    assert any(not torch.equal(v, heads[1].state_dict()[k]) for k, v in parameters[1].items())
