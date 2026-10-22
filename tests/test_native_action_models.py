"""Native Gaussian/Beta interfaces, export and real bounded PPO updates."""

import copy
import io
from typing import cast

import pytest
import torch
from rsl_rl.modules import BetaDistribution, GaussianDistribution
from test_guided_ppo import algorithm, collect, observations, teacher

from zbot_rl_mjlab.native_action_models import (
    NativeActionRoutedTaskNormalizedMLP,
    NativeTaskDistribution,
)
from zbot_rl_mjlab.task_models import TaskRoutedNetwork


def model(obs):
    return NativeActionRoutedTaskNormalizedMLP(
        obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
        distribution_cfg={'class_name': 'zbot_rl_mjlab.native_action_models:NativeTaskDistribution',
                          'init_std': 1., 'std_type': 'scalar', 'action_range': (-1., 1.)},
    )


def packed():
    output = torch.randn(24, 3, 6)
    output[:, 2] = 0
    output[:12, 2] = 1
    return output


def test_selected_statistics_and_likelihood_equal_native_distributions():
    output = packed()
    native = NativeTaskDistribution(6)
    gaussian, beta = GaussianDistribution(6), BetaDistribution(6)
    native.update(output)
    gaussian.update(output[:, 0])
    beta.update(output[:, :2])
    assert native.input_dim == [3, 6]
    assert torch.equal(native.mean[:12], gaussian.mean[:12])
    assert torch.equal(native.std[:12], gaussian.std[:12])
    assert torch.equal(native.entropy[:12], gaussian.entropy[:12])
    assert torch.equal(native.mean[12:], beta.mean[12:])
    assert torch.equal(native.std[12:], beta.std[12:])
    assert torch.equal(native.entropy[12:], beta.entropy[12:])
    actions = native.sample()
    actions[:12] = 2.5  # Gaussian likelihood must not clamp to Beta's bounds.
    assert torch.equal(native.log_prob(actions)[:12], gaussian.log_prob(actions)[:12])
    assert torch.equal(native.log_prob(actions)[12:], beta.log_prob(actions)[12:])
    assert torch.isfinite(native.log_prob(actions)).all()
    assert len(native.params) == 5
    assert native.params[-1].shape == (24, 1)


def test_sampling_keeps_gaussian_unbounded_and_beta_bounded():
    output = packed().repeat(20, 1, 1)
    distribution = NativeTaskDistribution(6)
    distribution.update(output)
    sample = distribution.sample()
    mask = output[:, 2, 0].bool()
    assert (sample[mask].abs() > 1).any()
    assert (sample[~mask].abs() < 1).all()


def test_kl_exactly_reuses_each_native_distribution():
    output = packed()
    distribution = NativeTaskDistribution(6)
    distribution.update(output)
    old = tuple(value.detach().clone() for value in distribution.params)
    distribution.gaussian.std_param.data.fill_(.7)
    distribution.update(output + torch.randn_like(output) * .1)
    new = distribution.params
    kl = distribution.kl_divergence(old, new)
    expected_gaussian = distribution.gaussian.kl_divergence(old[:2], new[:2])
    expected_beta = distribution.beta.kl_divergence(old[2:4], new[2:4])
    assert torch.equal(kl[:12], expected_gaussian[:12])
    assert torch.equal(kl[12:], expected_beta[12:])
    assert torch.isfinite(kl).all()


@pytest.mark.parametrize('tasks', [(0,) * 8, (2,) * 8, (0, 1, 2, 3) * 4])
def test_actor_native_shapes_jit_and_state_reload(tasks):
    obs = observations(tasks)
    actor = model(obs)
    actor.update_normalization(obs)
    router = cast(TaskRoutedNetwork, actor.mlp)
    packed_output = router(actor.get_latent(obs))
    assert packed_output.shape == (len(tasks), 3, 6)
    assert torch.equal(packed_output[:, 2, 0].bool(), obs['actor'][:, 39] == 0)
    gaussian_head = cast(torch.nn.Sequential, router.heads[0])
    gaussian_mlp = cast(torch.nn.Sequential, gaussian_head[1])
    assert cast(torch.nn.Linear, gaussian_mlp[0]).in_features == 61
    assert actor.native_biped_actions
    assert actor.obs_dim == 56
    actor.eval()
    expected = actor(obs)
    scripted = torch.jit.script(actor.as_jit())
    assert torch.equal(scripted(obs['actor']), expected)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    assert torch.equal(torch.jit.load(buffer)(obs['actor']), expected)
    restored = model(obs)
    restored.load_state_dict(copy.deepcopy(actor.state_dict()))
    assert torch.equal(restored(obs), expected)


def test_beta_only_objective_has_no_gaussian_std_gradient():
    obs = observations((2,) * 8)
    actor = model(obs)
    actions = actor(obs, stochastic_output=True)
    loss = -actor.get_output_log_prob(actions).mean() - actor.output_entropy.mean()
    loss.backward()
    distribution = cast(NativeTaskDistribution, actor.distribution)
    assert distribution.gaussian.std_param.grad is not None
    assert torch.equal(distribution.gaussian.std_param.grad, torch.zeros(6))
    for task, head in enumerate(cast(TaskRoutedNetwork, actor.mlp).heads):
        if task == 2:
            assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.parameters())
        else:
            assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0 for parameter in head.parameters())


def test_biped_deterministic_mean_and_guidance_labels_are_not_clipped():
    obs = observations((0, 1, 2, 3) * 8)
    alg = algorithm(obs, actor_cls=NativeActionRoutedTaskNormalizedMLP,
                    guidance_tasks=(0, 1, 2, 3), guidance_updates=1)
    gaussian_head = cast(torch.nn.Sequential, cast(TaskRoutedNetwork, alg.actor.mlp).heads[0])
    last_layer = cast(torch.nn.Linear, cast(torch.nn.Sequential, gaussian_head[1])[-1])
    with torch.no_grad():
        last_layer.weight.zero_()
        last_layer.bias.fill_(2.5)
    assert torch.equal(alg.actor(obs)[::4], torch.full((8, 6), 2.5))
    alg.teachers = teacher(obs, action=2.5)
    collect(alg, obs)
    assert alg._guidance_samples[0]
    assert all(torch.equal(actions, torch.full_like(actions, 2.5))
               for _, actions, _ in alg._guidance_samples[0])
    metrics = alg.update()
    assert torch.isfinite(torch.tensor(metrics['guidance']))


def test_real_guided_ppo_rollout_update_and_storage_for_native_actions():
    obs = observations((0, 1, 2, 3) * 8)
    alg = algorithm(obs, actor_cls=NativeActionRoutedTaskNormalizedMLP,
                    guidance_tasks=(0, 1, 2, 3), guidance_updates=1)
    alg.teachers = teacher(obs, action=.6)
    collect(alg, obs)
    assert len(alg.storage.distribution_params) == 5
    assert (alg.storage.actions[:, ::4].abs() > 1).any()
    assert (alg.storage.actions[:, 1::4].abs() < 1).all()
    metrics = alg.update()
    assert {'value', 'surrogate', 'guidance'} <= metrics.keys()
    assert all(torch.isfinite(parameter).all() for parameter in alg.actor.parameters())


def test_beta_guidance_preserves_inactive_gaussian_adam_state():
    obs = observations((0, 1, 2, 3) * 8)
    alg = algorithm(obs, actor_cls=NativeActionRoutedTaskNormalizedMLP,
                    guidance_tasks=(0, 1, 2, 3))
    distribution = cast(NativeTaskDistribution, alg.actor.distribution)
    # Seed Adam momentum for all heads and the Gaussian exploration parameter.
    alg.actor(obs, stochastic_output=True)
    (-alg.actor.output_entropy.mean() + alg.actor.output_mean.square().mean()).backward()
    alg.optimizer.step()
    before = copy.deepcopy(alg.actor.state_dict())
    std_state = copy.deepcopy(alg.optimizer.state[distribution.gaussian.std_param])
    alg.optimizer.zero_grad(set_to_none=True)
    beta_obs = observations((2,) * 8)
    actions = alg.actor(beta_obs, stochastic_output=True)
    (-alg.actor.get_output_log_prob(actions).mean() + alg.actor.output_mean.square().mean()).backward()
    alg._clear_inactive_actor_gradients({2})
    alg.optimizer.step()
    for name, value in alg.actor.state_dict().items():
        if name.startswith(('mlp.heads.0.', 'mlp.heads.1.', 'mlp.heads.3.', 'distribution.gaussian.')):
            assert torch.equal(value, before[name]), name
    assert distribution.gaussian.std_param.grad is None
    assert all(torch.equal(value, alg.optimizer.state[distribution.gaussian.std_param][key])
               for key, value in std_state.items())
