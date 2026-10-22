"""Intrinsic Gaussian control axes, mirror exploration, exports and real PPO."""

import copy
import io
from typing import cast

import pytest
import torch
from rsl_rl.modules import GaussianDistribution
from test_guided_ppo import algorithm, collect, observations, teacher

from zbot_rl_mjlab.gaussian_action_models import (
    GaussianActionRoutedTaskNormalizedMLP,
    GaussianTaskDistribution,
    expand_mirror,
)
from zbot_rl_mjlab.task_models import TaskRoutedNetwork


def packed():
    output = torch.randn(48, 2, 6)
    output[:, 1] = torch.tensor([0., 1., 2., 3.] * 12).unsqueeze(-1)
    for task in (1, 3):
        output[task::4, 0] = expand_mirror(output[task::4, 0, :3])
    return output


def model(obs):
    return GaussianActionRoutedTaskNormalizedMLP(
        obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
    )


def test_native_gaussian_statistics_likelihood_entropy_and_parameter_storage():
    output = packed()
    dist = GaussianTaskDistribution(6)
    dist.update(output)
    sample = dist.sample()
    for task, width in enumerate((6, 3, 6, 3)):
        mask = output[:, 1, 0] == task
        native = GaussianDistribution(width, init_std=1. if task == 0 else .5)
        native.update(output[:, 0, :width])
        expected_mean = expand_mirror(native.mean) if width == 3 else native.mean
        expected_std = torch.cat((native.std, native.std.flip(-1)), -1) if width == 3 else native.std
        assert torch.equal(dist.mean[mask], expected_mean[mask])
        assert torch.equal(dist.std[mask], expected_std[mask])
        assert torch.equal(dist.entropy[mask], native.entropy[mask])
        assert torch.equal(dist.log_prob(sample)[mask], native.log_prob(sample[:, :width])[mask])
        if width == 3:
            assert torch.equal(sample[mask], expand_mirror(sample[mask, :3]))
    assert dist.input_dim == [2, 6]
    assert len(dist.params) == 3
    assert torch.equal(dist.params[-1], output[:, 1, :1])
    sample[1::2, 5] += .01
    assert torch.isneginf(dist.log_prob(sample)[1::2]).all()


def test_all_actions_are_unbounded_mirror_variance_covariance_and_snake_independence():
    torch.manual_seed(81)
    output = torch.zeros(30000, 2, 6)
    dist = GaussianTaskDistribution(6)
    for task in range(4):
        output[:, 1] = task
        dist.update(output)
        sample = dist.sample()
        assert (sample.abs() > 1).any()
        assert torch.allclose(sample.var(0), dist.std[0].square(), atol=.015, rtol=.025)
        covariance = torch.cov(sample.T)
        if task in (1, 3):
            assert torch.equal(sample, expand_mirror(sample[:, :3]))
            assert torch.allclose(covariance[0, 5], -covariance[0, 0])
        else:
            assert not torch.equal(sample, expand_mirror(sample[:, :3]))
            assert covariance[0, 5].abs() < .035


def test_kl_uses_intrinsic_dimensions_and_all_gradients_are_finite():
    output = packed().requires_grad_()
    dist = GaussianTaskDistribution(6)
    dist.update(output)
    old = tuple(value.detach().clone() for value in dist.params)
    changed = torch.cat((output[:, :1] + .2, output[:, 1:]), 1)
    with torch.no_grad():
        for gaussian in dist.gaussians:
            gaussian.std_param.fill_(.7)
    dist.update(changed)
    new = dist.params
    kl = dist.kl_divergence(old, new)
    for task, width in enumerate((6, 3, 6, 3)):
        mask = output[:, 1, 0] == task
        expected = dist.gaussians[task].kl_divergence(
            tuple(x[:, :width] for x in old[:2]), tuple(x[:, :width] for x in new[:2]),
        )
        assert torch.equal(kl[mask], expected[mask])
    actions = dist.sample()
    (-dist.log_prob(actions).mean() - .01 * dist.entropy.mean() + kl.mean()).backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    for gaussian in dist.gaussians:
        assert gaussian.std_param.grad is not None
        assert torch.isfinite(gaussian.std_param.grad).all()


@pytest.mark.parametrize('tasks', [(0,) * 8, (1,) * 8, (2,) * 8, (3,) * 8, (0, 1, 2, 3) * 4])
def test_four_head_dimensions_jit_and_serialization(tasks):
    obs = observations(tasks)
    actor = model(obs)
    actor.update_normalization(obs)
    router = cast(TaskRoutedNetwork, actor.mlp)
    for task, width in enumerate((6, 3, 6, 3)):
        head = cast(torch.nn.Sequential, router.heads[task])
        mlp = cast(torch.nn.Sequential, head[1])
        assert cast(torch.nn.Linear, mlp[-1]).out_features == width
    output = router(actor.get_latent(obs))
    assert output.shape == (len(tasks), 2, 6)
    assert torch.equal(output[:, 1, 0], torch.tensor(tasks).float())
    assert actor.native_biped_actions and actor.mirrored_transform_actions and actor.all_gaussian_actions
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


def test_actual_guided_ppo_ratio_one_mirror_rollout_and_update():
    obs = observations((0, 1, 2, 3) * 8)
    alg = algorithm(obs, actor_cls=GaussianActionRoutedTaskNormalizedMLP,
                    guidance_tasks=(0, 1, 2, 3), guidance_updates=1)
    alg.teachers = teacher(obs, action=.6)
    collect(alg, obs)
    actions = alg.storage.actions
    for task in (1, 3):
        assert torch.equal(actions[:, task::4], expand_mirror(actions[:, task::4, :3]))
    assert len(alg.storage.distribution_params) == 3
    alg.actor(obs, stochastic_output=True)
    actual = alg.actor.get_output_log_prob(actions[0])
    stored = alg.storage.actions_log_prob[0].reshape(-1)
    assert torch.equal((actual - stored).exp(), torch.ones_like(actual))
    metrics = alg.update()
    assert {'value', 'surrogate', 'guidance'} <= metrics.keys()
    assert all(torch.isfinite(parameter).all() for parameter in alg.actor.parameters())


def test_transform_only_step_preserves_inactive_head_and_distribution_adam_state():
    obs = observations((0, 1, 2, 3) * 8)
    alg = algorithm(obs, actor_cls=GaussianActionRoutedTaskNormalizedMLP)
    dist = cast(GaussianTaskDistribution, alg.actor.distribution)
    alg.actor(obs, stochastic_output=True)
    (alg.actor.output_mean.square().mean() - alg.actor.output_entropy.mean()).backward()
    alg.optimizer.step()
    before = copy.deepcopy(alg.actor.state_dict())
    std_states = {task: copy.deepcopy(alg.optimizer.state[dist.gaussians[task].std_param]) for task in (0, 2)}
    alg.optimizer.zero_grad(set_to_none=True)
    transform_obs = observations((1, 3) * 8)
    actions = alg.actor(transform_obs, stochastic_output=True)
    (-alg.actor.get_output_log_prob(actions).mean() - alg.actor.output_entropy.mean()).backward()
    # Actor method is also the hook used by GuidedTaskPPO to avoid Adam momentum updates.
    alg.actor.clear_inactive_distribution_gradients({1, 3})
    router = cast(TaskRoutedNetwork, alg.actor.mlp)
    for task in (0, 2):
        for parameter in router.heads[task].parameters():
            parameter.grad = None
    alg.optimizer.step()
    for name, value in alg.actor.state_dict().items():
        if name.startswith(('mlp.heads.0.', 'mlp.heads.2.', 'distribution.gaussians.0.', 'distribution.gaussians.2.')):
            assert torch.equal(value, before[name]), name
    for task, state in std_states.items():
        parameter = dist.gaussians[task].std_param
        assert parameter.grad is None
        assert all(torch.equal(value, alg.optimizer.state[parameter][key]) for key, value in state.items())
