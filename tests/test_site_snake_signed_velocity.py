"""Relative head-frame tracking: stationary=-1, half command=0, exact command=1."""

import math

import pytest
import torch
from test_site_snake_heading import fixture, set_yaw
from test_site_snake_task import reward_env

from zbot_rl_mjlab.experiments.site_snake import (
    site_snake_env_cfg as _recipe_site_snake_env_cfg,
)
from zbot_rl_mjlab.tasks.snake import site_omni as task
from zbot_rl_mjlab.tasks.universal import omni


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_scalar_tracking_penalizes_both_under_and_overspeed(monkeypatch, axis, sign):
    actual, target = torch.zeros(7, 3), torch.zeros(7, 3)
    actual[:, axis] = torch.tensor([-.1, 0., .05, .1, .15, .2, .3]) * sign
    target[:, axis] = .1 * sign
    env = reward_env(monkeypatch, actual.tolist(), target.tolist())
    reward = task.angular_tracking if axis == 2 else task.linear_tracking
    torch.testing.assert_close(reward(env), torch.tensor([-2., -1., 0., 1., 0., -1., -2.]), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('speed', [.001, .025, .05, .1])
def test_zero_command_drift_is_penalized_without_zero_division(monkeypatch, speed):
    env = reward_env(monkeypatch, [[speed, -speed, 0.], [0., 0., .3]], [[0., 0., 0.], [0., 0., .3]])
    torch.testing.assert_close(task.linear_tracking(env), torch.tensor([-2*speed/.2, 0.]))


@pytest.mark.parametrize('yaw', [0., .7, math.pi-.01, -math.pi+.01])
@pytest.mark.parametrize('target', [(.1, 0.), (-.1, 0.), (0., .1), (0., -.1), (.1, -.1)])
def test_immediate_site_velocity_reward_uses_current_head_frame(yaw, target):
    env, term = fixture((yaw,))
    set_yaw(env, (yaw+.8,))
    heading = omni.site_heading(env).clone()
    left = torch.stack((-heading[:, 1], heading[:, 0]), -1)
    term.velocity_command[:, :2] = torch.tensor(target)
    data = env.scene['robot'].data
    data.site_lin_vel_w[:, 0, :2] = target[0] * heading + target[1] * left
    expected = sum(value != 0 for value in target)
    assert task.linear_tracking(env).item() == pytest.approx(expected, abs=1e-6)
    data.site_lin_vel_w[:, 0, :2].neg_()
    assert task.linear_tracking(env).item() == pytest.approx(-2*expected, abs=1e-6)
    assert not hasattr(env, '_site_net_progress')


@pytest.mark.parametrize('forward_fraction', [.5, .9, .99])
def test_zero_net_cycles_keep_per_step_scores_bounded_despite_reverse_spikes(monkeypatch, forward_fraction):
    forward_steps = round(100*forward_fraction)
    reverse_steps = 100-forward_steps
    velocities = torch.zeros(100, 3)
    velocities[:forward_steps, 0] = .1
    velocities[forward_steps:, 0] = -.1*forward_steps/reverse_steps
    env = reward_env(monkeypatch, velocities.tolist(), [[.1, 0., 0.]]*100)
    reward = task.linear_tracking(env)
    assert abs((velocities[:, 0]*.01).sum()) < 1e-7
    assert reward[forward_steps:].max() < 0
    assert (reward >= -2.).all() and (reward <= 1.).all()
    torch.testing.assert_close(reward[forward_steps:], torch.full((reverse_steps,), -2.))


def test_lateral_errors_add_even_when_forward_speed_is_perfect(monkeypatch):
    env = reward_env(monkeypatch, [[.1, 0., 0.], [.1, .025, 0.], [.1, .1, 0.]], [[.1, 0., 0.]]*3)
    torch.testing.assert_close(task.linear_tracking(env), torch.tensor([1., .875, .5]))


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('command', [0., .001, .1, .2])
def test_relative_error_scales_by_command_and_stop_uses_axis_bound(monkeypatch, axis, command):
    target = torch.zeros(1, 3)
    target[:, axis] = command
    actual = target.clone()
    actual[:, axis] += .001
    env = reward_env(monkeypatch, actual.tolist(), target.tolist())
    reward = task.angular_tracking if axis == 2 else task.linear_tracking
    expected = 1.-2*.001/abs(command) if command else -.001/task.mdp.command_term(env).cfg.velocity_limits[axis]
    assert reward(env).item() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_large_error_is_clipped_at_negative_two(monkeypatch, axis):
    actual = torch.zeros(1, 3)
    actual[:, axis] = 2.
    env = reward_env(monkeypatch, actual.tolist(), [[0., 0., 0.]])
    reward = task.angular_tracking if axis == 2 else task.linear_tracking
    assert reward(env).item() == pytest.approx(-2.)


def test_each_axis_uses_its_own_configured_command_limit(monkeypatch):
    env = reward_env(monkeypatch, [[.1, .1, .1]], [[0., 0., 0.]])
    task.mdp.command_term(env).cfg.velocity_limits = (.1, .2, .4)
    assert task.linear_tracking(env).item() == pytest.approx(-1.5)
    assert task.angular_tracking(env).item() == pytest.approx(-.25)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_pure_rotation_cannot_collect_positive_translation_reward(monkeypatch, sign):
    env = reward_env(monkeypatch, [[0., sign*.066, sign*.2]], [[0., 0., sign*.2]])
    cfg = _recipe_site_snake_env_cfg()
    linear, angular = (cfg.rewards[name] for name in ('track_lateral_velocity', 'track_angular_velocity'))
    assert (linear.weight * linear.func(env, **linear.params)).item() == pytest.approx(linear.weight * -.33)
    assert (angular.weight * angular.func(env, **angular.params)).item() == pytest.approx(angular.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_relative_velocity_reward_has_unit_optimum_and_negative_reverse():
    env, term = fixture((0., 0.), backend='cuda_graph', device='cuda')
    term.velocity_command[:, 0] = .1
    env.scene['robot'].data.site_lin_vel_w[:, 0, 0] = torch.tensor([.1, -.1], device='cuda')
    torch.testing.assert_close(task.linear_tracking(env), torch.tensor([1., -2.], device='cuda'))


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('command', [-.2, -.1, -.001, 0., .001, .1, .2])
def test_stationary_motion_is_penalized_but_correct_stop_is_free(monkeypatch, axis, command):
    target = torch.zeros(1, 3)
    target[:, axis] = command
    env = reward_env(monkeypatch, [[0., 0., 0.]], target.tolist())
    reward = task.angular_tracking if axis == 2 else task.linear_tracking
    assert reward(env).item() == pytest.approx(-1. if command else 0.)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('command', [-.25, -.01, .01, .25])
def test_fractional_response_scores_do_not_depend_on_command_scale(monkeypatch, axis, command):
    ratios = torch.tensor([-1., 0., .25, .5, .75, 1., 1.25, 1.5, 2.])
    actual, target = torch.zeros(len(ratios), 3), torch.zeros(len(ratios), 3)
    actual[:, axis] = ratios*command
    target[:, axis] = command
    env = reward_env(monkeypatch, actual.tolist(), target.tolist())
    reward = (task.forward_tracking, task.lateral_tracking, task.angular_tracking)[axis]
    torch.testing.assert_close(reward(env), torch.tensor([-2., -1., -.5, 0., .5, 1., .5, 0., -1.]), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_tiny_moving_commands_cannot_produce_extreme_penalties(monkeypatch, axis):
    actual, target = torch.zeros(4, 3), torch.zeros(4, 3)
    target[:, axis] = torch.tensor([1e-5, -1e-5, .001, -.001])
    actual[:, axis] = .02
    env = reward_env(monkeypatch, actual.tolist(), target.tolist())
    reward = (task.forward_tracking, task.lateral_tracking, task.angular_tracking)[axis]
    torch.testing.assert_close(reward(env), torch.full((4,), -2.))
