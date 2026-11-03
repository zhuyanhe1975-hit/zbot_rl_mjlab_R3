"""Sampled heading, head-site velocity, and the observation/reward contract."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.snake import rz270_heading as task


def test_forward_reward_uses_site_velocity_not_com_or_world_x(monkeypatch):
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(
        site_lin_vel_w=torch.tensor([[[0., .1, 9.]], [[0., -.1, 0.]], [[0., .5, 0.]]]),
        body_com_lin_vel_w=torch.full((3, 1, 3), 99.),
    ))})
    monkeypatch.setattr(task.mdp, 'command_term', lambda env: SimpleNamespace(requested_velocity=torch.tensor([[.1, 0., 0.]]).expand(3, -1)))
    monkeypatch.setattr(task.mdp, 'heading', lambda env: torch.tensor([[0., 1.]]).expand(3, -1))
    monkeypatch.setattr(task, 'motion_site_index', lambda env: 0)
    torch.testing.assert_close(task.head_forward_velocity(env), torch.tensor([.1, -.1, .5]))
    torch.testing.assert_close(task.head_forward_speed(env), torch.tensor([1., -1., -1.]))


def test_heading_error_wraps_at_pi_and_is_visible_to_policy(monkeypatch):
    term = SimpleNamespace(target_heading=torch.tensor([math.radians(-179)]))
    current = math.radians(179)
    monkeypatch.setattr(task.mdp, 'command_term', lambda env: term)
    monkeypatch.setattr(task.mdp, 'heading', lambda env: torch.tensor([[math.cos(current), math.sin(current)]]))
    monkeypatch.setattr(task, 'site_snake_state', lambda env: torch.zeros(1, 61))
    error = task.heading_error(None)
    torch.testing.assert_close(error, torch.tensor([math.radians(2)]), atol=1e-6, rtol=0)
    observation = task.heading_observation(None)
    assert observation.shape == (1, 63)
    torch.testing.assert_close(observation[:, -2], error.sin())
    torch.testing.assert_close(observation[:, -1], error.cos())
    torch.testing.assert_close(task.heading_alignment(None), -error.abs())


def test_heading_sampling_preserves_unselected_environments_and_has_no_sideways_or_yaw():
    command = SimpleNamespace(
        cfg=SimpleNamespace(heading_range=(-.8, .9), forward_speed_cap=.2, training=True,
                            forward_speed_range=(0., .2), forward_stop_probability=.1),
        device='cpu', manual_control=torch.zeros(1000, dtype=torch.bool),
        target_heading=torch.full((1000,), 99.), requested_velocity=torch.full((1000, 3), 99.),
    )
    ids = torch.arange(1, 1000)
    task.HeadingSnakeCommand._sample_velocity(command, ids)
    assert command.target_heading[0] == 99.
    assert command.requested_velocity[0].eq(99.).all()
    assert command.target_heading[ids].min() >= -.8
    assert command.target_heading[ids].max() <= .9
    assert command.target_heading[ids].min() < -.7
    assert command.target_heading[ids].max() > .8
    speeds = command.requested_velocity[ids, 0]
    assert speeds.min() >= 0. and speeds.max() <= .2
    assert speeds.max() > .19
    assert .04 < float((speeds == 0.).float().mean()) < .16
    assert command.requested_velocity[ids, 1:].eq(0.).all()


@pytest.mark.parametrize('heading_range', [(1., -1.), (-4., 1.), (0., float('nan'))])
def test_invalid_heading_sampling_ranges_are_rejected(heading_range):
    with pytest.raises(ValueError):
        task.HeadingSnakeCommandCfg(heading_range=heading_range)


def test_rz270_objectives_replace_three_axis_velocity_tracking():
    cfg = load_env_cfg('Mjlab-Zbot-6dof-Snake-RZ270-Teacher')
    assert cfg.rewards['head_forward_speed'].func is task.head_forward_speed
    assert cfg.rewards['heading_alignment'].func is task.heading_alignment
    assert not {'track_forward_velocity', 'track_lateral_velocity', 'track_angular_velocity'} & cfg.rewards.keys()
    assert cfg.observations['actor'].terms['privileged'].func is task.heading_observation
    assert isinstance(cfg.commands['transform'], task.HeadingSnakeCommandCfg)


@pytest.mark.parametrize('yaw', [0., math.pi / 2, math.pi, -math.pi / 2])
def test_head_lateral_penalty_is_symmetric_and_excludes_forward_and_vertical_motion(monkeypatch, yaw):
    forward = torch.tensor([math.cos(yaw), math.sin(yaw)])
    left = torch.tensor([-math.sin(yaw), math.cos(yaw)])
    lateral = torch.tensor([0., .1, -.1, .2])
    velocity = torch.zeros(4, 1, 3)
    velocity[:, 0, :2] = .5 * forward + lateral[:, None] * left
    velocity[:, 0, 2] = 3.
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(
        site_lin_vel_w=velocity, body_com_lin_vel_w=torch.full_like(velocity, 99.),
    ))})
    monkeypatch.setattr(task.mdp, 'heading', lambda env: forward.expand(4, -1))
    monkeypatch.setattr(task, 'motion_site_index', lambda env: 0)
    torch.testing.assert_close(task.head_lateral_speed(env), lateral.abs(), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(task.head_lateral_speed_penalty(env), lateral.abs(), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('yaw', [0., math.pi / 2, math.pi, -math.pi / 2])
def test_head_tilt_angle_penalty_uses_gravity_and_is_yaw_invariant(yaw):
    from mjlab.utils.lab_api.math import quat_mul

    angles = torch.tensor([0., math.pi / 6, math.pi / 3], dtype=torch.float64)
    rotation = torch.zeros(3, 4, dtype=torch.float64)
    rotation[:, 0], rotation[:, 2] = (angles / 2).cos(), (angles / 2).sin()
    canonical = rotation.new_tensor([2**-.5, 0., -(2**-.5), 0.]).expand_as(rotation)
    yaw_quat = rotation.new_tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]).expand_as(rotation)
    quat = quat_mul(yaw_quat, quat_mul(rotation, canonical))
    env = SimpleNamespace(
        cfg=SimpleNamespace(site_up_axis=(1., 0., 0.), sim=SimpleNamespace(mujoco=SimpleNamespace(gravity=(0., 0., -9.81)))),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(site_quat_w=quat[:, None]))},
        _omni_motion_site_index=0,
    )
    torch.testing.assert_close(task.head_tilt_angle_penalty(env), angles.square(), atol=1e-8, rtol=1e-5)
    env.cfg.sim.mujoco.gravity = (0., 0., -1.)
    torch.testing.assert_close(task.head_tilt_angle_penalty(env), angles.square(), atol=1e-8, rtol=1e-5)


def test_linear_speed_score_has_reverse_penalties_and_overspeed_decay():
    actual = torch.tensor([-.1, -.05, 0., .05, .1, .15, .2, .25, .3])
    requested = torch.full_like(actual, .1)
    expected = torch.tensor([-1., -.5, 0., .5, 1., .5, 0., -.5, -1.])
    torch.testing.assert_close(task.linear_speed_score(actual, requested), expected, atol=1e-6, rtol=0)


def test_zero_speed_command_rewards_stopping_without_division_by_zero():
    actual = torch.tensor([-.1, -.05, -.025, 0., .025, .05, .075, .1])
    expected = torch.tensor([-1., 0., .5, 1., .5, 0., -.5, -1.])
    score = task.linear_speed_score(actual, torch.zeros_like(actual))
    assert torch.isfinite(score).all()
    torch.testing.assert_close(score, expected, atol=1e-6, rtol=0)


def test_tiny_nonzero_command_still_has_relative_scoring():
    actual = torch.tensor([-1e-10, 0., 1e-10, 2e-10, 3e-10])
    requested = torch.full_like(actual, 1e-10)
    torch.testing.assert_close(task.linear_speed_score(actual, requested), torch.tensor([-1., 0., 1., 0., -1.]))


@pytest.mark.parametrize('settings', [
    {'forward_speed_range': (-.1, .2)}, {'forward_speed_range': (.2, .1)},
    {'forward_speed_range': (0., .3)}, {'forward_stop_probability': 1.1},
    {'forward_stop_probability': float('nan')}, {'play_forward_speed': -.1},
])
def test_invalid_speed_command_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        task.HeadingSnakeCommandCfg(**settings)


def test_heading_is_a_pure_deviation_penalty(monkeypatch):
    monkeypatch.setattr(task, 'heading_error', lambda env: torch.tensor([0., math.radians(1), -math.radians(1), math.pi / 2, -math.pi / 2, math.pi]))
    torch.testing.assert_close(task.heading_alignment(None), torch.tensor([0., -math.radians(1), -math.radians(1), -math.pi / 2, -math.pi / 2, -math.pi]))
