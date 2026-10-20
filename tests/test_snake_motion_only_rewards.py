"""全向速度跟踪与蛇头倾斜契约；绕重力轴转动不受姿态惩罚。"""
import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def historical_tracking_env():
    """Explicit old recipe for retained tracking helper compatibility, not today's task."""
    env = state_env()
    env.cfg.observation_reference = 'command'
    env.cfg.rewards = {
        'translation_tracking': RewardTermCfg(func=mdp.translation_tracking, weight=5., params={'std': .05}),
        'yaw_tracking': RewardTermCfg(func=mdp.yaw_tracking, weight=5., params={'std': .3}),
        'head_tilt': RewardTermCfg(func=mdp.head_tilt_cost, weight=-5.),
    }
    return env


@pytest.mark.parametrize('play', [False, True])
def test_world_speed_pose_rewards_and_tilt_reset_are_active(play):
    cfg = snake_env_cfg(play=play)
    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        'head_forward_speed': 15.,
        'head_lateral_speed_cost': -5.,
        'head_vertical_speed_cost': -.2,
        'joint_pose_deviation': -.02, 'joint_fold_cost': -1.,
        'head_tilt_reset_cost': -5., 'target_saturation_fraction': -.1,
        'head_angular_velocity_cost': -.2,
        'head_yaw_cost': -.5,
        'head_yaw_integral_cost': -.5,
        'body_yaw_cost': -.1,
        'body_yaw_integral_cost': -1.,
        'body_yaw_reset_cost': -5.,
        'head_yaw_reset_cost': -5.,
        'alive': .02,
        'action_rate': -.0,
        'self_collision_reset_cost': -5.,
    }
    assert set(cfg.terminations) == {'unsafe', 'head_tilt', 'head_yaw', 'body_yaw', 'self_collision', 'time_out'}
    assert cfg.head_tilt_limit == pytest.approx(math.pi / 3)
    assert cfg.terminations['head_tilt'].func is mdp.head_tilt_limit_reached
    assert not cfg.terminations['head_tilt'].time_out
    assert 'head_tilt_rad' in cfg.metrics
    assert 'head_tilt' not in cfg.rewards


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('degrees', [0., 15., 30., 59., 60., 61., 90., 180.])
@pytest.mark.parametrize('yaw', [0., .7, -2.3])
def test_head_pose_cost_penalizes_tilt_and_is_invariant_to_world_yaw(axis, degrees, yaw):
    env = historical_tracking_env()
    data = env.scene['robot'].data
    angle = math.radians(degrees)
    tilt = torch.zeros(4)
    tilt[0] = math.cos(angle / 2)
    tilt[axis + 1] = math.sin(angle / 2)
    turn = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    rotation = quat_mul(turn, tilt).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(rotation, data.body_link_quat_w[:, 0])
    expected = torch.full((2,), min(.5, 1 - math.cos(angle)))
    torch.testing.assert_close(mdp.head_tilt_cost(env), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize('play', [False, True])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('degrees', [-90., -60.001, -60., -59.999, -30., 0., 30., 59.999, 60., 60.001, 90.])
@pytest.mark.parametrize('yaw', [0., 1.2])
def test_head_tilt_reset_boundary_ignores_yaw_and_preserves_other_environment(play, axis, degrees, yaw):
    from mjlab.managers.termination_manager import TerminationManager

    env = historical_tracking_env()
    env.cfg = snake_env_cfg(play=play)
    data = env.scene['robot'].data
    angle = math.radians(degrees)
    tilt = torch.zeros(4)
    tilt[0] = math.cos(angle / 2)
    tilt[axis + 1] = math.sin(angle / 2)
    turn = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    data.body_link_quat_w[0, 0] = quat_mul(quat_mul(turn, tilt), data.body_link_quat_w[0, 0])
    # 手动控制只豁免超时，不应豁免头倾复位。
    mdp.command_term(env).manual_control = torch.ones(2, dtype=torch.bool)
    manager = TerminationManager({'head_tilt': env.cfg.terminations['head_tilt']}, env)
    expected = torch.tensor([abs(degrees) >= 60., False])
    torch.testing.assert_close(manager.compute(), expected)
    torch.testing.assert_close(manager.terminated, expected)
    assert not manager.time_outs.any()
    assert manager.reset()['Episode_Termination/head_tilt'] == int(expected[0])


def test_reward_manager_applies_negative_head_pose_reward_at_rest():
    from mjlab.managers.reward_manager import RewardManager

    env = historical_tracking_env()
    data = env.scene['robot'].data
    tilt = torch.tensor([math.cos(math.pi / 6), math.sin(math.pi / 6), 0., 0.])
    data.body_link_quat_w[:, 0] = quat_mul(tilt.expand(2, -1), data.body_link_quat_w[:, 0])
    manager = RewardManager(env.cfg.rewards, env)
    # 零指令且静止时，两项速度跟踪各得5，再减去0.5*5的倾斜惩罚。
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((2,), 7.5 * env.step_dt))


def test_stationary_robot_has_lower_reward_than_matching_nonzero_command(monkeypatch):
    env = historical_tracking_env()
    mdp.command_term(env).command[:] = torch.tensor([.1, 0., 0.])
    monkeypatch.setattr(mdp, 'planar_velocity', lambda _: torch.zeros(2, 3))
    torch.testing.assert_close(mdp.translation_tracking(env), torch.full((2,), math.exp(-.1**2 / .05**2)))
    torch.testing.assert_close(mdp.yaw_tracking(env), torch.ones(2))
    monkeypatch.setattr(mdp, 'planar_velocity', lambda _: torch.tensor([[.1, 0., 0.]]).expand(2, -1))
    torch.testing.assert_close(mdp.translation_tracking(env), torch.ones(2))


@pytest.mark.parametrize('yaw_speed', [-.6, -.3, 0., .3, .6])
def test_zero_yaw_command_tracks_stationary_yaw_independently_of_translation(monkeypatch, yaw_speed):
    env = historical_tracking_env()
    mdp.command_term(env).command[:] = torch.tensor([.1, 0., 0.])
    monkeypatch.setattr(mdp, 'planar_velocity', lambda _: torch.tensor([[.08, 0., yaw_speed]]).expand(2, -1))
    monkeypatch.setattr(mdp, 'ground_support', lambda _: torch.ones(2))
    reward = sum(term.weight * term.func(env, **term.params) for term in env.cfg.rewards.values())
    expected_linear = math.exp(-.02**2 / .05**2)
    expected_yaw = math.exp(-yaw_speed**2 / .3**2)
    torch.testing.assert_close(mdp.translation_tracking(env), torch.full((2,), expected_linear))
    torch.testing.assert_close(reward, torch.full((2,), 5 * (expected_linear + expected_yaw)))


@pytest.mark.parametrize('target', [-.3, .3])
@pytest.mark.parametrize('actual_ratio', [1., 0., -1., 2.])
def test_reward_manager_tracks_both_yaw_directions(monkeypatch, target, actual_ratio):
    from mjlab.managers.reward_manager import RewardManager

    env = historical_tracking_env()
    monkeypatch.setattr(mdp, 'planar_velocity', lambda _: torch.tensor([[0., 0., target * actual_ratio]]).expand(2, -1))
    monkeypatch.setattr(mdp, 'ground_support', lambda _: torch.ones(2))
    mdp.command_term(env).command[:] = torch.tensor([0., 0., target])
    manager = RewardManager(env.cfg.rewards, env)
    expected = 5 * (1 + math.exp(-(target - target * actual_ratio)**2 / .3**2))
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((2,), expected * env.step_dt))


def test_training_samples_six_directions_mixed_commands_and_stops():
    from types import SimpleNamespace
    from typing import cast

    from mjlab.envs import ManagerBasedRlEnv
    cfg = SnakeVelocityCommandCfg(
        training=True, sample_velocity_commands=True, resampling_time_range=(5., 9.),
        translation_probability=.5, turn_probability=.2, stop_probability=.1,
        translation_min_speed=.02, translation_curriculum=(.06, .10, .15, .20, .25),
    )
    env = SimpleNamespace(num_envs=16000, device='cpu', step_dt=.02)
    term = cfg.build(cast(ManagerBasedRlEnv, env))
    torch.manual_seed(73)
    term.reset(None)
    requested = term.requested_velocity
    active = requested.ne(0)
    counts = active.sum(-1)
    assert active[:, :2].any(-1).logical_and(~active[:, 2]).float().mean().item() == pytest.approx(.5, abs=.02)
    assert active[:, 2].logical_and(~active[:, :2].any(-1)).float().mean().item() == pytest.approx(.2, abs=.02)
    assert counts.eq(3).float().mean().item() == pytest.approx(.2, abs=.02)
    assert counts.eq(0).float().mean().item() == pytest.approx(.1, abs=.02)
    for axis in (0, 1, 2):
        for sign in (-1, 1):
            assert ((counts == 1) & (requested[:, axis] * sign > 0)).sum() > 1400
    limits = requested.new_tensor(cfg.velocity_limits)
    limits[:2] = .06
    assert requested.abs().le(limits).all()
    assert (requested.max(0).values > .95 * limits).all()
    assert (requested.min(0).values < -.95 * limits).all()
    assert requested[:, :2][active[:, :2]].abs().ge(.02).all()


def test_mixed_velocity_reward_tracks_linear_and_yaw_independently(monkeypatch):
    from mjlab.managers.reward_manager import RewardManager

    env = historical_tracking_env()
    target = torch.tensor([[.1, -.05, .3]]).expand(2, -1)
    mdp.command_term(env).command.copy_(target)
    monkeypatch.setattr(mdp, 'planar_velocity', lambda _: target)
    monkeypatch.setattr(mdp, 'ground_support', lambda _: torch.ones(2))
    manager = RewardManager(env.cfg.rewards, env)
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((2,), 10 * env.step_dt))


@pytest.mark.parametrize('yaw', [0., .4, -1.8])
@pytest.mark.parametrize('linear_std,angular_std', [(.5, math.sqrt(.5)), (.05, .3)])
def test_tracking_rewards_match_upstream_mjlab_g1_in_site_motion_frame(yaw, linear_std, angular_std):
    from mjlab.tasks.velocity.mdp.rewards import (
        track_angular_velocity,
        track_linear_velocity,
    )

    env = historical_tracking_env()
    data = env.scene['robot'].data
    turn = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    data.site_quat_w[:, 0] = quat_mul(turn.expand(2, -1), data.site_quat_w[:, 0])
    generator = torch.Generator().manual_seed(911)
    data.body_com_lin_vel_w.copy_(torch.randn(2, 12, 3, generator=generator) * .3)
    data.site_ang_vel_w.copy_(torch.randn(2, 1, 3, generator=generator))
    # 各刚体质量相等，直接求均值并旋转到与site一致的水平运动坐标。
    velocity_w = data.body_com_lin_vel_w.mean(1)
    angular_w = data.site_ang_vel_w[:, 0]
    frame = torch.tensor([
        [math.cos(yaw), math.sin(yaw), 0.],
        [-math.sin(yaw), math.cos(yaw), 0.],
        [0., 0., 1.],
    ])
    data.root_link_lin_vel_b = velocity_w @ frame.T
    data.root_link_ang_vel_b = angular_w @ frame.T
    term = mdp.command_term(env)
    term.command[:] = torch.tensor([[.1, 0., -.4], [0., 0., 0.]])
    env.command_manager.get_command = lambda name: term.command
    torch.testing.assert_close(
        mdp.translation_tracking(env, std=linear_std),
        track_linear_velocity(env, std=linear_std, command_name='motion'),
    )
    torch.testing.assert_close(
        mdp.yaw_tracking(env, std=angular_std),
        track_angular_velocity(env, std=angular_std, command_name='motion'),
    )


def test_zero_command_tracks_all_axes_without_contact_gate():
    env = historical_tracking_env()
    data = env.scene['robot'].data
    torch.testing.assert_close(mdp.translation_tracking(env), torch.ones(2))
    torch.testing.assert_close(mdp.yaw_tracking(env), torch.ones(2))
    data.body_com_lin_vel_w[:] = torch.tensor([.1, .2, .3])
    data.site_ang_vel_w[:] = torch.tensor([.4, .5, .6])
    linear = math.exp(-(.1**2 + .2**2 + .3**2) / .05**2)
    angular = math.exp(-(.4**2 + .5**2 + .6**2) / .3**2)
    torch.testing.assert_close(mdp.translation_tracking(env), torch.full((2,), linear))
    torch.testing.assert_close(mdp.yaw_tracking(env), torch.full((2,), angular))
    env.scene['snake_ground_contact'].data.force.zero_()
    torch.testing.assert_close(mdp.translation_tracking(env), torch.full((2,), linear))
    torch.testing.assert_close(mdp.yaw_tracking(env), torch.full((2,), angular))


@pytest.mark.parametrize('error,expected', [(.02, .85214379), (.05, .36787944), (.1, .01831564)])
def test_slow_linear_velocity_errors_receive_distinct_rewards(error, expected):
    env = historical_tracking_env()
    mdp.command_term(env).command[:, 0] = error
    torch.testing.assert_close(mdp.translation_tracking(env), torch.full((2,), expected))


@pytest.mark.parametrize('error,expected', [(.1, .89483932), (.3, .36787944), (.6, .01831564)])
def test_slow_yaw_velocity_errors_receive_distinct_rewards(error, expected):
    env = historical_tracking_env()
    mdp.command_term(env).command[:, 2] = error
    torch.testing.assert_close(mdp.yaw_tracking(env), torch.full((2,), expected))
