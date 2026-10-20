"""推进越快越抑制侧滑和摇头，同时保留真实物理速度指标。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp


def moving_env(forward=(.1, .2)):
    env = state_env()
    # 显式覆盖下限，隔离倍率数学测试与实验默认调参。
    env.cfg.motion_penalty_min_scale = 0.
    data = env.scene['robot'].data
    data.body_com_lin_vel_w[..., 0] = torch.tensor(forward)[:, None]
    data.body_com_lin_vel_w[..., 1] = .04
    data.body_com_lin_vel_w[..., 2] = .03
    data.body_link_ang_vel_w[:, 0] = torch.tensor([.2, -.3, .4])
    return env


def test_penalty_scale_grows_per_environment_with_forward_velocity():
    env = moving_env()
    assert env.cfg.motion_penalty_speed_gain == 5.
    assert env.cfg.motion_penalty_min_scale == 0.
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.tensor([.5, 1.]))
    torch.testing.assert_close(mdp.head_lateral_speed_cost(env), torch.tensor([.02, .04]))
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), .29 * torch.tensor([.5, 1.]))


def test_reverse_and_stationary_keep_minimum_costs():
    env = moving_env((-.2, 0.))
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.zeros(2))
    torch.testing.assert_close(mdp.head_lateral_speed_cost(env), torch.zeros(2))
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), torch.zeros(2))


def test_penalty_scale_saturates_at_target_forward_speed():
    env = moving_env((.2, .4))
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), .2))
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.ones(2))


def test_scale_uses_whole_body_translation_instead_of_local_head_motion():
    env = moving_env((0., 0.))
    env.scene['robot'].data.body_link_lin_vel_w[:, 0, 0] = 10.
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.zeros(2))


def test_rotating_world_velocities_changes_costs_using_world_xy():
    env = moving_env()
    data = env.scene['robot'].data
    yaw = torch.tensor([math.cos(.4), 0., 0., math.sin(.4)]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(yaw, data.body_link_quat_w[:, 0])
    data.body_com_lin_vel_w[:] = quat_apply(yaw[:, None].expand(-1, 12, -1), data.body_com_lin_vel_w)
    data.body_link_ang_vel_w[:, 0] = quat_apply(yaw, data.body_link_ang_vel_w[:, 0])
    vx = torch.tensor([.1, .2]) * math.cos(.8) - .04 * math.sin(.8)
    vy = torch.tensor([.1, .2]) * math.sin(.8) + .04 * math.cos(.8)
    expected_scale = (5. * vx.clamp_min(0.)).clamp(max=1.)
    expected_lateral = vy.abs() * expected_scale
    expected_angular = .29 * expected_scale
    torch.testing.assert_close(mdp.motion_penalty_scale(env), expected_scale)
    torch.testing.assert_close(mdp.head_lateral_speed_cost(env), expected_lateral)
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), expected_angular)


def test_physical_angular_speed_metric_does_not_include_penalty_multiplier():
    env = moving_env()
    metric = env.cfg.metrics['head_angular_speed_rad_s']
    torch.testing.assert_close(metric.func(env), torch.full((2,), math.sqrt(.29)))
    torch.testing.assert_close(env.cfg.metrics['motion_penalty_scale'].func(env), torch.tensor([.5, 1.]))


def test_zero_gain_keeps_minimum_scale_at_all_speeds():
    env = moving_env()
    env.cfg.motion_penalty_speed_gain = 0.
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.zeros(2))
    torch.testing.assert_close(mdp.head_lateral_speed_cost(env), torch.zeros(2))
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), torch.zeros(2))


@pytest.mark.parametrize('gain', [-1., float('nan'), float('inf'), -float('inf')])
def test_invalid_gain_is_rejected(gain):
    env = moving_env()
    env.cfg.motion_penalty_speed_gain = gain
    with pytest.raises(ValueError):
        mdp.motion_penalty_scale(env)


def test_speed_scaling_leaves_base_weights_and_unrelated_costs_unchanged():
    env = moving_env()
    weights = {name: term.weight for name, term in env.cfg.rewards.items()}
    before_vertical = mdp.head_vertical_speed_cost(env).clone()
    before_pose = mdp.joint_pose_deviation_cost(env).clone()
    for _ in range(2):
        mdp.head_lateral_speed_cost(env)
        mdp.head_angular_velocity_cost(env)
    assert {name: term.weight for name, term in env.cfg.rewards.items()} == weights
    torch.testing.assert_close(mdp.head_vertical_speed_cost(env), before_vertical)
    torch.testing.assert_close(mdp.joint_pose_deviation_cost(env), before_pose)
    assert mdp.actor_state(env).shape == (2, 51)
    assert mdp.critic_state(env).shape == (2, 66)


def test_reward_manager_applies_base_weights_and_timestep_to_scaled_costs():
    env = moving_env()
    terms = {name: env.cfg.rewards[name] for name in ('head_lateral_speed_cost', 'head_angular_velocity_cost')}
    manager = RewardManager(terms, env)
    weighted_cost = terms['head_lateral_speed_cost'].weight * .04 + terms['head_angular_velocity_cost'].weight * .29
    expected = weighted_cost * torch.tensor([.5, 1.]) * env.step_dt
    torch.testing.assert_close(manager.compute(env.step_dt), expected)


def test_penalty_scale_uses_actual_speed_independently_of_reward_cap():
    env = moving_env()
    env.cfg.forward_speed_cap = .01
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), .01))
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.tensor([.5, 1.]))


def test_full_minimum_scale_restores_original_motion_costs():
    env = moving_env()
    env.cfg.motion_penalty_min_scale = 1.
    torch.testing.assert_close(mdp.motion_penalty_scale(env), torch.ones(2))
    torch.testing.assert_close(mdp.head_lateral_speed_cost(env), torch.full((2,), .04))
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), torch.full((2,), .29))


@pytest.mark.parametrize('minimum', [-.1, 1.1, float('nan'), float('inf'), -float('inf')])
def test_invalid_minimum_scale_is_rejected(minimum):
    env = moving_env()
    env.cfg.motion_penalty_min_scale = minimum
    with pytest.raises(ValueError):
        mdp.motion_penalty_scale(env)


def test_rotating_only_head_does_not_change_world_speed_costs():
    env = moving_env()
    expected = [func(env).clone() for func in (mdp.head_forward_speed, mdp.head_lateral_speed_cost,
                                              mdp.motion_penalty_scale, mdp.head_angular_velocity_cost)]
    data = env.scene['robot'].data
    yaw = torch.tensor([0., 0., 0., 1.]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(yaw, data.body_link_quat_w[:, 0])
    for func, value in zip((mdp.head_forward_speed, mdp.head_lateral_speed_cost,
                            mdp.motion_penalty_scale, mdp.head_angular_velocity_cost), expected, strict=True):
        torch.testing.assert_close(func(env), value)
