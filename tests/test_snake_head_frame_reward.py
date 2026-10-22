"""前进奖励沿世界X，侧向成本沿世界Y，不受头尾距离门控。"""

import math

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.experiments.snake import snake_env_cfg
from zbot_rl_mjlab.tasks.snake import mdp


@pytest.mark.parametrize('yaw', [0., .4, math.pi / 2, math.pi, -math.pi / 2])
def test_head_yaw_rotates_projection_without_moving_tail(yaw):
    env = state_env()
    data = env.scene['robot'].data
    rotation = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(rotation, data.body_link_quat_w[:, 0])
    data.body_com_lin_vel_w[..., 0] = .05
    torch.testing.assert_close(mdp.head_forward_velocity(env), torch.full((2,), .05 * math.cos(yaw)), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), .05))
    torch.testing.assert_close(mdp.head_lateral_speed_cost(env), torch.zeros(2), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('tail_xy', [(0., 0.), (0., -.6), (.6, 0.), (-.001, .001)])
@pytest.mark.parametrize('speed', [-.05, .05])
def test_tail_displacement_and_coincident_endpoints_do_not_gate_forward_reward(tail_xy, speed):
    env = state_env()
    data = env.scene['robot'].data
    data.site_pos_w[:, -1, :2] = torch.tensor(tail_xy)
    data.body_com_lin_vel_w[..., 0] = speed
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), speed))


def test_head_origin_velocity_cannot_replace_com_progress():
    env = state_env()
    env.scene['robot'].data.body_link_lin_vel_w[:, 0] = 100.
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.zeros(2))
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = .05
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), .05))


@pytest.mark.parametrize('play', [False, True])
def test_task_has_no_curl_gate_parameters_rewards_or_metrics(play):
    cfg = snake_env_cfg(play=play)
    assert not hasattr(cfg, 'min_body_extension')
    assert not hasattr(cfg, 'full_body_extension')
    assert 'body_curl_cost' not in cfg.rewards
    assert 'body_extension_ratio' not in cfg.metrics
    assert 'forward_extension_gate' not in cfg.metrics
    assert cfg.rewards['action_rate'].weight == -.0


@pytest.mark.parametrize('yaw', [0., math.pi / 2, math.pi, -math.pi / 2])
@pytest.mark.parametrize('velocity', [(.3, 0., 0.), (-.4, 0., 0.), (0., .3, 0.), (0., -.3, 0.), (.1, -.2, .4), (0., 0., .3)])
def test_world_x_reward_ignores_head_yaw_and_other_velocity_axes(yaw, velocity):
    env = state_env()
    data = env.scene['robot'].data
    rotation = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(rotation, data.body_link_quat_w[:, 0])
    data.body_com_lin_vel_w[:] = torch.tensor(velocity)
    term = env.cfg.rewards['head_forward_speed']
    torch.testing.assert_close(term.func(env), torch.full((2,), min(velocity[0], .2)))
    lateral = env.cfg.rewards['head_lateral_speed_cost']
    scale = min(1., .25 + 5. * max(velocity[0], 0.))
    torch.testing.assert_close(lateral.func(env), torch.full((2,), abs(velocity[1]) * scale))
