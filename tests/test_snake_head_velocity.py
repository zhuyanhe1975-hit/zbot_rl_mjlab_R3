"""教师能直接观察蛇头速度，并以旋转而非前进速度计摆动成本。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp


def test_full_head_velocity_is_appended_without_replacing_com_or_gyro():
    env = state_env()
    data = env.scene['robot'].data
    linear = torch.tensor([[.02, -.03, .04], [-.01, .05, -.02]])
    angular = torch.tensor([[.2, -.3, .4], [-.5, .6, -.7]])
    quat = data.body_link_quat_w[:, 0]
    data.body_link_lin_vel_w[:, 0] = quat_apply(quat, linear)
    data.body_link_ang_vel_w[:, 0] = quat_apply(quat, angular)
    obs = mdp.actor_state(env)
    assert obs.shape == (2, mdp.ACTOR_DIM)
    assert mdp.critic_state(env).shape == (2, mdp.CRITIC_DIM)
    torch.testing.assert_close(obs[:, 40:43], linear)
    torch.testing.assert_close(obs[:, 33:36], angular)
    torch.testing.assert_close(obs[:, 27:29], torch.zeros(2, 2))


@pytest.mark.parametrize('axis', range(3))
def test_angular_penalty_is_symmetric_quadratic_on_every_axis(axis):
    env = state_env()
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = .2
    env.scene['robot'].data.body_link_ang_vel_w[:, 0, axis] = torch.tensor([.2, -.4])
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), torch.tensor([.04, .16]))
    torch.testing.assert_close(mdp.head_angular_speed(env), torch.tensor([.2, .4]))
    # 此教师不跟踪旋转指令：不能用指令抵消摇头成本。
    mdp.command_term(env).command[:, 2] = .2
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), torch.tensor([.04, .16]))


def test_straight_translation_is_free_of_angular_penalty_and_reward_is_applied():
    env = state_env()
    data = env.scene['robot'].data
    data.body_link_lin_vel_w[:] = torch.tensor([.05, 0., 0.])
    data.body_com_lin_vel_w[:] = torch.tensor([.05, 0., 0.])
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), torch.zeros(2))
    data.body_link_ang_vel_w[:, 0, 2] = 1.
    term = env.cfg.rewards['head_angular_velocity_cost']
    assert term.weight == -.2
    manager = RewardManager({'head_angular_velocity_cost': term}, env)
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((2,), -.2 * .5 * env.step_dt))


def test_head_observation_and_angular_cost_ignore_global_yaw():
    env = state_env()
    data = env.scene['robot'].data
    data.body_link_lin_vel_w[:, 0] = torch.tensor([.03, -.02, .01])
    data.body_link_ang_vel_w[:, 0] = torch.tensor([.2, -.3, .4])
    expected_linear = mdp.head_linear_velocity(env).clone()
    expected_cost = mdp.head_angular_velocity_cost(env).clone()
    yaw = torch.tensor([math.cos(.7), 0., 0., math.sin(.7)]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(yaw, data.body_link_quat_w[:, 0])
    for name in ('body_link_lin_vel_w', 'body_link_ang_vel_w'):
        value = getattr(data, name)
        value[:, 0] = quat_apply(yaw, value[:, 0])
    torch.testing.assert_close(mdp.head_linear_velocity(env), expected_linear)
    torch.testing.assert_close(mdp.head_angular_velocity_cost(env), expected_cost)
