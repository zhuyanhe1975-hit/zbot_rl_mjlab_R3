"""World COM objective, observation reference and preserved reset contracts."""

import math

import pytest
import torch
from mjlab.envs import mdp as common_mdp
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


@pytest.mark.parametrize('play', [False, True])
def test_only_requested_rewards_are_active_and_resets_unchanged(play):
    cfg = snake_env_cfg(play=play)
    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        'forward_speed': 1., 'lateral_speed_cost': -1., 'vertical_speed_cost': -1.,
        'joint_pose_deviation': -.01, 'head_tilt_reset_cost': -.5, 'target_saturation_fraction': -.1,
    }
    for name, term in cfg.rewards.items():
        assert term.func is getattr(mdp, 'joint_pose_deviation_cost' if name == 'joint_pose_deviation' else name)
    assert {name: term.func for name, term in cfg.terminations.items()} == {
        'unsafe': mdp.unsafe, 'head_tilt': mdp.head_tilt_limit_reached,
        'self_collision': mdp.self_collision,
        'time_out': mdp.play_time_out if play else common_mdp.time_out,
    }
    assert cfg.terminations['time_out'].time_out
    assert cfg.head_tilt_limit == pytest.approx(math.pi / 3)
    assert cfg.scene.entities['robot'].init_state.rot == pytest.approx((math.sqrt(.5), 0., -math.sqrt(.5), 0.))
    assert cfg.events['reset_snake'].func is mdp.reset_snake
    assert cfg.events['reset_scene'].func is common_mdp.reset_scene_to_default
    assert cfg.commands['motion'].training is (not play)
    assert not cfg.commands['motion'].sample_velocity_commands
    assert cfg.commands['motion'].translation_curriculum == ()
    assert not any(key.startswith(('group_', 'translation_', 'command_')) for key in cfg.metrics)
    assert len(cfg.metrics) <= 12


@pytest.mark.parametrize('velocity', [(0., 0., 0.), (.06, -.02, .01), (-.06, .02, -.01)])
def test_world_rewards_use_signed_x_absolute_yz_and_exact_total(velocity):
    env = state_env()
    env.scene['robot'].data.body_com_lin_vel_w[:] = torch.tensor(velocity)
    cfg = env.cfg
    env.scene['robot'].data.joint_pos[:] = .2
    rewards = {name: term.func(env) for name, term in cfg.rewards.items()}
    expected = dict(zip(cfg.rewards, (velocity[0], abs(velocity[1]), abs(velocity[2]), .04, 0., 0.), strict=True))
    for name, value in rewards.items():
        torch.testing.assert_close(value, torch.full((2,), expected[name]))
    total = sum(cfg.rewards[name].weight * value for name, value in rewards.items())
    torch.testing.assert_close(total, torch.full((2,), velocity[0] - abs(velocity[1]) - abs(velocity[2]) - .01 * .04))
    for axis in range(3):
        torch.testing.assert_close(mdp.world_velocity_component(env, axis), torch.full((2,), velocity[axis]))


@pytest.mark.parametrize('axis', [1, 2])
@pytest.mark.parametrize('velocity', [-.01, -.001, 0., .001, .01])
def test_small_lateral_and_vertical_speeds_have_linear_symmetric_penalty(axis, velocity):
    env = state_env()
    env.scene['robot'].data.body_com_lin_vel_w[..., axis] = velocity
    cost = mdp.lateral_speed_cost if axis == 1 else mdp.vertical_speed_cost
    torch.testing.assert_close(cost(env), torch.full((2,), abs(velocity)), atol=1e-8, rtol=1e-6)


def test_mass_weighted_com_cancels_internal_motion_and_ignores_link_spike():
    env = state_env()
    data = env.scene['robot'].data
    env.sim.model.body_mass[:] = torch.arange(1., 13.)
    data.body_com_lin_vel_w[:, 0] = torch.tensor([12., -24., 36.])
    data.body_com_lin_vel_w[:, -1] = torch.tensor([-1., 2., -3.])
    data.body_link_lin_vel_w[:, 0] = 999.
    torch.testing.assert_close(mdp.world_com_velocity(env), torch.zeros(2, 3))
    data.body_com_lin_vel_w += torch.tensor([.03, -.02, .01])
    torch.testing.assert_close(mdp.world_com_velocity(env), torch.tensor([[.03, -.02, .01]]).expand(2, -1))
    # Read again after mutation: no stale per-step cache.
    data.body_com_lin_vel_w += .01
    torch.testing.assert_close(mdp.world_com_velocity(env), torch.tensor([[.04, -.01, .02]]).expand(2, -1))


@pytest.mark.parametrize('yaw', [0., .4, math.pi / 2, math.pi, -math.pi / 2])
def test_world_x_reference_is_observable_after_random_yaw_without_tilt_reset(yaw):
    env = state_env()
    data = env.scene['robot'].data
    q_yaw = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(q_yaw, data.body_link_quat_w[:, 0])
    reference = mdp.observation_reference(env)
    torch.testing.assert_close(reference.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(quat_apply(data.body_link_quat_w[:, 0], reference), torch.tensor([[1., 0., 0.]]).expand(2, -1), atol=1e-6, rtol=0)
    assert not mdp.head_tilt_limit_reached(env).any()
    assert mdp.actor_state(env).shape == (2, mdp.ACTOR_DIM)
    assert mdp.critic_state(env).shape == (2, mdp.CRITIC_DIM)
    torch.testing.assert_close(mdp.actor_state(env)[:, 24:27], reference)
    data.body_com_lin_vel_w[..., 0] = .05
    torch.testing.assert_close(mdp.forward_speed(env), torch.full((2,), .05 * math.cos(yaw)))
    torch.testing.assert_close(mdp.planar_velocity(env)[:, :2], torch.tensor([[.05 * math.cos(yaw), -.05 * math.sin(yaw)]]).expand(2, -1), atol=1e-6, rtol=0)


def test_legacy_command_observation_is_explicitly_preserved():
    env = state_env()
    env.cfg.observation_reference = 'command'
    mdp.command_term(env).command[:] = torch.tensor([.03, -.01, .2])
    torch.testing.assert_close(mdp.actor_state(env)[:, 24:27], mdp.command_term(env).command)
    env.cfg.observation_reference = 'invalid'
    with pytest.raises(ValueError, match='未知蛇形观测参考系'):
        mdp.observation_reference(env)


def test_joint_pose_cost_is_relative_to_initial_configuration():
    env = state_env()
    data = env.scene['robot'].data
    initial = torch.tensor([.1, -.2, .3, -.4, .5, -.6])
    data.default_joint_pos[:] = initial
    data.joint_pos[:] = initial
    term = env.cfg.rewards['joint_pose_deviation']
    torch.testing.assert_close(term.func(env), torch.zeros(2))
    deviation = torch.tensor([.2, -.2, .2, -.2, .2, -.2])
    data.joint_pos += deviation
    torch.testing.assert_close(term.func(env) * term.weight, torch.full((2,), -.0004))


@pytest.mark.parametrize('yaw', [0., .4, math.pi / 2, math.pi, -math.pi / 2])
@pytest.mark.parametrize('velocity', [(.06, -.02, .01), (-.06, .02, -.01), (0., 0., 0.)])
def test_snake_rewards_rotate_with_heading_and_use_whole_body_com(yaw, velocity):
    env = state_env()
    data = env.scene['robot'].data
    rotation = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]).expand(2, -1)
    data.body_link_quat_w[:, 0] = quat_mul(rotation, data.body_link_quat_w[:, 0])
    forward, lateral, vertical = velocity
    data.body_com_lin_vel_w[:] = torch.tensor([
        forward * math.cos(yaw) - lateral * math.sin(yaw),
        forward * math.sin(yaw) + lateral * math.cos(yaw), vertical,
    ])
    data.body_link_lin_vel_w[:, 0] = 999.
    torch.testing.assert_close(mdp.forward_speed(env), torch.full((2,), forward), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(mdp.lateral_speed_cost(env), torch.full((2,), abs(lateral)), atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(mdp.vertical_speed_cost(env), torch.full((2,), abs(vertical)), atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_target_limit_penalty_uses_commanded_target_and_stops_after_release(sign):
    env = state_env()
    target = env.action_manager.get_term('joint_pos').commanded_target
    term = env.cfg.rewards['target_saturation_fraction']
    target[0, :3] = sign * math.pi / 2
    target[1] = sign * (math.pi / 2 - .002)
    torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([-.05, 0.]))
    # The target can saturate before the physical joint reaches its limit.
    assert not env.scene['robot'].data.joint_pos.any()
    torch.testing.assert_close(term.func(env) * term.weight, torch.tensor([-.05, 0.]))
    target[:] = 0.
    torch.testing.assert_close(term.func(env), torch.zeros(2))
    target[:] = sign * math.pi / 2
    torch.testing.assert_close(term.func(env) * term.weight, torch.full((2,), -.1))
