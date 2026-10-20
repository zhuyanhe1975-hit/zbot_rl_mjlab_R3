"""回合初始航向约束及可观测的蛇头姿态。"""

import math

import pytest
import torch
from mjlab.managers.termination_manager import TerminationManager
from mjlab.utils.lab_api.math import quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def set_heading(env, yaw):
    angles = torch.as_tensor(yaw, dtype=torch.float32).expand(env.num_envs)
    rotation = torch.stack((angles.div(2).cos(), torch.zeros_like(angles),
                            torch.zeros_like(angles), angles.div(2).sin()), -1)
    initial = torch.tensor(env.cfg.scene.entities['robot'].init_state.rot)
    env.scene['robot'].data.body_link_quat_w[:, 0] = quat_mul(rotation, initial.expand_as(rotation))


def test_yaw_deviation_wraps_across_pi_using_shortest_signed_angle():
    env = state_env()
    env._snake_initial_yaw = torch.deg2rad(torch.tensor([179., -179.]))
    set_heading(env, torch.deg2rad(torch.tensor([-179., 179.])))
    torch.testing.assert_close(mdp.head_yaw_deviation(env), torch.deg2rad(torch.tensor([2., -2.])),
                               atol=1e-6, rtol=0)


def test_yaw_cost_is_symmetric_and_normalized_to_limit():
    env = state_env()
    env._snake_initial_yaw = torch.zeros(2)
    set_heading(env, torch.tensor([math.pi / 6, -math.pi / 6]))
    torch.testing.assert_close(mdp.head_yaw_cost(env), torch.full((2,), .25))
    assert env.cfg.rewards['head_yaw_cost'].weight == -.5


@pytest.mark.parametrize('degrees,terminated', [(60., False), (-60., False),
                                               (60.001, True), (-60.001, True)])
def test_yaw_reset_requires_strictly_more_than_sixty_degrees(degrees, terminated):
    env = state_env()
    env._snake_initial_yaw = torch.zeros(2)
    set_heading(env, math.radians(degrees))
    assert mdp.head_yaw_limit_reached(env).tolist() == [terminated, terminated]


def test_yaw_reference_is_independent_for_each_environment():
    env = state_env()
    env._snake_initial_yaw = torch.tensor([.7, -1.2])
    set_heading(env, env._snake_initial_yaw + torch.tensor([.2, -.4]))
    torch.testing.assert_close(mdp.head_yaw_deviation(env), torch.tensor([.2, -.4]), atol=1e-6, rtol=0)


@pytest.mark.parametrize('play', [False, True])
def test_yaw_termination_remains_active_in_training_and_manual_play(play):
    env = state_env()
    env.cfg = snake_env_cfg(play=play)
    env._snake_initial_yaw = torch.zeros(2)
    mdp.command_term(env).manual_control = torch.ones(2, dtype=torch.bool)
    assert env.cfg.head_yaw_limit == pytest.approx(math.pi / 3)
    assert env.cfg.terminations['head_yaw'].func is mdp.head_yaw_limit_reached
    assert not env.cfg.terminations['head_yaw'].time_out
    manager = TerminationManager({name: env.cfg.terminations[name]
                                  for name in ('head_yaw', 'head_tilt')}, env)
    set_heading(env, torch.tensor([math.radians(61), math.radians(59)]))
    torch.testing.assert_close(manager.compute(), torch.tensor([True, False]))
    torch.testing.assert_close(manager.get_term('head_tilt'), torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize('play', [False, True])
def test_partial_reset_records_new_root_heading_instead_of_stale_site(monkeypatch, play):
    env = state_env()
    env.cfg = snake_env_cfg(play=play)
    robot = env.scene['robot']
    robot.data.default_root_state = torch.zeros(2, 13)
    robot.data.default_root_state[:, 3:7] = torch.tensor(env.cfg.scene.entities['robot'].init_state.rot)
    roots = []
    robot.write_root_state_to_sim = lambda states, env_ids: roots.append(states.clone())
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: None
    mdp.command_term(env).cfg.training = not play
    env._snake_initial_yaw = torch.tensor([.6, -1.1])
    set_heading(env, torch.tensor([-2., 1.7]))
    monkeypatch.setattr(torch, 'rand', lambda count, device: torch.full((count,), .25))
    mdp.reset_snake(env, torch.tensor([0]))
    expected_yaw = 0. if play else math.pi / 2
    torch.testing.assert_close(env._snake_initial_yaw, torch.tensor([expected_yaw, -1.1]),
                               atol=1e-6, rtol=0)
    # Sim buffers deliberately remain stale until the physics forward update.
    robot.data.body_link_quat_w[0, 0] = roots[0][0, 3:7]
    torch.testing.assert_close(mdp.head_yaw_deviation(env)[0], torch.tensor(0.), atol=1e-6, rtol=0)


def test_boot_yaw_reference_uses_default_root_pose_before_first_reset():
    env = state_env()
    if hasattr(env, '_snake_initial_yaw'):
        del env._snake_initial_yaw
    robot = env.scene['robot']
    set_heading(env, torch.tensor([.8, -1.4]))
    robot.data.default_root_state = torch.zeros(2, 13)
    robot.data.default_root_state[:, 3:7] = robot.data.body_link_quat_w[:, 0]
    set_heading(env, torch.tensor([1., -1.7]))
    torch.testing.assert_close(mdp.head_yaw_deviation(env), torch.tensor([.2, -.3]), atol=1e-6, rtol=0)


def test_actor_appends_canonical_normalized_head_pose_and_relative_yaw():
    env = state_env()
    env._snake_initial_yaw = torch.tensor([.4, -.6])
    set_heading(env, torch.tensor([.7, -1.1]))
    expected = env.scene['robot'].data.body_link_quat_w[:, 0].clone()
    env.scene['robot'].data.body_link_quat_w[:, 0] *= -2.
    torch.testing.assert_close(mdp.head_orientation(env), expected)
    # Restore unit inputs for the existing gravity and velocity observations.
    env.scene['robot'].data.body_link_quat_w[:, 0] = -expected
    obs = mdp.actor_state(env)
    assert obs.shape == (2, 51)
    assert mdp.critic_state(env).shape == (2, 66)
    torch.testing.assert_close(obs[:, 43:47], expected)
    deviation = torch.tensor([.3, -.5])
    torch.testing.assert_close(obs[:, 47:49], torch.stack((deviation.sin(), deviation.cos()), -1),
                               atol=1e-6, rtol=0)
