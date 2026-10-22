"""头尾连线相对回合初始航向的约束独立于头部姿态。"""

import math

import pytest
import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from test_snake_task import state_env
from test_snake_yaw import set_heading

from zbot_rl_mjlab.experiments.snake import snake_env_cfg
from zbot_rl_mjlab.tasks.snake import mdp


def set_body_heading(env, yaw):
    angles = torch.as_tensor(yaw, dtype=torch.float32).expand(env.num_envs)
    sites = env.scene['robot'].data.site_pos_w
    direction = torch.stack((angles.cos(), angles.sin()), -1)
    sites[:, 1, :2] = sites[:, 0, :2] - .6 * direction


def heading_manager(env):
    return TerminationManager({name: env.cfg.terminations[name]
                               for name in ('head_yaw', 'body_yaw', 'head_tilt')}, env)


def test_body_yaw_wraps_across_pi_using_shortest_signed_angle():
    env = state_env()
    env._snake_initial_yaw = torch.deg2rad(torch.tensor([179., -179.]))
    set_body_heading(env, torch.deg2rad(torch.tensor([-179., 179.])))
    torch.testing.assert_close(mdp.body_yaw_deviation(env),
                               torch.deg2rad(torch.tensor([2., -2.])), atol=1e-6, rtol=0)


def test_body_yaw_cost_is_symmetric_and_normalized_to_thirty_degrees():
    env = state_env()
    set_body_heading(env, torch.tensor([math.pi / 6, -math.pi / 6]))
    torch.testing.assert_close(mdp.body_yaw_cost(env), torch.ones(2))
    assert env.cfg.rewards['body_yaw_cost'].weight == -.1


@pytest.mark.parametrize('degrees,terminated', [(30., False), (-30., False),
                                               (31., True), (-31., True)])
def test_body_yaw_reset_requires_strictly_more_than_thirty_degrees(degrees, terminated):
    env = state_env()
    set_body_heading(env, math.radians(degrees))
    assert mdp.body_yaw_limit_reached(env).tolist() == [terminated, terminated]


def test_body_yaw_uses_each_environments_initial_heading():
    env = state_env()
    env._snake_initial_yaw = torch.tensor([.7, -1.2])
    set_body_heading(env, env._snake_initial_yaw + torch.tensor([.2, -.4]))
    torch.testing.assert_close(mdp.body_yaw_deviation(env), torch.tensor([.2, -.4]),
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize('play', [False, True])
def test_body_yaw_failure_is_independent_of_head_heading_and_remains_active_in_manual_play(play):
    env = state_env()
    env.cfg = snake_env_cfg(play=play)
    mdp.command_term(env).manual_control = torch.ones(2, dtype=torch.bool)
    set_body_heading(env, torch.deg2rad(torch.tensor([61., 0.])))
    set_heading(env, torch.deg2rad(torch.tensor([0., 61.])))
    assert env.cfg.tail_site == 'foot_1_sole'
    assert env.cfg.body_yaw_limit == pytest.approx(math.pi / 6)
    assert env.cfg.terminations['body_yaw'].func is mdp.body_yaw_limit_reached
    assert not env.cfg.terminations['body_yaw'].time_out
    manager = heading_manager(env)
    torch.testing.assert_close(manager.compute(), torch.tensor([True, True]))
    torch.testing.assert_close(manager.get_term('body_yaw'), torch.tensor([True, False]))
    torch.testing.assert_close(manager.get_term('head_yaw'), torch.tensor([False, True]))
    assert not manager.get_term('head_tilt').any()


def test_body_yaw_projects_sites_horizontally():
    env = state_env()
    set_body_heading(env, math.pi / 6)
    env.scene['robot'].data.site_pos_w[:, 1, 2] = 100.
    torch.testing.assert_close(mdp.body_yaw_deviation(env), torch.full((2,), math.pi / 6))


@pytest.mark.parametrize('invalid', [0., 1e-5, 1e-4, float('nan'), float('inf')])
def test_invalid_head_tail_direction_has_finite_cost_and_resets(invalid):
    env = state_env()
    sites = env.scene['robot'].data.site_pos_w
    sites[0, 1] = sites[0, 0]
    sites[0, 1, 0] += invalid
    torch.testing.assert_close(mdp.body_yaw_deviation(env), torch.tensor([math.pi, 0.]))
    torch.testing.assert_close(mdp.body_yaw_cost(env), torch.tensor([36., 0.]))
    assert mdp.body_yaw_limit_reached(env).tolist() == [True, False]


@pytest.mark.parametrize('dt', [.01, .02, .04])
@pytest.mark.parametrize('body_failed,head_failed', [(False, False), (True, False),
                                                   (False, True), (True, True)])
def test_either_heading_failure_costs_five_without_double_charging(dt, body_failed, head_failed):
    env = state_env()
    env.step_dt = dt
    env.max_episode_length_s = 30.
    set_body_heading(env, torch.deg2rad(torch.tensor([61. if body_failed else 0., 0.])))
    set_heading(env, torch.deg2rad(torch.tensor([61. if head_failed else 0., 0.])))
    env.termination_manager = heading_manager(env)
    env.termination_manager.compute()
    rewards = RewardManager({name: env.cfg.rewards[name]
                             for name in ('head_yaw_reset_cost', 'body_yaw_reset_cost')}, env)
    expected = -5. if body_failed or head_failed else 0.
    torch.testing.assert_close(rewards.compute(dt), torch.tensor([expected, 0.]))
    assert env.cfg.rewards['body_yaw_reset_cost'].weight == -5.


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_alive_excludes_body_yaw_failure_and_recovers_after_reset(dt):
    env = state_env()
    set_body_heading(env, torch.deg2rad(torch.tensor([61., 0.])))
    env.termination_manager = heading_manager(env)
    env.termination_manager.compute()
    rewards = RewardManager({'alive': env.cfg.rewards['alive']}, env)
    torch.testing.assert_close(common_mdp.is_alive(env), torch.tensor([0., 1.]))
    torch.testing.assert_close(rewards.compute(dt), torch.tensor([0., .02 * dt]))
    env.termination_manager.reset(torch.tensor([0]))
    set_body_heading(env, 0.)
    env.termination_manager.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.full((2,), .02 * dt))
