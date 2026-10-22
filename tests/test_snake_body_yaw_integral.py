"""带符号体轴航向积分区分左右摆动与持续单侧偏转。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from test_snake_body_yaw import set_body_heading
from test_snake_task import state_env

from zbot_rl_mjlab.experiments.snake import snake_env_cfg
from zbot_rl_mjlab.tasks.snake import mdp


def integral_env(dt=.02):
    env = state_env()
    env.cfg.body_yaw_limit = math.pi / 3
    env.common_step_counter = 0
    env.step_dt = dt
    mdp.body_yaw_integral(env)
    return env


def advance(env, yaw):
    set_body_heading(env, yaw)
    env.common_step_counter += 1
    return mdp.body_yaw_integral(env).clone()


def test_initial_observation_does_not_integrate_before_a_control_step():
    env = state_env()
    env.cfg.body_yaw_limit = math.pi / 3
    env.common_step_counter = 0
    set_body_heading(env, math.pi / 6)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.zeros(2))
    torch.testing.assert_close(mdp.actor_state(env)[:, 49], torch.zeros(2))


def test_equal_positive_and_negative_heading_durations_cancel_signed_integral():
    env = integral_env(.1)
    for _ in range(10):
        advance(env, math.pi / 6)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.full((2,), math.pi / 6))
    for _ in range(10):
        advance(env, -math.pi / 6)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.body_yaw_integral_cost(env), torch.zeros(2), atol=1e-10, rtol=0)


def test_sustained_opposite_biases_build_equal_symmetric_costs():
    env = integral_env(.1)
    heading = torch.tensor([math.pi / 6, -math.pi / 6])
    advance(env, heading)
    early = mdp.body_yaw_integral_cost(env).clone()
    for _ in range(9):
        advance(env, heading)
    torch.testing.assert_close(mdp.body_yaw_integral(env), heading)
    torch.testing.assert_close(mdp.body_yaw_integral_normalized(env), torch.tensor([.5, -.5]))
    expected = torch.full((2,), math.tanh(.5) ** 2)
    torch.testing.assert_close(mdp.body_yaw_integral_cost(env), expected)
    assert (expected > early).all()


def test_zero_heading_does_not_leak_away_accumulated_bias():
    env = integral_env(.1)
    accumulated = advance(env, math.pi / 6)
    for _ in range(20):
        advance(env, 0.)
    torch.testing.assert_close(mdp.body_yaw_integral(env), accumulated)


@pytest.mark.parametrize('dt', [.01, .02, .1])
def test_one_second_integral_has_radian_seconds_units_independent_of_timestep(dt):
    env = integral_env(dt)
    for _ in range(round(1 / dt)):
        advance(env, .4)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.full((2,), .4), atol=1e-6, rtol=0)
    env.cfg.body_yaw_integral_time_scale = 2.
    torch.testing.assert_close(mdp.body_yaw_integral_normalized(env),
                               torch.full((2,), .4 / (2 * math.pi / 3)), atol=1e-6, rtol=0)


def test_integral_cost_is_bounded_for_long_sustained_heading_error():
    env = integral_env(1.)
    for _ in range(10):
        advance(env, math.pi / 6)
    cost = mdp.body_yaw_integral_cost(env)
    assert torch.isfinite(cost).all()
    assert ((cost > 0) & (cost < 1)).all()
    torch.testing.assert_close(cost, torch.full((2,), math.tanh(5.) ** 2))


def test_reward_metric_and_observation_reads_share_one_update_per_step():
    env = integral_env(.1)
    set_body_heading(env, math.pi / 6)
    env.common_step_counter += 1
    env.max_episode_length_s = 30.
    rewards = RewardManager({'body_yaw_integral_cost': env.cfg.rewards['body_yaw_integral_cost']}, env)
    expected = torch.full((2,), math.pi / 60)
    reward = rewards.compute(env.step_dt)
    torch.testing.assert_close(reward, torch.full((2,), -1. * .1 * math.tanh(.05) ** 2))
    metric = env.cfg.metrics['body_yaw_integral_abs_rad_s'].func(env)
    torch.testing.assert_close(metric, expected)
    actor = mdp.actor_state(env)
    critic = mdp.critic_state(env)
    assert actor.shape == (2, 51)
    assert critic.shape == (2, 66)
    torch.testing.assert_close(actor[:, 49], torch.full((2,), .05))
    torch.testing.assert_close(critic[:, :51], actor)
    # Even changed buffers cannot integrate a second sample at this counter.
    set_body_heading(env, -math.pi / 6)
    torch.testing.assert_close(mdp.body_yaw_integral(env), expected)


def test_partial_state_reset_clears_selected_history_without_sampling_stale_sites():
    env = integral_env(.1)
    advance(env, torch.tensor([math.pi / 6, -math.pi / 6]))
    previous = mdp.body_yaw_integral(env).clone()
    env._snake_body_yaw_integral.reset(torch.tensor([0]), env.common_step_counter)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([0., previous[1]]))
    advance(env, 0.)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([0., previous[1]]))


def test_partial_reset_marks_only_selected_environment_as_sampled_this_step():
    env = integral_env(.1)
    advance(env, torch.tensor([.3, -.3]))
    env.common_step_counter += 1
    mdp.body_yaw_integral_state(env).reset(torch.tensor([0]), env.common_step_counter)
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([0., -.06]))
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([0., -.06]))


def test_absolute_integral_metric_preserves_opposite_environment_biases():
    env = integral_env(.1)
    advance(env, torch.tensor([.3, -.3]))
    metric = env.cfg.metrics['body_yaw_integral_abs_rad_s'].func(env)
    torch.testing.assert_close(metric, torch.full((2,), .03))


@pytest.mark.parametrize('invalid', [0., 1e-5, float('nan'), float('inf')])
def test_invalid_direction_does_not_add_arbitrary_positive_integral(invalid):
    env = integral_env(.1)
    advance(env, torch.tensor([.3, -.3]))
    previous = mdp.body_yaw_integral(env).clone()
    sites = env.scene['robot'].data.site_pos_w
    sites[:, 1] = sites[:, 0]
    sites[:, 1, 0] += invalid
    env.common_step_counter += 1
    torch.testing.assert_close(mdp.body_yaw_integral(env), previous)


@pytest.mark.parametrize('play', [False, True])
def test_reset_snake_clears_selected_integral_with_new_yaw_reference(monkeypatch, play):
    env = integral_env(.1)
    env.cfg = snake_env_cfg(play=play)
    advance(env, torch.tensor([.3, -.3]))
    previous = mdp.body_yaw_integral(env).clone()
    robot = env.scene['robot']
    robot.data.default_root_state = torch.zeros(2, 13)
    robot.data.default_root_state[:, 3:7] = torch.tensor(env.cfg.scene.entities['robot'].init_state.rot)
    robot.write_root_state_to_sim = lambda states, env_ids: None
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: None
    mdp.command_term(env).cfg.training = not play
    env._snake_initial_yaw = torch.tensor([.6, -1.1])
    monkeypatch.setattr(torch, 'rand', lambda count, device: torch.full((count,), .25))
    mdp.reset_snake(env, torch.tensor([0]))
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([0., previous[1]]))
    expected_yaw = 0. if play else math.pi / 2
    torch.testing.assert_close(env._snake_initial_yaw, torch.tensor([expected_yaw, -1.1]),
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize('play', [False, True])
def test_integral_penalty_defaults_are_shared_by_training_and_play(play):
    cfg = snake_env_cfg(play=play)
    assert cfg.body_yaw_integral_time_scale == 1.
    assert cfg.rewards['body_yaw_integral_cost'].func is mdp.body_yaw_integral_cost
    assert cfg.rewards['body_yaw_integral_cost'].weight == -1.
