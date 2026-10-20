"""头部有符号航向积分抑制持续偏航，并独立于体轴积分。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from test_snake_body_yaw import set_body_heading
from test_snake_task import state_env
from test_snake_yaw import set_heading

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def integral_env(dt=.1):
    env = state_env()
    env.step_dt = dt
    mdp.head_yaw_integral(env)
    mdp.body_yaw_integral(env)
    return env


def advance(env, yaw):
    set_heading(env, yaw)
    env.common_step_counter += 1
    return mdp.head_yaw_integral(env).clone()


def test_initial_observation_does_not_accumulate_before_first_control_step():
    env = state_env()
    set_heading(env, .4)
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.zeros(2))
    torch.testing.assert_close(mdp.actor_state(env)[:, 50], torch.zeros(2))


def test_equal_left_and_right_heading_durations_cancel_signed_history():
    env = integral_env()
    for _ in range(10):
        advance(env, math.pi / 6)
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.full((2,), math.pi / 6))
    for _ in range(10):
        advance(env, -math.pi / 6)
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.head_yaw_integral_cost(env), torch.zeros(2), atol=1e-10, rtol=0)


def test_sustained_opposite_biases_increase_symmetric_cost():
    env = integral_env()
    yaw = torch.tensor([math.pi / 6, -math.pi / 6])
    advance(env, yaw)
    early = mdp.head_yaw_integral_cost(env).clone()
    for _ in range(9):
        advance(env, yaw)
    torch.testing.assert_close(mdp.head_yaw_integral_normalized(env), torch.tensor([.5, -.5]))
    expected = torch.full((2,), math.tanh(.5) ** 2)
    torch.testing.assert_close(mdp.head_yaw_integral_cost(env), expected)
    assert (expected > early).all()


def test_zero_deviation_preserves_accumulated_history():
    env = integral_env()
    expected = advance(env, .3)
    for _ in range(10):
        advance(env, 0.)
    torch.testing.assert_close(mdp.head_yaw_integral(env), expected)


def test_integral_uses_shortest_signed_error_across_pi():
    env = integral_env()
    env._snake_initial_yaw = torch.deg2rad(torch.tensor([179., -179.]))
    actual = advance(env, torch.deg2rad(torch.tensor([-179., 179.])))
    expected = torch.deg2rad(torch.tensor([2., -2.])) * env.step_dt
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize('dt', [.01, .02, .1])
def test_one_second_integral_is_independent_of_control_timestep(dt):
    env = integral_env(dt)
    for _ in range(round(1 / dt)):
        advance(env, .4)
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.full((2,), .4), atol=1e-6, rtol=0)
    env.cfg.head_yaw_integral_time_scale = 2.
    expected = torch.full((2,), .4 / (2 * env.cfg.head_yaw_limit))
    torch.testing.assert_close(mdp.head_yaw_integral_normalized(env), expected, atol=1e-6, rtol=0)


def test_reward_metric_and_observations_update_history_only_once_per_step():
    env = integral_env()
    set_heading(env, math.pi / 6)
    env.common_step_counter += 1
    env.max_episode_length_s = 30.
    manager = RewardManager({'head_yaw_integral_cost': env.cfg.rewards['head_yaw_integral_cost']}, env)
    torch.testing.assert_close(manager.compute(env.step_dt),
                               torch.full((2,), -.5 * .1 * math.tanh(.05) ** 2))
    metric = env.cfg.metrics['head_yaw_integral_abs_rad_s'].func(env)
    expected = torch.full((2,), math.pi / 60)
    torch.testing.assert_close(metric, expected)
    actor, critic = mdp.actor_state(env), mdp.critic_state(env)
    assert actor.shape == (2, 51)
    assert critic.shape == (2, 66)
    torch.testing.assert_close(actor[:, 50], torch.full((2,), .05))
    torch.testing.assert_close(critic[:, :51], actor)
    set_heading(env, -math.pi / 6)
    torch.testing.assert_close(mdp.head_yaw_integral(env), expected)


def test_head_and_body_histories_follow_their_own_heading():
    env = integral_env()
    set_body_heading(env, torch.tensor([-.2, .1]))
    advance(env, torch.tensor([.3, -.4]))
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.tensor([.03, -.04]))
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([-.02, .01]))
    actor = mdp.actor_state(env)
    torch.testing.assert_close(actor[:, 49], torch.tensor([-.02, .01]) / env.cfg.body_yaw_limit)
    torch.testing.assert_close(actor[:, 50], torch.tensor([.03, -.04]) / env.cfg.head_yaw_limit)


def test_partial_head_state_reset_does_not_clear_body_or_other_environment():
    env = integral_env()
    set_body_heading(env, .2)
    advance(env, torch.tensor([.3, -.3]))
    body = mdp.body_yaw_integral(env).clone()
    mdp.head_yaw_integral_state(env).reset(torch.tensor([0]), env.common_step_counter)
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.tensor([0., -.03]))
    torch.testing.assert_close(mdp.body_yaw_integral(env), body)


def test_partial_reset_only_marks_selected_environment_as_sampled():
    env = integral_env()
    advance(env, torch.tensor([.3, -.3]))
    env.common_step_counter += 1
    mdp.head_yaw_integral_state(env).reset(torch.tensor([0]), env.common_step_counter)
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.tensor([0., -.06]))
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.tensor([0., -.06]))


@pytest.mark.parametrize('play', [False, True])
def test_reset_snake_clears_both_histories_only_for_selected_environments(monkeypatch, play):
    env = integral_env()
    env.cfg = snake_env_cfg(play=play)
    set_body_heading(env, torch.tensor([.2, -.2]))
    advance(env, torch.tensor([.3, -.3]))
    body = mdp.body_yaw_integral(env).clone()
    head = mdp.head_yaw_integral(env).clone()
    robot = env.scene['robot']
    robot.data.default_root_state = torch.zeros(2, 13)
    robot.data.default_root_state[:, 3:7] = torch.tensor(env.cfg.scene.entities['robot'].init_state.rot)
    robot.write_root_state_to_sim = lambda states, env_ids: None
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: None
    mdp.command_term(env).cfg.training = not play
    monkeypatch.setattr(torch, 'rand', lambda count, device: torch.full((count,), .25))
    mdp.reset_snake(env, torch.tensor([0]))
    torch.testing.assert_close(mdp.head_yaw_integral(env), torch.tensor([0., head[1]]))
    torch.testing.assert_close(mdp.body_yaw_integral(env), torch.tensor([0., body[1]]))


def test_long_sustained_error_has_bounded_finite_cost():
    env = integral_env(1.)
    for _ in range(10):
        advance(env, math.pi / 6)
    cost = mdp.head_yaw_integral_cost(env)
    assert torch.isfinite(cost).all()
    assert ((cost > 0) & (cost < 1)).all()
    torch.testing.assert_close(cost, torch.full((2,), math.tanh(5.) ** 2))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_yaw_error_does_not_poison_accumulated_history(monkeypatch, invalid):
    env = integral_env()
    previous = advance(env, torch.tensor([.3, -.3]))
    monkeypatch.setattr(mdp, 'head_yaw_deviation', lambda env: torch.full((2,), invalid))
    env.common_step_counter += 1
    torch.testing.assert_close(mdp.head_yaw_integral(env), previous)


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf')])
def test_invalid_normalization_time_scale_is_rejected(invalid):
    env = integral_env()
    env.cfg.head_yaw_integral_time_scale = invalid
    with pytest.raises(ValueError):
        mdp.head_yaw_integral_normalized(env)


@pytest.mark.parametrize('play', [False, True])
def test_training_and_play_enable_same_head_integral_penalty_and_metric(play):
    cfg = snake_env_cfg(play=play)
    assert cfg.head_yaw_integral_time_scale == 1.
    assert cfg.rewards['head_yaw_integral_cost'].func is mdp.head_yaw_integral_cost
    assert cfg.rewards['head_yaw_integral_cost'].weight == -.5
    assert cfg.metrics['head_yaw_integral_abs_rad_s'].func is mdp.head_yaw_integral_abs
