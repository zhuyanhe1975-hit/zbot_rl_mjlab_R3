"""Pure tensor and fake-environment checks; no simulation or training required."""

import math
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.universal import mdp


@pytest.mark.parametrize('cmd', [
    [.03, 0., 0.], [0., .5, 0.], [0., 0., .8],
    [-.03, 0., 0.], [0., -.5, 0.], [0., 0., -.8], [.03, .5, -.8],
])
def test_progress_stationary_tracking_saturation_and_reverse(cmd):
    cmd = torch.tensor([cmd])
    for fraction, expected in [(0., 0.), (.5, .5), (1., 1.), (1.1, 1.),
                               (1.2, 1.), (2., .6), (4., -.4), (-1., -1.), (-2., -1.)]:
        torch.testing.assert_close(
            mdp.saturating_progress_score(fraction * cmd, cmd), torch.tensor([expected]),
        )


def test_axis_normalization_and_mixed_translation_yaw_weights():
    limits = (.03, .5, .8)
    cmd = torch.diag(torch.tensor(limits))
    for fraction in (.25, .5, 1.):
        torch.testing.assert_close(
            mdp.saturating_progress_score(fraction * cmd, cmd, limits),
            torch.full((3,), fraction),
        )
    # vx and vy each request their full limit, yaw requests half: weights 2 : .5.
    cmd = torch.tensor([[.03, .5, .4]])
    v = torch.tensor([[.03, .5, 0.]])
    torch.testing.assert_close(mdp.saturating_progress_score(v, cmd, limits), torch.tensor([.8]))


def test_lateral_penalty_floor_cap_and_configurable_weight():
    cmd = torch.tensor([[.03, 0., 0.], [.03, 0., 0.]])
    v = torch.tensor([[.03, .05, 0.], [.03, 10., 0.]])
    torch.testing.assert_close(mdp.saturating_progress_score(v, cmd), torch.tensor([.7, -.2]))
    torch.testing.assert_close(mdp.saturating_progress_score(v, cmd, w_lat=.1), torch.tensor([.9, .6]))
    torch.testing.assert_close(mdp.saturating_progress_score(v, cmd, w_lat=0.), torch.ones(2))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_stop_and_vanishing_translation_are_finite(dtype):
    cmd = torch.tensor([[0., 0., 0.], [1e-30, -1e-30, 0.], [.001, 0., .001],
                        [0., 0., .8], [1e-30, -1e-30, .8]], dtype=dtype)
    v = torch.tensor([[.2, -.5, .8]], dtype=dtype).expand_as(cmd)
    score = mdp.saturating_progress_score(v, cmd)
    assert torch.isfinite(score).all()
    torch.testing.assert_close(score[:3], torch.zeros(3, dtype=dtype))
    torch.testing.assert_close(score[3:], torch.ones(2, dtype=dtype))


def test_stationary_velocity_has_linear_forward_gradient():
    cmd = torch.tensor([[.03, 0., 0.], [0., -.5, 0.], [0., 0., .8]])
    v = torch.zeros_like(cmd, requires_grad=True)
    mdp.saturating_progress_score(v, cmd).sum().backward()
    torch.testing.assert_close(v.grad, torch.diag(torch.tensor([1 / .03, -2., 1 / .8])))


def _env(monkeypatch, n=2, step_dt=.02):
    term = SimpleNamespace(
        velocity_command=torch.tensor([[0., .5, 0.]]).repeat(n, 1),
        transition_mask=torch.zeros(n, dtype=torch.bool),
        cfg=SimpleNamespace(velocity_limits=(.03, .5, .8)),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: term),
        step_dt=step_dt, common_step_counter=0,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        current_velocity=torch.zeros(n, 3),
    )
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda env: env.current_velocity)
    return env, term


@pytest.mark.parametrize('step_dt,time_constant', [(.02, .75), (.1, .2)])
def test_ema_coefficient_and_same_step_idempotence(monkeypatch, step_dt, time_constant):
    env, _ = _env(monkeypatch, step_dt=step_dt)
    torch.testing.assert_close(mdp.smoothed_tracking_velocity(env, time_constant), torch.zeros(2, 3))
    env.common_step_counter += 1
    env.episode_length_buf += 1
    env.current_velocity.fill_(1.)
    expected = torch.full((2, 3), -math.expm1(-step_dt / time_constant))
    torch.testing.assert_close(mdp.smoothed_tracking_velocity(env, time_constant), expected)
    torch.testing.assert_close(mdp.smoothed_tracking_velocity(env, time_constant), expected)


def test_ema_partial_reset_and_repeated_one_step_episodes(monkeypatch):
    env, _ = _env(monkeypatch)
    env.current_velocity.fill_(1.)
    mdp.smoothed_tracking_velocity(env)
    env.common_step_counter = 10
    env.episode_length_buf[:] = torch.tensor([1, 10])
    env.current_velocity.zero_()
    result = mdp.smoothed_tracking_velocity(env)
    torch.testing.assert_close(result[0], torch.zeros(3))
    torch.testing.assert_close(result[1], torch.full((3,), math.exp(-.02 / .75)))
    # First environment resets again and is sampled at episode length 1 again.
    env.common_step_counter += 1
    env.episode_length_buf[:] = torch.tensor([1, 11])
    env.current_velocity[0].fill_(2.)
    torch.testing.assert_close(mdp.smoothed_tracking_velocity(env)[0], torch.full((3,), 2.))
    # Reset can also be observed within the same global step.
    env.episode_length_buf[0] = 0
    env.current_velocity[0].zero_()
    torch.testing.assert_close(mdp.smoothed_tracking_velocity(env)[0], torch.zeros(3))


def test_transition_mask_and_independent_environment_state(monkeypatch):
    env, term = _env(monkeypatch)
    env.current_velocity.copy_(term.velocity_command)
    term.transition_mask[0] = True
    torch.testing.assert_close(mdp.smoothed_velocity_tracking(env), torch.tensor([0., 1.]))
    other, _ = _env(monkeypatch)
    torch.testing.assert_close(mdp.smoothed_velocity_tracking(other), torch.zeros(2))


def test_ema_removes_zero_mean_sway_without_reward_drift(monkeypatch):
    env, term = _env(monkeypatch)
    # Environment 0 oscillates along its lateral command; environment 1 sways
    # perpendicular to a forward command. Both have zero net translation.
    term.velocity_command[1] = torch.tensor([.5, 0., 0.])
    scores, velocities = [], []
    for step in range(2000):
        env.common_step_counter = step
        env.episode_length_buf.fill_(step)
        env.current_velocity[:, 1] = .5 * math.sin(2 * math.pi * 2 * step * env.step_dt)
        score = mdp.smoothed_velocity_tracking(env)
        if step >= 500:  # discard the startup transient, then measure 60 periods
            scores.append(score.clone())
            velocities.append(mdp.smoothed_tracking_velocity(env).clone())
    assert torch.stack(scores).mean(0).abs().max() < .005
    assert torch.stack(velocities)[:, :, 1].square().mean().sqrt() < .05


def test_ema_rejects_nonpositive_time_constant(monkeypatch):
    env, _ = _env(monkeypatch)
    for time_constant in (0., -1.):
        with pytest.raises(ValueError, match='time_constant'):
            mdp.smoothed_tracking_velocity(env, time_constant)
