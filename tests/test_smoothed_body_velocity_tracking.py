"""COM reward regressions using actual frame math, without simulation/training."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.tasks.universal import mdp

LIMITS = {'biped': (.5, .2, .5), 'snake': (.06, .06, .5)}


def environment(n=2):
    quat = torch.tensor([2**-.5, 0., -2**-.5, 0.]).expand(n, 2, 4).clone()
    data = SimpleNamespace(
        body_link_quat_w=quat, body_com_lin_vel_w=torch.zeros(n, 2, 3),
        body_link_ang_vel_w=torch.zeros(n, 2, 3),
        site_quat_w=quat[:, :1].clone(), site_lin_vel_w=torch.zeros(n, 1, 3),
    )
    term = SimpleNamespace(
        velocity_command=torch.tensor([[0., .06, 0.]]).repeat(n, 1),
        standing_mode=torch.zeros(n, dtype=torch.bool),
        current_task=torch.full((n,), 2), transition_mask=torch.zeros(n, dtype=torch.bool),
        heading=torch.tensor([[1., 0.]]).repeat(n, 1),
        initial_heading=torch.tensor([[1., 0.]]).repeat(n, 1),
        heading_initialized=torch.zeros(n, dtype=torch.bool),
    )
    robot = SimpleNamespace(
        data=data, find_bodies=lambda _: ([0], ['base']),
        find_sites=lambda _: ([0], ['foot_0_sole']),
        indexing=SimpleNamespace(body_ids=torch.arange(2)),
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(omni_site_frame=True, guided_local_reference=True),
        scene={'robot': robot}, sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.tensor([[1., 3.]]))),
        command_manager=SimpleNamespace(get_term=lambda _: term),
        common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long), step_dt=.02,
    )
    return env, term, data


def advance(env):
    env.common_step_counter += 1
    env.episode_length_buf += 1


def yaw_quat(angle):
    return torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])


def test_head_swing_and_mass_balanced_body_motion_cannot_fake_com_progress():
    env, _, data = environment()
    initial = data.site_quat_w.clone()
    scores = []
    for step in range(120):
        sway = math.sin(2 * math.pi * step / 20)
        data.site_quat_w[:] = quat_mul(yaw_quat(2. * sway).expand_as(initial), initial)
        data.site_lin_vel_w[:, :, 1] = .3 * sway
        # Unequal masses: the body point velocities cancel only when weighted.
        data.body_com_lin_vel_w[:, 0, 1] = .3 * sway
        data.body_com_lin_vel_w[:, 1, 1] = -.1 * sway
        scores.append(mdp.smoothed_body_velocity_tracking(env, LIMITS))
        advance(env)
    torch.testing.assert_close(torch.stack(scores), torch.zeros(120, 2), atol=1e-6, rtol=0.)


def test_uniform_com_lateral_motion_scores_one():
    env, _, data = environment()
    data.body_com_lin_vel_w[:, :, 1] = .06
    torch.testing.assert_close(mdp.smoothed_body_velocity_tracking(env, LIMITS), torch.ones(2))


def test_head_local_command_is_rotated_into_body_frame():
    env, term, data = environment()
    # Head points world +Y but BASE still points world +X. Local +Y is world -X.
    data.site_quat_w[:] = quat_mul(yaw_quat(math.pi / 2).expand_as(data.site_quat_w), data.site_quat_w)
    data.body_com_lin_vel_w[:, :, 0] = -.06
    torch.testing.assert_close(mdp.smoothed_body_velocity_tracking(env, LIMITS), torch.ones(2))
    assert torch.allclose(term.velocity_command[:, 1], torch.full((2,), .06))


def test_world_velocity_and_unit_heading_ema_once_per_step():
    env, _, data = environment()
    mdp.smoothed_body_velocity_tracking(env, LIMITS)
    advance(env)
    data.body_com_lin_vel_w[:, :, 0] = .06
    data.body_link_quat_w[:] = quat_mul(yaw_quat(math.pi / 2).expand_as(data.body_link_quat_w), data.body_link_quat_w)
    first = mdp.smoothed_body_velocity_tracking(env, LIMITS)
    alpha = -math.expm1(-env.step_dt / .75)
    state = env.command_manager.get_term('transform')._smoothed_body_tracking_state
    torch.testing.assert_close(state[0][:, 0], torch.full((2,), .06 * alpha))
    torch.testing.assert_close(state[1], torch.tensor([[1 - alpha, alpha]]).repeat(2, 1))
    torch.testing.assert_close(mdp.smoothed_body_velocity_tracking(env, LIMITS), first)


def test_partial_reset_repeated_short_episodes_and_same_step_reset():
    env, term, data = environment()
    data.body_com_lin_vel_w[:, :, 1] = .06
    mdp.smoothed_body_velocity_tracking(env, LIMITS)
    advance(env)
    data.body_com_lin_vel_w.zero_()
    env.episode_length_buf[0] = 0
    result = mdp.smoothed_body_velocity_tracking(env, LIMITS)
    torch.testing.assert_close(result, torch.tensor([0., math.exp(-.02 / .75)]))
    # Reset heading and velocity together, including repeated one-step episodes.
    for _ in range(2):
        advance(env)
        env.episode_length_buf[0] = 0
        data.body_link_quat_w[0] = quat_mul(yaw_quat(math.pi / 2).expand_as(data.body_link_quat_w[0]), data.body_link_quat_w[0])
        mdp.smoothed_body_velocity_tracking(env, LIMITS)
        assert term._smoothed_body_tracking_state[0][0].abs().max() == 0
    advance(env)
    mdp.smoothed_body_velocity_tracking(env, LIMITS)
    data.body_com_lin_vel_w[0, :, 1] = .06
    env.episode_length_buf[0] = 0
    torch.testing.assert_close(mdp.smoothed_body_velocity_tracking(env, LIMITS)[0], torch.tensor(1.))


def test_mode_limits_stop_transition_and_root_yaw():
    env, term, data = environment(4)
    term.standing_mode[0] = True
    term.current_task[0] = 0
    term.velocity_command[:] = torch.tensor([0., .06, .5])
    # Biped forward +Y, lateral -X. Snake forward +X, lateral +Y.
    data.body_com_lin_vel_w[0, :, 0] = -.06
    data.body_com_lin_vel_w[1:, :, 1] = .06
    term.velocity_command[2] = 0.
    term.transition_mask[3] = True
    score = mdp.smoothed_body_velocity_tracking(env, LIMITS)
    torch.testing.assert_close(score, torch.tensor([.3 / 1.3, .5, 0., 0.]))
    # Reset bypasses EMA startup; root world wz is tracked, not head angular sway.
    advance(env)
    env.episode_length_buf.zero_()
    term.velocity_command[:] = torch.tensor([0., 0., .5])
    data.body_link_ang_vel_w[:, 0, 2] = .5
    torch.testing.assert_close(mdp.smoothed_body_velocity_tracking(env, LIMITS), torch.tensor([1., 1., 1., 0.]))


@pytest.mark.parametrize('value', [0., -1., math.nan, math.inf])
def test_invalid_time_constant(value):
    env, _, _ = environment()
    with pytest.raises(ValueError, match='time_constant'):
        mdp.smoothed_body_velocity_tracking(env, LIMITS, time_constant=value)
