"""低速课程仅由四向纯平移的实际跟踪能力升级。"""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg


def curriculum_env(monkeypatch):
    env = SimpleNamespace(num_envs=8, device='cpu', step_dt=.02, common_step_counter=0)
    term = SnakeVelocityCommandCfg(
        training=True, sample_velocity_commands=True, resampling_time_range=(5., 9.),
        translation_min_speed=.02, translation_curriculum=(.06, .10, .15, .20, .25),
    ).build(cast(ManagerBasedRlEnv, env))
    requested = torch.tensor([[.06, 0., 0.], [-.06, 0., 0.], [0., .06, 0.], [0., -.06, 0.]]).repeat(2, 1)
    term.requested_velocity.copy_(requested)
    term._command.copy_(requested)
    env.actual = requested.clone()
    env.tilted = torch.zeros(8, dtype=torch.bool)
    monkeypatch.setattr(mdp, 'planar_velocity', lambda _: env.actual)
    monkeypatch.setattr(mdp, 'head_tilt_limit_reached', lambda _: env.tilted)
    return env, term


def advance(env, term, steps=600):
    for _ in range(steps):
        env.common_step_counter += 1
        term._update_translation_curriculum()


def test_four_directions_must_track_for_sustained_time_before_promotion(monkeypatch):
    env, term = curriculum_env(monkeypatch)
    advance(env, term, 400)
    assert term.translation_stage == 0
    advance(env, term, 200)
    assert term.translation_stage == 1
    assert term.current_translation_cap == .10
    # 旧阶段的低速指令不能让新阶段未经考核继续升级。
    advance(env, term, 600)
    assert term.translation_stage == 1


def test_each_stage_requires_new_near_cap_tracking_and_final_cap_is_bounded(monkeypatch):
    env, term = curriculum_env(monkeypatch)
    for stage, cap in enumerate((.06, .10, .15, .20)):
        term.requested_velocity[:] = torch.tensor([[cap, 0., 0.], [-cap, 0., 0.], [0., cap, 0.], [0., -cap, 0.]]).repeat(2, 1)
        term._command.copy_(term.requested_velocity)
        env.actual.copy_(term.command)
        advance(env, term)
        assert term.translation_stage == stage + 1
    assert term.current_translation_cap == .25
    advance(env, term)
    assert term.translation_stage == 4


@pytest.mark.parametrize('case', ['stationary', 'stop', 'yaw', 'mixed', 'manual', 'missing_direction', 'bad_direction', 'unsettled', 'too_easy', 'tilted', 'spinning'])
def test_easy_or_failed_samples_never_promote(monkeypatch, case):
    env, term = curriculum_env(monkeypatch)
    if case == 'stationary':
        env.actual.zero_()
    elif case == 'stop':
        term.requested_velocity.zero_()
        term._command.zero_()
        env.actual.zero_()
    elif case == 'yaw':
        term.requested_velocity[:, :2] = 0
        term.requested_velocity[:, 2] = .3
        term._command.copy_(term.requested_velocity)
        env.actual.copy_(term.command)
    elif case == 'mixed':
        term.requested_velocity[:, 2] = .3
        term._command.copy_(term.requested_velocity)
        env.actual.copy_(term.command)
    elif case == 'manual':
        term.manual_control.fill_(True)
    elif case == 'missing_direction':
        term.manual_control[[3, 7]] = True
    elif case == 'bad_direction':
        env.actual[[3, 7]] = 0
    elif case == 'unsettled':
        term._command.mul_(.5)
    elif case == 'too_easy':
        term.requested_velocity.mul_(1 / 3)
        term._command.copy_(term.requested_velocity)
        env.actual.copy_(term.command)
    elif case == 'tilted':
        env.tilted[[3, 7]] = True
    elif case == 'spinning':
        env.actual[:, 2] = .5
    advance(env, term)
    assert term.translation_stage == 0


def test_duplicate_compute_and_partial_reset_do_not_advance_curriculum(monkeypatch):
    env, term = curriculum_env(monkeypatch)
    env.common_step_counter = 1
    term._update_translation_curriculum()
    observed = term._translation_observed_s.clone()
    success = term._translation_success.clone()
    term._update_translation_curriculum()
    term._update_command(torch.tensor([0]))
    torch.testing.assert_close(term._translation_observed_s, observed)
    torch.testing.assert_close(term._translation_success, success)


def test_low_speed_samples_and_manual_play_use_distinct_limits(monkeypatch):
    torch.manual_seed(18)
    _, term = curriculum_env(monkeypatch)
    term.manual_control.zero_()
    term._resample_command(torch.arange(8))
    assert term.requested_velocity[:, :2].abs().le(.06).all()
    nonzero = term.requested_velocity[:, :2].abs()
    assert nonzero[nonzero > 0].ge(.02).all()
    term.set_velocity(.25, -.25, 1.)
    torch.testing.assert_close(term.requested_velocity[0], torch.tensor([.25, -.25, 1.]))
    term.cfg.training = False
    assert term.current_translation_cap == .25


def test_stage_round_trip_and_reset_preserve_learned_cap(monkeypatch):
    _, term = curriculum_env(monkeypatch)
    term.translation_stage = 2
    state = term.curriculum_state()
    _, target = curriculum_env(monkeypatch)
    target.load_curriculum_state(state)
    target.reset(None)
    assert target.translation_stage == 2
    assert target.current_translation_cap == .15
    assert target.requested_velocity[:, :2].abs().le(.15).all()
    target.translation_stage = 4
    assert target.current_translation_cap == .25


@pytest.mark.parametrize('state', [None, {}, {'caps': (.06,), 'min_speed': .02, 'stage': 0},
    {'caps': (.06, .10, .15, .20, .25), 'min_speed': .02, 'stage': True},
    {'caps': (.06, .10, .15, .20, .25), 'min_speed': .02, 'stage': 5}])
def test_invalid_saved_course_does_not_mutate_stage(monkeypatch, state):
    _, term = curriculum_env(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        term.load_curriculum_state(state)
    assert term.translation_stage == 0


@pytest.mark.parametrize('kwargs', [
    {'translation_min_speed': -.01}, {'translation_min_speed': math.nan},
    {'translation_curriculum': (.06, .05)}, {'translation_curriculum': (.06, .3)},
    {'translation_curriculum': (math.nan,)}, {'translation_curriculum': (.06,), 'translation_min_speed': .07},
])
def test_invalid_course_rejected(kwargs):
    env = SimpleNamespace(num_envs=1, device='cpu', step_dt=.02)
    with pytest.raises(ValueError):
        SnakeVelocityCommandCfg(resampling_time_range=(5., 9.), **kwargs).build(cast(ManagerBasedRlEnv, env))


def test_stationary_and_moving_joints_are_visible_in_diagnostics():
    from test_snake_task import state_env

    env = state_env()
    torch.testing.assert_close(mdp.joint_velocity_rms(env), torch.zeros(2))
    env.scene['robot'].data.joint_vel[1] = torch.tensor([.3, -.3, .3, -.3, .3, -.3])
    torch.testing.assert_close(mdp.joint_velocity_rms(env), torch.tensor([0., .3]))
    assert env.cfg.metrics['joint_velocity_rms_rad_s'].func is mdp.joint_velocity_rms
    assert 'target_windup_cost' not in env.cfg.metrics
    assert env.cfg.metrics['target_saturation_fraction'].func is mdp.target_saturation_fraction
