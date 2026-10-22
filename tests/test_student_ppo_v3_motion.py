"""V3 rewards and sampling require movement without changing V2 defaults."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.guided_transitions import GuidedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.student_ppo_env import (
    StudentPPOCommandCfg,
    reset_student_ppo_state,
)
from zbot_rl_mjlab.tasks.universal.student_ppo_rewards import (
    student_motion_progress_score,
    student_velocity_progress,
)
from zbot_rl_mjlab.tasks.universal.task_fsm import TaskState


def score(actual, command):
    return student_motion_progress_score(torch.tensor(actual), torch.tensor(command), (.1, .1, .25))


@pytest.mark.parametrize('command', [[.08, 0., 0.], [-.08, 0., 0.], [0., .05, 0.], [0., 0., -.3], [.08, .05, .3], [1e-8, 0., 0.]])
def test_motion_requires_requested_direction_and_penalizes_overspeed(command):
    requested = torch.tensor(command)
    actual = torch.stack((requested * 0, requested * .5, requested, -requested, requested * 2, requested * 3))
    value = student_motion_progress_score(actual, requested.expand_as(actual), (.1, .1, .25))
    torch.testing.assert_close(value, torch.tensor([0., 2., 4., -4., 0., -4.]))


def test_mixed_axis_failure_and_inactive_drift_cannot_be_hidden():
    values = score([[.08, 0., 0.], [.08, .05, .3], [.08, .05, -.3]], [[.08, .05, .3]] * 3)
    assert 0 < values[0] < 1
    assert values[1] == 4
    assert values[2] < 0
    drift = score([[.08, 0., 0.], [.08, .1, 0.], [0., .1, 0.]], [[.08, 0., 0.]] * 3)
    assert drift[0] == 4 and 0 < drift[1] < 4 and drift[2] < 0


def test_stop_requires_all_axes_still_and_nonfinite_never_receives_credit():
    values = score([[0., 0., 0.], [.1, 0., 0.], [float('nan'), 0., 0.], [float('inf'), 0., 0.]], [[0., 0., 0.]] * 4)
    assert values[0] == 4 and 0 < values[1] < 4
    assert torch.isfinite(values).all() and (values[2:] <= 0).all()
    assert score([[0., 0., 0.]], [[float('nan'), 0., 0.]])[0] <= 0


@pytest.mark.parametrize('std', [(0., .1, .25), (-.1, .1, .25), (float('nan'), .1, .25), (.1, .1)])
def test_invalid_reward_widths_are_rejected(std):
    with pytest.raises(ValueError, match='std'):
        student_motion_progress_score(torch.zeros(1, 3), torch.zeros(1, 3), std)


def test_wrapper_gates_transitions_and_applies_snake_drift_width(monkeypatch):
    command = SimpleNamespace(
        current_task=torch.tensor([TaskState.BIPED_MOTION, TaskState.SNAKE_MOTION, TaskState.ASCENDING, TaskState.DESCENDING]),
        standing_mode=torch.tensor([True, False, True, False]),
        transition_mask=torch.tensor([False, False, True, True]),
        velocity_command=torch.tensor([[.08, 0., 0.]] * 4),
    )
    monkeypatch.setattr(mdp, 'command_term', lambda _: command)
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda _: torch.tensor([[.08, .05, 0.]] * 4))
    values = student_velocity_progress(None)
    assert 0 < values[1] < values[0] < 4
    assert values[2:].eq(0).all()


def make_command(n=18, **settings):
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0)
    cfg = StudentPPOCommandCfg(
        resampling_time_range=(6., 12.), task_observation_encoding='scalar',
        aligned_teacher_sampling=True, **settings,
    )
    command = cfg.build(env)
    command._command[n // 2:, 0] = -1.
    return env, command


def test_balanced_sampler_covers_each_validation_direction_for_both_morphologies():
    _, command = make_command(balanced_motion_commands=True, snake_command_initial_scale=1.)
    command._sample_velocity(torch.arange(18))
    expected = torch.tensor([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.], [1., 1., 1.], [-1., -1., -1.]])
    for samples in command.requested_velocity.split(9):
        torch.testing.assert_close(samples.sign().unique(dim=0), expected.unique(dim=0))
        target = samples.new_tensor((.08, .05, .3))
        nonzero = samples != 0
        assert ((samples.abs() / target)[nonzero] >= 1).all()
        assert ((samples.abs() / target)[nonzero] <= 1.5).all()


def test_balanced_sampler_keeps_per_morphology_cursors_and_manual_commands():
    _, command = make_command(balanced_motion_commands=True)
    observed = []
    for _ in range(9):
        command._sample_velocity(torch.tensor([0, 9]))
        observed.append(command.requested_velocity[[0, 9]].sign().clone())
    for samples in torch.stack(observed).unbind(1):
        assert len(samples.unique(dim=0)) == 9
    command.manual_control[[0, 9]] = True
    command.requested_velocity[[0, 9]] = .123
    before = command._balanced_cursors.copy()
    command._sample_velocity(torch.tensor([0, 9]))
    torch.testing.assert_close(command.requested_velocity[[0, 9]], torch.full((2, 3), .123))
    assert command._balanced_cursors == before


def test_balanced_sampler_preserves_commands_during_handoff_continuation():
    env, command = make_command(balanced_motion_commands=True)
    env.cfg = SimpleNamespace(
        completion_enter_tolerance=.15, completion_exit_tolerance=.20,
        student_post_transition_seconds=3., student_continuation_fraction=.5,
    )
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    ids = torch.arange(18)
    command._sample_velocity(ids)
    before = command.requested_velocity.clone()
    reset_student_ppo_state(env)
    env._student_ppo_state.handoff[:] = True
    env._student_ppo_state.continuation[:] = True
    cursors = command._balanced_cursors.copy()
    command._resample_command(ids)
    torch.testing.assert_close(command.requested_velocity, before)
    assert command.time_left.isinf().all()
    assert command._balanced_cursors == cursors


@pytest.mark.parametrize(('iteration', 'scale'), [(0, .5), (50, .75), (100, 1.)])
def test_balanced_commands_respect_native_limits_and_snake_amplitude_ramp(iteration, scale):
    env, command = make_command(
        balanced_motion_commands=True, training=True,
        biped_velocity_ranges=((- .09, .09), (-.06, .06), (-.35, .35)),
        snake_velocity_limits=(.06, .04, .2),
    )
    env._student_rl_iteration = iteration
    command._sample_velocity(torch.arange(18))
    assert (command.requested_velocity[:9].abs() <= torch.tensor([.09, .06, .35])).all()
    snake = command.requested_velocity[9:].abs()
    limits = torch.tensor([.06, .04, .2]) * scale
    assert (snake <= limits).all()
    torch.testing.assert_close(snake.max(0).values, limits)


def test_balancing_preserves_evaluation_construction_sampling_gait_and_rng():
    snapshots = []
    for balanced in (False, True):
        torch.manual_seed(456)
        _, command = make_command(training=False, balanced_motion_commands=balanced)
        states = [torch.random.get_rng_state()]
        for ids in (torch.arange(18), torch.tensor([0, 3, 9, 15])):
            command._sample_velocity(ids)
            states.extend((
                command.requested_velocity.clone(), command.gait_frequency.clone(),
                command._biped_gait_frequency.clone(), torch.random.get_rng_state(),
            ))
        assert command._balanced_categories == {}
        snapshots.append(states)
    for original, enabled in zip(*snapshots, strict=True):
        torch.testing.assert_close(original, enabled, atol=0, rtol=0)


def test_default_biped_sampler_preserves_prior_rng_and_outputs():
    env, command = make_command()
    reference = GuidedOmniCommandCfg(
        resampling_time_range=(6., 12.), task_observation_encoding='scalar', aligned_teacher_sampling=True,
    ).build(env)
    ids = torch.arange(9)
    torch.manual_seed(123)
    reference._sample_velocity(ids)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(123)
    command._sample_velocity(ids)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    torch.testing.assert_close(command.requested_velocity[ids], reference.requested_velocity[ids], atol=0, rtol=0)
    assert command._balanced_categories == {} and not cast(StudentPPOCommandCfg, command.cfg).balanced_motion_commands
