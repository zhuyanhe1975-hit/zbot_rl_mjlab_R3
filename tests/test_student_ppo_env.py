"""Student PPO mechanics keep physical completion and exercise destination motion."""

from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.guided_transitions import GuidedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.student_ppo_env import (
    StudentPPOCommandCfg,
    reset_student_ppo_state,
    student_training_skill_complete,
)
from zbot_rl_mjlab.tasks.universal.task_fsm import TaskState


def make_env(monkeypatch, n=4, training=True, **settings):
    cfg = SimpleNamespace(
        student_ppo_completion_gate=True, completion_enter_tolerance=.15,
        completion_exit_tolerance=.20, student_post_transition_seconds=3.,
        student_continuation_fraction=.5, play_transition_sync=not training,
        standing_joint_pos=(0.,) * 6, standing_base_height=.264,
    )
    cfg.__dict__.update(settings)
    data = SimpleNamespace(joint_pos=torch.zeros(n, 6), joint_vel=torch.zeros(n, 6))
    target = torch.zeros(n, 6)
    env = SimpleNamespace(
        cfg=cfg, num_envs=n, device='cpu', step_dt=.02, common_step_counter=0,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        scene={'robot': SimpleNamespace(data=data)},
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(commanded_target=target)),
        heights=torch.full((n,), .06), support=torch.ones(n),
    )
    command = StudentPPOCommandCfg(
        resampling_time_range=(6., 12.), task_observation_encoding='scalar',
        aligned_teacher_sampling=True, training=training,
    ).build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    ids = torch.arange(n)
    command._command[:, 0] = -1.
    command.task_fsm.reset(ids, torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
    command._publish_task_state()
    monkeypatch.setattr(mdp, 'base_height', lambda _: env.heights)
    monkeypatch.setattr(mdp.standing, 'base_gravity_alignment_cost', lambda _: torch.zeros(n))
    monkeypatch.setattr(mdp, 'contacts', lambda _: torch.ones(n, 3))
    monkeypatch.setattr(mdp, 'normalized_force', lambda _: env.support)
    monkeypatch.setattr(mdp.transform, 'com_state', lambda _: torch.tensor([[.05, 0., 0.]]).expand(n, -1))
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda _: torch.zeros(n, 3))
    return env, command, data, target


def advance(env, command, steps):
    for _ in range(steps):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        command._update_command(None)


@pytest.mark.parametrize('training', [True, False])
def test_hysteresis_survives_small_crossings_and_requires_two_real_seconds(monkeypatch, training):
    env, command, data, target = make_env(monkeypatch, training=training)
    data.joint_pos[:] = .14
    target[:] = .14
    advance(env, command, 1)
    for _ in range(98):
        target[:] = .17 if env.common_step_counter % 2 else .14
        advance(env, command, 1)
        # Duplicate queries and command updates cannot advance the hold timer.
        held = command.hold_time.clone()
        mdp.configuration_ready(env)
        command._update_command(None)
        torch.testing.assert_close(command.hold_time, held)
    assert command.transition_mask.all()
    assert torch.all(command.hold_time < 2.)
    advance(env, command, 2)
    assert not command.transition_mask.any()
    assert torch.all(command.current_task == TaskState.SNAKE_MOTION)
    torch.testing.assert_close(data.joint_pos, torch.full_like(data.joint_pos, .14))


def test_actual_target_exit_and_geometric_failures_restart_the_hold(monkeypatch):
    env, command, data, target = make_env(monkeypatch)
    advance(env, command, 60)
    target[0] = .21
    data.joint_pos[1] = .21
    env.support[2] = 0.
    env.heights[3] = .2
    advance(env, command, 1)
    assert command.hold_time.eq(0).all()
    target[:] = .17
    data.joint_pos[:] = .17
    env.support[:] = 1.
    env.heights[:] = .06
    advance(env, command, 20)
    assert command.transition_mask.all() and command.hold_time.eq(0).all()
    target[:] = .14
    data.joint_pos[:] = .14
    advance(env, command, 101)
    assert not command.transition_mask.any()


def test_collapsed_bent_pose_cannot_handoff_and_reset_is_per_environment(monkeypatch):
    env, command, data, target = make_env(monkeypatch)
    data.joint_pos[:] = .6
    target[:] = .6
    advance(env, command, 130)
    assert command.transition_mask.all()
    data.joint_pos[:] = 0.
    target[:] = 0.
    advance(env, command, 50)
    before = command.hold_time.clone()
    reset_student_ppo_state(env, torch.tensor([0]))
    # Episode reset clears both the candidate and its continuous hold.
    assert not env._student_ppo_state.candidate[0]
    assert env._student_ppo_state.candidate[1:].all()
    assert command.hold_time[0] == 0.
    torch.testing.assert_close(command.hold_time[1:], before[1:])
    command._clear(torch.tensor([0]))
    assert command.hold_time[0] == 0.
    torch.testing.assert_close(command.hold_time[1:], before[1:])


@pytest.mark.parametrize(('iteration', 'categories', 'max_axes', 'scale'), [(0, 7, 1, .5), (20, 19, 2, .6), (40, 27, 3, .7), (100, 27, 3, 1.)])
def test_snake_curriculum_cycles_balanced_signed_categories_with_native_bounds(monkeypatch, iteration, categories, max_axes, scale):
    env, command, _, _ = make_env(monkeypatch, n=categories * 2)
    env._student_rl_iteration = iteration
    ids = torch.arange(env.num_envs)
    command._sample_velocity(ids)
    samples = command.requested_velocity.clone()
    signs, counts = samples.sign().unique(dim=0, return_counts=True)
    assert len(signs) == categories
    assert counts.eq(2).all()
    assert (signs != 0).sum(-1).max() == max_axes
    assert signs.sum(0).eq(0).all()
    limits = samples.new_tensor((.2, .2, .5)) * scale
    assert (samples.abs() <= limits + 1e-6).all()
    nonzero = samples != 0
    assert ((samples.abs() / limits)[nonzero] >= .4 - 1e-6).all()
    # Sampling a subset must continue the cycle rather than always selecting one sign.
    observed = []
    for _ in range(categories):
        command._sample_velocity(ids[:1])
        observed.append(command.requested_velocity[0].sign().clone())
    assert len(torch.stack(observed).unique(dim=0)) == categories


def test_biped_sampler_is_unchanged_and_manual_rows_are_preserved(monkeypatch):
    env, command, _, _ = make_env(monkeypatch, n=64)
    command._command[:, 0] = 1.
    reference = GuidedOmniCommandCfg(
        resampling_time_range=(6., 12.), task_observation_encoding='scalar', aligned_teacher_sampling=True,
    ).build(env)
    ids = torch.arange(env.num_envs)
    torch.manual_seed(14)
    reference._sample_velocity(ids)
    torch.manual_seed(14)
    command._sample_velocity(ids)
    torch.testing.assert_close(command.requested_velocity, reference.requested_velocity, atol=0, rtol=0)
    command._command[:, 0] = -1.
    command.manual_control[0] = True
    command.requested_velocity[0] = .123
    command._sample_velocity(ids)
    torch.testing.assert_close(command.requested_velocity[0], torch.full((3,), .123))


def complete_roles(env, command):
    command._training_roles[:] = torch.tensor([0, 1, 2, 3])
    command.task_fsm.update_completion(torch.ones(env.num_envs, dtype=torch.bool))
    command._publish_task_state()


def test_continuations_keep_motion_for_three_seconds_without_teleport_or_manual_override(monkeypatch):
    env, command, data, target = make_env(monkeypatch, student_continuation_fraction=1.)
    complete_roles(env, command)
    before, targets = data.joint_pos.clone(), target.clone()
    assert not student_training_skill_complete(env).any()
    assert command.requested_velocity[2:].norm(dim=-1).gt(0).all()
    assert command.time_left[2:].isinf().all() and not command.manual_control.any()
    desired = command.requested_velocity[2:].clone()
    command._resample(torch.tensor([2, 3]))
    torch.testing.assert_close(command.requested_velocity[2:], desired)
    env.common_step_counter += 149
    assert not student_training_skill_complete(env).any()
    env.common_step_counter += 1
    assert student_training_skill_complete(env).tolist() == [False, False, True, True]
    torch.testing.assert_close(data.joint_pos, before)
    torch.testing.assert_close(target, targets)


def test_immediate_half_and_play_do_not_cut_motion_trials_or_mutate_commands(monkeypatch):
    env, command, _, _ = make_env(monkeypatch, student_continuation_fraction=0.)
    complete_roles(env, command)
    assert student_training_skill_complete(env).tolist() == [False, False, True, True]
    command.cfg.training = False
    desired = command.requested_velocity.clone()
    assert not student_training_skill_complete(env).any()
    torch.testing.assert_close(command.requested_velocity, desired)


def test_new_request_episode_reset_and_clock_rollback_clear_stale_candidates(monkeypatch):
    env, command, data, target = make_env(monkeypatch)
    advance(env, command, 50)
    data.joint_pos[:] = .17
    target[:] = .17
    command.start_step[0] += 1
    env.episode_length_buf[1] = 0
    ready = mdp.configuration_ready(env)
    assert ready.tolist() == [False, False, True, True]
    env.common_step_counter = 0
    assert not mdp.configuration_ready(env).any()


def test_completion_buffers_created_during_play_remain_mutable_outside_inference(monkeypatch):
    env, command, _, _ = make_env(monkeypatch, training=False)
    with torch.inference_mode():
        assert mdp.configuration_ready(env).all()
    assert not env._student_ppo_state.candidate.is_inference()
    reset_student_ppo_state(env, torch.tensor([1]))
    assert env._student_ppo_state.candidate.tolist() == [True, False, True, True]
    assert command.hold_time.eq(0).all()


def test_handoff_samples_once_and_stop_fallback_does_not_depend_on_random_luck(monkeypatch):
    env, command, _, _ = make_env(monkeypatch, student_continuation_fraction=1.)
    complete_roles(env, command)
    calls = []
    def sample(ids):
        calls.append(ids.clone())
        command.requested_velocity[ids] = 0.
    monkeypatch.setattr(command, '_sample_velocity', sample)
    assert not student_training_skill_complete(env).any()
    torch.testing.assert_close(command.requested_velocity[2:, 0], torch.full((2,), .08))
    for _ in range(5):
        student_training_skill_complete(env)
    assert len(calls) == 1
    reset_student_ppo_state(env, torch.tensor([2]))
    assert not env._student_ppo_state.handoff[2]
    assert env._student_ppo_state.handoff[3]


@pytest.mark.parametrize('settings', [
    {'completion_enter_tolerance': 0.},
    {'completion_enter_tolerance': .25, 'completion_exit_tolerance': .2},
    {'completion_exit_tolerance': float('nan')},
    {'student_post_transition_seconds': -1.},
    {'student_continuation_fraction': 1.1},
])
def test_invalid_completion_configuration_fails_explicitly(monkeypatch, settings):
    env, _, _, _ = make_env(monkeypatch, **settings)
    with pytest.raises(ValueError):
        mdp.configuration_ready(env)


def test_play_sampling_uses_full_categories_and_amplitudes_independent_of_actor_progress(monkeypatch):
    env, command, _, _ = make_env(monkeypatch, n=2700, training=False)
    ids = torch.arange(env.num_envs)
    torch.manual_seed(123)
    command._sample_velocity(ids)
    first = command.requested_velocity.clone()
    assert len(first.sign().unique(dim=0)) == 27
    normalized = first.abs() / first.new_tensor((.2, .2, .5))
    assert normalized.amax(0).gt(.99).all()
    assert normalized.le(1.).all()
    # Replay commands do not change their distribution with a restored iteration.
    env._student_rl_iteration = 100
    command._category_cursor = 0
    torch.manual_seed(123)
    # First call also shuffled categories, so compare bounds and coverage instead of RNG sequence.
    command._sample_velocity(ids)
    assert len(command.requested_velocity.sign().unique(dim=0)) == 27
    normalized = command.requested_velocity.abs() / first.new_tensor((.2, .2, .5))
    assert normalized.amax(0).gt(.99).all()
    command.set_velocity(.03, -.02, .1, torch.tensor([0]))
    command._sample_velocity(ids)
    torch.testing.assert_close(command.requested_velocity[0], torch.tensor([.03, -.02, .1]))


@pytest.mark.parametrize('name', [
    'snake_single_axis_iterations', 'snake_two_axis_iterations', 'snake_command_ramp_iterations',
])
@pytest.mark.parametrize('value', [True, False, -1, .5])
def test_curriculum_iteration_counts_require_nonnegative_integers(name, value):
    cfg = StudentPPOCommandCfg(
        resampling_time_range=(6., 12.), task_observation_encoding='scalar', **{name: value},
    )
    env = SimpleNamespace(num_envs=1, device='cpu', step_dt=.02, common_step_counter=0)
    with pytest.raises(ValueError, match='nonnegative integer'):
        cfg.build(env)
