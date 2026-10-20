"""Guided-only recovery preserves the task and executes bounded physical targets."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from test_transformation_action_bounds import (
    make_action as make_action,  # noqa: PLC0414
)

from zbot_rl_mjlab.tasks.transformation.actions import expand_mirror, project_mirror
from zbot_rl_mjlab.tasks.universal import guided_transitions as guided
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateAction
from zbot_rl_mjlab.tasks.universal.task_fsm import TaskState


def command_env(monkeypatch):
    n = 4
    initial = torch.tensor([True, False, True, False])
    requested = torch.tensor([True, False, False, True])
    data = SimpleNamespace(joint_pos=torch.zeros(n, 6), joint_vel=torch.zeros(n, 6))
    target = torch.zeros(n, 6)
    env = SimpleNamespace(
        num_envs=n, device='cpu', step_dt=.02, common_step_counter=0,
        _omni_initial_standing=initial, _omni_play_standing=requested,
        scene={'robot': SimpleNamespace(data=data)}, heights=torch.full((n,), .06),
    )
    command = guided.GuidedOmniCommandCfg(resampling_time_range=(6., 12.), training=False, task_observation_encoding='scalar').build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    env.action_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(commanded_target=target))
    monkeypatch.setattr(guided.mdp, 'base_height', lambda _: env.heights)
    monkeypatch.setattr(guided.mdp, 'configuration_ready', lambda _, relaxed=False: torch.zeros(n, dtype=torch.bool))
    command.reset(torch.arange(n))
    return command, env, data, target


def step(command, env, count=1):
    for _ in range(count):
        env.common_step_counter += 1
        command._update_command(None)


def test_failed_rise_latches_straightening_until_pose_and_targets_settle(monkeypatch):
    command, env, data, target = command_env(monkeypatch)
    # A straight starting snake tries to rise; bending itself does not relatch.
    step(command, env)
    data.joint_pos[3] = .4
    target[3] = .4
    env.heights[3] = .15
    step(command, env, 5)
    assert not command.straightening_mask.any()
    env.heights[3] = .07
    step(command, env, 5)
    assert command.straightening_mask.tolist() == [False, False, False, True]
    assert command.retry_attempts.tolist() == [0, 0, 0, 1]
    assert command.current_task[3] == TaskState.ASCENDING
    assert command.current_morphology[3] == 0
    # A zero software target cannot release a robot whose physical joints lag.
    target[3] = 0
    step(command, env, 20)
    assert command.straightening_mask[3]
    assert command.start_step[3] == env.common_step_counter
    data.joint_pos[3] = 0
    step(command, env, 4)
    assert command.straightening_mask[3]
    step(command, env)
    assert not command.straightening_mask[3]
    assert command.current_task[3] == TaskState.ASCENDING
    data.joint_pos[3] = .5
    target[3] = .5
    step(command, env)
    assert not command.straightening_mask[3]  # No pending initial latch remains.
    assert command.retry_attempts[3] == 1


def test_initial_bent_pose_and_partial_reset_are_isolated(monkeypatch):
    command, env, data, target = command_env(monkeypatch)
    data.joint_pos[:] = .5
    target[:] = .5
    step(command, env)
    assert command.straightening_mask.tolist() == [False, False, False, True]
    command.retry_attempts[3] = 2
    saved = command._retry_peak.clone()
    command.reset(torch.tensor([0]))
    assert command.straightening_mask[3]
    assert command.retry_attempts[3] == 2
    torch.testing.assert_close(command._retry_peak[1:], saved[1:])


def test_stable_fall_requires_explicit_selected_stand_request(monkeypatch):
    command, env, data, target = command_env(monkeypatch)
    data.joint_pos[0] = .5
    target[0] = .5
    step(command, env, 20)
    assert command.current_task[0] == TaskState.BIPED_MOTION
    assert command.standing_achieved[0]
    assert not command.straightening_mask[0]
    command.set_target('stand', torch.tensor([0]))
    step(command, env)
    assert command.current_task.tolist() == [3, 2, 1, 3]
    assert command.straightening_mask[0]
    assert not command.standing_achieved[0]
    command.retry_attempts[0] = 2
    command.set_target('stand', torch.tensor([0]))
    assert command.retry_attempts[0] == 2  # Repeated key cannot restart rise.


def test_repeated_stand_on_ready_biped_does_not_restart(monkeypatch):
    command, _env, _, _ = command_env(monkeypatch)
    monkeypatch.setattr(guided.mdp, 'configuration_ready', lambda _, relaxed=False: torch.ones(4, dtype=torch.bool))
    command.set_target('stand', torch.tensor([0]))
    assert command.current_task[0] == TaskState.BIPED_MOTION


def test_normal_rise_small_idle_noise_and_duplicate_updates_do_not_retry(monkeypatch):
    command, env, data, target = command_env(monkeypatch)
    step(command, env)
    # A meaningful attempt rising monotonically never triggers the fall rule.
    data.joint_pos[3] = .6
    target[3] = .6
    for height in (.08, .10, .12, .18, .25):
        env.heights[3] = height
        step(command, env, 6)
    assert command.retry_attempts.eq(0).all()
    # Low idle jitter is not an attempted stand even if the base bounces.
    command.reset(torch.tensor([3]))
    data.joint_pos[3] = 0
    target[3] = 0
    env.heights[3] = .075
    step(command, env)
    env.heights[3] = .06
    step(command, env, 10)
    assert command.retry_attempts.eq(0).all()
    before = command._retry_low_time.clone()
    command._update_command(None)
    torch.testing.assert_close(command._retry_low_time, before)


def action_env(make_action):
    base = make_action(initial=0., target_limit=2.)
    command = SimpleNamespace(
        standing_mode=torch.tensor([True, False]),
        transition_mask=torch.ones(2, dtype=torch.bool),
        straightening_mask=torch.zeros(2, dtype=torch.bool),
    )
    base._env.command_manager = SimpleNamespace(get_term=lambda _: command)
    cfg = guided.GuidedTransitionActionCfg(
        entity_name='robot', actuator_names=('.*',), target_limit=2.,
        max_target_speed=1., biped_residual_scale=.375,
    )
    return guided.GuidedTransitionAction(cfg, base._env), command


def test_transition_actions_mirror_without_changing_sampled_actions(make_action):
    action, _ = action_env(make_action)
    sampled = torch.tensor([[1., -.6, .4, .8, -.2, -.4]]).expand(2, -1)
    for _ in range(10):
        before = action.commanded_target.clone()
        action.process_actions(sampled)
        assert (action.commanded_target - before).abs().max() <= .020001
        torch.testing.assert_close(action.commanded_target, expand_mirror(project_mirror(action.commanded_target)))
    torch.testing.assert_close(action.raw_action, sampled)


def test_asymmetric_transition_entry_and_recovery_are_gradual(make_action):
    action, command = action_env(make_action)
    action.commanded_target[:] = torch.tensor([.5, .3, -.2, .7, .1, -.8])
    for _ in range(60):
        before = action.commanded_target.clone()
        action.process_actions(torch.zeros(2, 6))
        assert (action.commanded_target - before).abs().max() <= .020001
    torch.testing.assert_close(action.commanded_target, expand_mirror(project_mirror(action.commanded_target)))
    command.straightening_mask[0] = True
    preserved = action.commanded_target[1].clone()
    for _ in range(60):
        before = action.commanded_target.clone()
        action.process_actions(torch.ones(2, 6))
        assert (action.commanded_target - before).abs().max() <= .020001
    torch.testing.assert_close(action.commanded_target[0], torch.zeros(6))
    torch.testing.assert_close(action.commanded_target[1], preserved)


def test_stable_motion_actions_match_existing_implementation(make_action):
    action, command = action_env(make_action)
    command.transition_mask[:] = False
    reference = SnakeCrawlJointTargetRateAction(action.rate_cfg, action._env)
    for _ in range(100):
        sampled = torch.rand(2, 6) * 2 - 1
        reference.process_actions(sampled)
        action.process_actions(sampled)
        torch.testing.assert_close(action.commanded_target, reference.commanded_target)


def test_recovery_stage_is_observable_and_routes_to_same_head(monkeypatch):
    command, env, data, target = command_env(monkeypatch)
    data.joint_pos[3] = .5
    target[3] = .5
    step(command, env)
    state = torch.randn(4, 56)
    state[:, 39] = command.current_task
    monkeypatch.setattr(guided, 'privileged_state', lambda _: state.clone())
    actual = guided.guided_privileged_state(env)
    assert actual[3, 39] == 3.25
    assert actual[:, 39].round().long().tolist() == [0, 2, 1, 3]
    torch.testing.assert_close(actual[:, :39], state[:, :39])
    torch.testing.assert_close(actual[:, 40:], state[:, 40:])


def test_only_descent_failure_terminates_and_invalid_encoding_fails(monkeypatch):
    _command, env, _, _ = command_env(monkeypatch)
    monkeypatch.setattr(guided, 'transition_failure', lambda _: torch.ones(4, dtype=torch.bool))
    assert guided.guided_transition_failure(env).tolist() == [False, False, True, False]
    with pytest.raises(ValueError, match='scalar'):
        guided.GuidedOmniCommandCfg(resampling_time_range=(6., 12.), task_observation_encoding='one_hot').build(env)


def test_entering_narrower_transition_target_bounds_is_still_rate_limited(make_action):
    action, _ = action_env(make_action)
    cast(guided.GuidedTransitionActionCfg, action.rate_cfg).transition_end_joint_radius = .2
    action.commanded_target[:] = torch.tensor([.8, .3, -.2, .2, -.3, -.8])
    before = action.commanded_target.clone()
    action.process_actions(torch.zeros(2, 6))
    assert (action.commanded_target - before).abs().max() <= .020001
    assert action.commanded_target[:, 0].min() > .2  # Gradually approach new bound.
    for _ in range(50):
        action.process_actions(torch.zeros(2, 6))
    assert action.commanded_target[:, (0, 5)].abs().max() <= .200001


def test_recovery_requires_physical_settlement_not_fast_crossing(monkeypatch):
    command, env, data, target = command_env(monkeypatch)
    data.joint_pos[3] = .4
    step(command, env)
    data.joint_pos[3] = 0
    target[3] = 0
    data.joint_vel[3] = .1
    step(command, env, 20)
    assert command.straightening_mask[3]
    data.joint_vel[3] = 0
    step(command, env, 5)
    assert not command.straightening_mask[3]
