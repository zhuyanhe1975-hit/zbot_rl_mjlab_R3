"""Native walking targets coexist with the archived bounded task controls."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from test_transformation_action_bounds import (
    make_action as make_action,  # noqa: PLC0414
)

from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS
from zbot_rl_mjlab.tasks.universal.guided_transitions import (
    GuidedTransitionAction,
    GuidedTransitionActionCfg,
)
from zbot_rl_mjlab.tasks.universal.native_actions import (
    NativeGuidedTransitionActionCfg,
)
from zbot_rl_mjlab.tasks.universal.task_fsm import TaskState


def make_native(make_action, task=TaskState.SNAKE_MOTION):
    base = make_action()
    base._entity.data.default_joint_pos[:] = torch.tensor(DEFAULT_JOINT_POS)
    command = SimpleNamespace(
        current_task=torch.tensor([TaskState.BIPED_MOTION, task]),
        standing_mode=torch.tensor([True, task == TaskState.ASCENDING]),
        transition_mask=torch.tensor([False, task in (TaskState.DESCENDING, TaskState.ASCENDING)]),
        straightening_mask=torch.zeros(2, dtype=torch.bool),
    )
    base._env.command_manager = SimpleNamespace(get_term=lambda _: command)
    cfg = NativeGuidedTransitionActionCfg(
        entity_name='robot', actuator_names=('.*',),
        max_target_speed=2 * math.pi, target_limit=None,
        biped_residual_scale=None, transition_end_joint_radius=1.,
    )
    return cfg.build(base._env), command


def test_walking_matches_native_joint_position_action_above_unit_range(make_action):
    action, _ = make_native(make_action)
    # Use the actual mjlab native action processing implementation as oracle;
    # the shared fixture only replaces its simulator-dependent construction.
    native = JointPositionAction(
        JointPositionActionCfg(entity_name='robot', actuator_names=('.*',), scale=.25),
        action._env,
    )
    native.cfg = JointPositionActionCfg(entity_name='robot', actuator_names=('.*',), scale=.25)
    native._scale = .25
    native._offset = action._entity.data.default_joint_pos.clone()
    for values in ([2., -3., 30., -40., 6., -7.], [-8., 9., -20., 30., -6., 7.]):
        samples = torch.tensor([values, values])
        native.process_actions(samples)
        action.process_actions(samples)
        torch.testing.assert_close(action.raw_action[0], samples[0])
        torch.testing.assert_close(action.commanded_target[0], native._processed_actions[0])
    assert action.commanded_target[0].abs().max() > 6.28


@pytest.mark.parametrize('task', [TaskState.DESCENDING, TaskState.SNAKE_MOTION, TaskState.ASCENDING])
@pytest.mark.parametrize('straightening', [False, True])
def test_other_tasks_exactly_preserve_guided_rate_mirror_and_recovery(make_action, task, straightening):
    action, command = make_native(make_action, task)
    command.straightening_mask[1] = straightening and task == TaskState.ASCENDING
    cfg = GuidedTransitionActionCfg(
        entity_name='robot', actuator_names=('.*',),
        max_target_speed=2 * math.pi, target_limit=None,
        biped_residual_scale=None, transition_end_joint_radius=1.,
    )
    reference = GuidedTransitionAction(cfg, action._env)
    action.commanded_target[1] = 0.
    reference.commanded_target[1] = 0.
    samples = torch.tensor([[3., -4., 1., -2., .6, -.8], [2., -.6, .4, .8, -.2, -3.]])
    for _ in range(40):
        before = action.commanded_target[1].clone()
        action.process_actions(samples)
        reference.process_actions(samples)
        torch.testing.assert_close(action.raw_action[1], reference.raw_action[1])
        torch.testing.assert_close(action.commanded_target[1], reference.commanded_target[1])
        assert (action.commanded_target[1] - before).abs().max() <= 2 * math.pi * .02 + 1e-6
    if task == TaskState.SNAKE_MOTION:
        assert action.commanded_target[1].abs().max() <= math.pi / 3 + 1e-6
    if command.straightening_mask[1]:
        torch.testing.assert_close(action.commanded_target[1], torch.zeros(6))


def test_partial_reset_and_task_switch_preserve_control_semantics(make_action):
    action, command = make_native(make_action)
    action.process_actions(torch.full((2, 6), 2.))
    saved = action.commanded_target[0].clone()
    action._entity.data.joint_pos[1] = .2
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.commanded_target[0], saved)
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), .2))
    assert action.raw_action[1].eq(0).all()
    command.current_task[0] = TaskState.DESCENDING
    command.transition_mask[0] = True
    before = action.commanded_target[0].clone()
    action.process_actions(torch.zeros(2, 6))
    assert (action.commanded_target[0] - before).abs().max() <= 2 * math.pi * .02 + 1e-6
    command.current_task[0] = TaskState.BIPED_MOTION
    command.transition_mask[0] = False
    action.process_actions(torch.full((2, 6), -3.))
    torch.testing.assert_close(action.commanded_target[0], torch.tensor(DEFAULT_JOINT_POS) - .75)


@pytest.mark.parametrize('scale', [0., -1., math.inf, math.nan])
def test_native_scale_must_be_finite_and_positive(scale):
    with pytest.raises(ValueError, match='native_biped_scale'):
        NativeGuidedTransitionActionCfg(entity_name='robot', actuator_names=('.*',), native_biped_scale=scale)
