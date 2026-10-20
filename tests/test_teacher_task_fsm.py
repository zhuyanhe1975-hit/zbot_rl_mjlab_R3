"""Requests, rather than instantaneous posture, select teacher tasks."""

import torch

from zbot_rl_mjlab.tasks.universal.task_fsm import (
    Morphology,
    TaskState,
    TaskStateMachine,
)


def reset_four() -> TaskStateMachine:
    fsm = TaskStateMachine(4, "cpu")
    fsm.reset(
        torch.arange(4), torch.tensor([True, False, True, False]),
        torch.tensor([True, False, False, True]),
    )
    return fsm


def snapshot(fsm: TaskStateMachine) -> list[torch.Tensor]:
    return [x.clone() for x in (
        fsm.current_morphology, fsm.source_morphology,
        fsm.target_morphology, fsm.task_state,
    )]


def test_reset_matched_and_opposed_tasks():
    fsm = reset_four()
    assert fsm.current_morphology.tolist() == [1, -1, 0, 0]
    assert fsm.source_morphology.tolist() == [1, -1, 1, -1]
    assert fsm.target_morphology.tolist() == [1, -1, -1, 1]
    assert fsm.task_state.tolist() == [0, 2, 1, 3]
    assert fsm.transition_mask.tolist() == [False, False, True, True]


def test_same_goal_requests_are_noops_even_during_transition():
    fsm = reset_four()
    before = snapshot(fsm)
    for _ in range(10):
        fsm.request(torch.arange(4), torch.tensor([True, False, False, True]))
        for expected, actual in zip(before, snapshot(fsm), strict=True):
            torch.testing.assert_close(actual, expected)


def test_request_changes_selected_robot_only():
    fsm = reset_four()
    before = snapshot(fsm)
    fsm.request(torch.tensor([1]), torch.tensor([True]))
    assert fsm.current_morphology[1] == Morphology.TRANSITION
    assert fsm.source_morphology[1] == Morphology.SNAKE
    assert fsm.target_morphology[1] == Morphology.BIPED
    assert fsm.task_state[1] == TaskState.ASCENDING
    for expected, actual in zip(before, snapshot(fsm), strict=True):
        torch.testing.assert_close(actual[[0, 2, 3]], expected[[0, 2, 3]])


def test_only_settled_transitions_complete_and_motion_does_not_flip():
    fsm = reset_four()
    fsm.update_completion(torch.tensor([True, True, True, False]))
    assert fsm.current_morphology.tolist() == [1, -1, -1, 0]
    assert fsm.task_state.tolist() == [0, 2, 2, 3]
    assert fsm.source_morphology.tolist() == [1, -1, -1, -1]
    # No posture/contact classification enters this API. Lost readiness cannot
    # turn a confirmed walking task into a crawl or restart transformation.
    before = snapshot(fsm)
    for _ in range(20):
        fsm.update_completion(torch.zeros(4, dtype=torch.bool))
    for expected, actual in zip(before, snapshot(fsm), strict=True):
        torch.testing.assert_close(actual, expected)


def test_reversing_transition_requires_settlement_even_for_source_goal():
    fsm = reset_four()
    fsm.request(torch.tensor([2, 3]), torch.tensor([True, False]))
    assert fsm.current_morphology.tolist() == [1, -1, 0, 0]
    assert fsm.task_state.tolist() == [0, 2, 3, 1]
    assert fsm.source_morphology.tolist() == [1, -1, 1, -1]
    fsm.update_completion(torch.tensor([False, False, True, False]))
    assert fsm.current_morphology.tolist() == [1, -1, 1, 0]
    assert fsm.task_state.tolist() == [0, 2, 0, 1]


def test_partial_reset_keeps_other_rows():
    fsm = reset_four()
    before = snapshot(fsm)
    fsm.reset(torch.tensor([2]), torch.tensor([False]), torch.tensor([True]))
    assert fsm.source_morphology[2] == Morphology.SNAKE
    assert fsm.task_state[2] == TaskState.ASCENDING
    for expected, actual in zip(before, snapshot(fsm), strict=True):
        torch.testing.assert_close(actual[[0, 1, 3]], expected[[0, 1, 3]])


def test_play_allocation_and_keyboard_callback_cross_inference_context():
    with torch.inference_mode():
        fsm = reset_four()
    assert not fsm.task_state.is_inference()
    fsm.request(torch.tensor([0]), torch.tensor([False]))
    fsm.update_completion(torch.tensor([True, False, False, False]))
    fsm.reset(torch.tensor([1]), torch.tensor([True]), torch.tensor([True]))
    assert fsm.task_state.tolist() == [2, 0, 1, 3]


def test_empty_selection_and_completion_are_noops():
    fsm = reset_four()
    before = snapshot(fsm)
    fsm.request(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool))
    fsm.reset(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.bool),
    )
    fsm.update_completion(torch.zeros(4, dtype=torch.bool))
    for expected, actual in zip(before, snapshot(fsm), strict=True):
        torch.testing.assert_close(actual, expected)


def test_unknown_initial_morphology_requires_confirmation_without_guess():
    fsm = TaskStateMachine(2, 'cpu')
    fsm.reset(torch.arange(2), None, torch.tensor([True, False]))
    assert fsm.current_morphology.tolist() == [0, 0]
    assert fsm.source_morphology.tolist() == [0, 0]
    assert fsm.task_state.tolist() == [3, 1]
    fsm.update_completion(torch.tensor([True, True]))
    assert fsm.current_morphology.tolist() == [1, -1]
    assert fsm.source_morphology.tolist() == [1, -1]


def test_forced_verification_preserves_known_source_even_when_goal_matches():
    fsm = TaskStateMachine(2, 'cpu')
    fsm.reset(
        torch.arange(2), torch.tensor([True, False]), torch.tensor([True, False]),
        force_transition=True,
    )
    assert fsm.source_morphology.tolist() == [1, -1]
    assert fsm.current_morphology.tolist() == [0, 0]
    assert fsm.task_state.tolist() == [3, 1]
    fsm.update_completion(torch.tensor([True, True]))
    assert fsm.task_state.tolist() == [0, 2]
