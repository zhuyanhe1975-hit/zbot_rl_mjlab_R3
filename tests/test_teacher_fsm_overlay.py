"""Native viewer exposes the selected robot's persistent task state."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch
from mjlab.viewer import NativeMujocoViewer

from zbot_rl_mjlab.tasks.universal.task_fsm import TaskStateMachine
from zbot_rl_mjlab.viewer import FollowCameraViewer, task_fsm_overlay_rows


def command():
    fsm = TaskStateMachine(2, "cpu")
    fsm.reset(torch.arange(2), torch.tensor([True, False]), torch.tensor([True, False]))
    return SimpleNamespace(
        task_fsm=fsm,
        requested_velocity=torch.tensor([[.15, -.1, .2], [-.05, .03, -.1]]),
        velocity_command=torch.tensor([[.12, -.08, .15], [-.04, .02, -.07]]),
    )


def test_rows_describe_selected_robot_only():
    term = command()
    biped = dict(task_fsm_overlay_rows(term, 0))
    snake = dict(task_fsm_overlay_rows(term, 1))
    assert biped == {
        "Morphology": "BIPED", "Target": "BIPED", "Task": "BIPED_MOTION",
        "Requested vx/vy/wz": "+0.150 / -0.100 / +0.200",
        "Effective vx/vy/wz": "+0.120 / -0.080 / +0.150",
    }
    assert snake["Morphology"] == "SNAKE"
    assert snake["Task"] == "SNAKE_MOTION"
    assert snake["Requested vx/vy/wz"] == "-0.050 / +0.030 / -0.100"
    assert snake["Effective vx/vy/wz"] == "-0.040 / +0.020 / -0.070"


def test_guided_recovery_stage_and_retry_count_are_explicit():
    term = command()
    term.task_fsm.request(torch.tensor([1]), torch.tensor([True]))
    term.retry_attempts = torch.tensor([0, 2])
    term.straightening_mask = torch.tensor([False, True])
    rows = dict(task_fsm_overlay_rows(term, 1))
    assert rows['Task'] == 'ASCENDING'
    assert rows['Stage'] == 'STRAIGHTEN' and rows['Retries'] == '2'
    term.straightening_mask[1] = False
    assert dict(task_fsm_overlay_rows(term, 1))['Stage'] == 'RISE'


@pytest.mark.parametrize(
    "selected,target,source,goal,task",
    [(0, False, "BIPED", "SNAKE", "DESCENDING"), (1, True, "SNAKE", "BIPED", "ASCENDING")],
)
def test_transition_labels_keep_confirmed_source(selected, target, source, goal, task):
    term = command()
    term.task_fsm.request(torch.tensor([selected]), torch.tensor([target]))
    term.velocity_command[selected] = 0
    rows = dict(task_fsm_overlay_rows(term, selected))
    assert rows["Morphology"] == "TRANSITION"
    assert rows["Source"] == source
    assert rows["Target"] == goal
    assert rows["Task"] == task
    assert rows["Effective vx/vy/wz"] == "+0.000 / +0.000 / +0.000"


def viewer_for(term):
    env = SimpleNamespace(
        cfg=SimpleNamespace(commands={"transform": object()} if term is not None else {}),
        command_manager=SimpleNamespace(get_term=lambda _: term),
    )
    viewer = cast(Any, FollowCameraViewer.__new__(FollowCameraViewer))
    viewer.env = SimpleNamespace(unwrapped=env, num_envs=2)
    viewer.env_idx = 1
    viewer.get_status = Mock(return_value=SimpleNamespace(
        capped=True, paused=False, step_count=123, speed_label="1x",
        target_realtime=1., actual_realtime=.95, smoothed_fps=48.,
    ))
    return viewer


def test_native_overlay_preserves_status_and_uses_selected_state():
    viewer = viewer_for(command())
    native = Mock()
    viewer._set_status_overlay(native)
    overlay = native.set_texts.call_args.args[0]
    assert overlay[:2] == (150, 0)
    assert overlay[2].startswith("Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT\n")
    assert overlay[3].startswith("2/2\n123\nRUNNING [CAPPED]\n1x\n1.00x\n0.95x (48 FPS)\n")
    assert "SNAKE_MOTION" in overlay[3]
    assert "BIPED_MOTION" not in overlay[3]


@pytest.mark.parametrize("term", [None, SimpleNamespace()])
def test_non_fsm_uses_native_parent_unchanged(term):
    viewer = viewer_for(term)
    native = Mock()
    with patch.object(NativeMujocoViewer, "_set_status_overlay") as parent:
        viewer._set_status_overlay(native)
    parent.assert_called_once_with(native)
    viewer.get_status.assert_not_called()
    native.set_texts.assert_not_called()
