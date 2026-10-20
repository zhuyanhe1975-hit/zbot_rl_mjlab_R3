"""综合任务的键盘事件只入队，由仿真线程更新目标指令。"""

from collections import deque
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer.base import ViewerAction

from zbot_rl_mjlab.viewer import FollowCameraViewer


@pytest.mark.parametrize("key,payload", [
    ("W", "motion_forward"), ("S", "motion_backward"),
    ("A", "motion_left"), ("D", "motion_right"),
    ("Q", "motion_turn_left"), ("E", "motion_turn_right"),
    ("X", "motion_stop"), ("[", "transform_snake"), ("]", "transform_stand"),
])
def test_keys_only_queue_one_command_without_native_shortcut_conflicts(key, payload):
    instance = object.__new__(FollowCameraViewer)
    instance._universal_controls = True
    instance._actions = deque()
    instance.user_key_callback = instance._on_key
    instance._safe_key_callback(ord(key))
    assert list(instance._actions) == [(ViewerAction.CUSTOM, payload)]


def test_simulation_thread_changes_velocity_and_mode_without_touching_physics():
    from test_universal_task import command_env

    env, term = command_env()
    observations = []
    env.observation_manager = SimpleNamespace(compute=lambda **kw: observations.append(kw))
    joints = torch.randn(2, 6)
    targets = torch.randn(2, 6)
    env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=joints))}
    env.action_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(commanded_target=targets))
    env.cfg = SimpleNamespace(commands={"transform": term.cfg})
    initial_joints, initial_targets = joints.clone(), targets.clone()
    instance = object.__new__(FollowCameraViewer)
    instance._universal_controls = True
    instance.env = cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=env))
    instance.env_idx = 0
    for payload in ("motion_forward", "motion_left", "motion_turn_left", "transform_snake"):
        assert instance._handle_custom_action(ViewerAction.CUSTOM, payload)
    assert term.requested_velocity[0].tolist() == pytest.approx([.05, .05, .1])
    assert term.standing_mode.tolist() == [False, True]
    assert term.manual_control.tolist() == [True, False]
    assert instance._handle_custom_action(ViewerAction.CUSTOM, "motion_stop")
    assert term.requested_velocity.eq(0).all()
    assert term.standing_mode.tolist() == [False, True]
    torch.testing.assert_close(joints, initial_joints)
    torch.testing.assert_close(targets, initial_targets)
    assert len(observations) == 5
