"""独立蛇形键盘复用线程事件队列，保留综合任务的原步长。"""

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
    ("X", "motion_stop"),
])
def test_snake_native_keys_queue_only(key, payload):
    instance = object.__new__(FollowCameraViewer)
    instance._snake_controls = True
    instance._actions = deque()
    instance.user_key_callback = instance._on_key
    instance._safe_key_callback(ord(key))
    assert list(instance._actions) == [(ViewerAction.CUSTOM, payload)]


def test_snake_brackets_do_not_request_transformation():
    instance = object.__new__(FollowCameraViewer)
    instance._snake_controls = True
    instance._actions = deque()
    for key in "[]":
        instance._on_key(ord(key))
    assert not instance._actions


def test_snake_simulation_event_uses_smaller_steps_and_only_motion_term():
    from test_snake_commands import command_env

    env, term = command_env()
    observations = []
    term_names = []

    def get_term(name):
        term_names.append(name)
        return term

    env.command_manager = SimpleNamespace(get_term=get_term)
    env.observation_manager = SimpleNamespace(compute=lambda **kw: observations.append(kw))
    env.cfg = SimpleNamespace(commands={"motion": term.cfg})
    instance = object.__new__(FollowCameraViewer)
    instance._snake_controls = True
    instance.env = cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=env))
    instance.env_idx = 0
    for payload in ("motion_forward", "motion_left", "motion_turn_left"):
        assert instance._handle_custom_action(ViewerAction.CUSTOM, payload)
    assert term.requested_velocity[0].tolist() == pytest.approx([.025, .025, .1])
    assert term.manual_control[0] and not term.manual_control[1:].any()
    assert term.command.eq(0).all()
    assert instance._handle_custom_action(ViewerAction.CUSTOM, "motion_stop")
    assert term.requested_velocity.eq(0).all()
    assert not instance._handle_custom_action(ViewerAction.CUSTOM, "transform_stand")
    assert term_names == ["motion"] * 4
    assert len(observations) == 4
    assert torch.isinf(term.time_left[0]) and not torch.isinf(term.time_left[1:]).any()
