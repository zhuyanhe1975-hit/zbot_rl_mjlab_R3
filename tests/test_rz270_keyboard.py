"""Playback keys change selected world-heading targets, preserving other robots."""

import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from mjlab.viewer.base import ViewerAction

from zbot_rl_mjlab.tasks.snake import rz270_heading as task
from zbot_rl_mjlab.viewer import FollowCameraViewer, task_fsm_overlay_rows


@pytest.fixture
def controls(monkeypatch):
    env = SimpleNamespace(num_envs=3, device='cpu', step_dt=.02, common_step_counter=0)
    term = task.HeadingSnakeCommandCfg(training=False, play_random_velocity=True).build(env)
    term.reset(torch.arange(3))
    term.target_heading[1] = 0.
    term.requested_velocity[1, 0] = .1
    calls = []
    env.cfg = SimpleNamespace(commands={'transform': term.cfg})
    env.command_manager = SimpleNamespace(get_term=lambda _: term)
    env.observation_manager = SimpleNamespace(compute=lambda **kw: calls.append(kw))
    monkeypatch.setattr(task.mdp, 'heading', lambda _: torch.tensor([[1., 0.]]).expand(3, -1))
    viewer = cast(Any, FollowCameraViewer.__new__(FollowCameraViewer))
    viewer.env = SimpleNamespace(unwrapped=env)
    viewer._universal_controls, viewer._snake_controls, viewer._site_snake_controls = True, False, True
    viewer.env_idx = 1
    return viewer, term, calls


@pytest.mark.parametrize('payload,angle,speed', [
    ('motion_forward', 0., .125), ('motion_backward', 0., .075),
    ('motion_left', math.pi / 18, .1), ('motion_right', -math.pi / 18, .1),
    ('motion_turn_left', math.pi / 18, .1), ('motion_turn_right', -math.pi / 18, .1),
])
def test_keys_control_forward_speed_and_heading_independently(controls, payload, angle, speed):
    viewer, term, calls = controls
    before = term.target_heading.clone()
    speeds_before = term.requested_velocity.clone()
    assert viewer._handle_custom_action(ViewerAction.CUSTOM, payload)
    assert float(term.target_heading[1]) == pytest.approx(angle, abs=1e-6)
    torch.testing.assert_close(term.target_heading[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(term.requested_velocity[[0, 2]], speeds_before[[0, 2]])
    assert term.manual_control.tolist() == [False, True, False]
    assert term.requested_velocity[1].tolist() == pytest.approx([speed, 0., 0.])
    assert torch.isinf(term.time_left[1]) and torch.isfinite(term.time_left[[0, 2]]).all()
    assert calls[0]['update_history'] is True
    torch.testing.assert_close(calls[0]['env_ids'], torch.tensor([1]))
    term._sample_velocity(torch.arange(3))
    assert float(term.target_heading[1]) == pytest.approx(angle, abs=1e-6)


def test_fine_turns_wrap_and_stop_preserves_heading_target(controls):
    viewer, term, _ = controls
    term.target_heading[1] = math.radians(179)
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_turn_left')
    assert math.degrees(float(term.target_heading[1])) == pytest.approx(-171., abs=1e-5)
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_turn_right')
    assert math.degrees(float(term.target_heading[1])) == pytest.approx(179., abs=1e-5)
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_stop')
    assert math.degrees(float(term.target_heading[1])) == pytest.approx(179., abs=1e-5)
    assert term.requested_velocity[1].eq(0.).all()
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_left')
    assert term.requested_velocity[1, 0] == 0.
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_forward')
    assert term.requested_velocity[1, 0] == pytest.approx(.025)
    term.set_forward_speed(-1., torch.tensor([1]))
    assert term.requested_velocity[1, 0] == 0.
    term.set_forward_speed(99., torch.tensor([1]))
    assert term.requested_velocity[1, 0] == pytest.approx(.2)
    term.reset(torch.tensor([1]))
    assert not term.manual_control[1] and torch.isfinite(term.time_left[1])


def test_heading_overlay_shows_actual_heading_units_and_keyboard_mode(controls):
    viewer, term, _ = controls
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_left')
    rows = dict(task_fsm_overlay_rows(term, 1, torch.zeros(3, 3)))
    assert rows['Heading target (deg)'] == '+10.0'
    assert rows['Heading error (deg)'] == '+10.0'
    assert rows['Heading control'] == 'MANUAL'
    assert 'Head site vx/vy/wz' in rows
    assert 'Requested vx/vy/wz' not in rows
