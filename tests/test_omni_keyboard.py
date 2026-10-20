"""键盘中途改三轴请求及构型，不重置物理状态或传感器历史。"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from mjlab.viewer.base import ViewerAction

from zbot_rl_mjlab.tasks.universal.omni import OmniCommandCfg
from zbot_rl_mjlab.viewer import FollowCameraViewer


@pytest.mark.parametrize('mode', ['stand', 'snake'])
def test_keyboard_combines_site_axes_stops_and_switches_without_reset(mode):
    env = SimpleNamespace(num_envs=1, device='cpu', step_dt=.02, common_step_counter=123)
    term = OmniCommandCfg(training=False, mode=mode, resampling_time_range=(6., 12.)).build(env)
    term.reset(torch.tensor([0]))
    term.set_target(mode)
    robot_state = torch.randn(1, 19)
    before = robot_state.clone()
    calls = []
    env.cfg = SimpleNamespace(commands={'transform': term.cfg})
    env.command_manager = SimpleNamespace(get_term=lambda _: term)
    env.observation_manager = SimpleNamespace(compute=lambda **kwargs: calls.append(kwargs))
    viewer = cast(Any, FollowCameraViewer.__new__(FollowCameraViewer))
    viewer.env = SimpleNamespace(unwrapped=env)
    viewer._universal_controls, viewer._snake_controls = True, False
    viewer.env_idx = 0
    for request in ('motion_forward', 'motion_right', 'motion_turn_left'):
        assert viewer._handle_custom_action(ViewerAction.CUSTOM, request)
    torch.testing.assert_close(term.requested_velocity, torch.tensor([[.05, -.05, .1]]))
    assert term.manual_control.all()
    assert viewer._handle_custom_action(ViewerAction.CUSTOM, 'transform_stand' if mode == 'snake' else 'transform_snake')
    torch.testing.assert_close(term.requested_velocity, torch.tensor([[.05, -.05, .1]]))
    assert viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_stop')
    assert term.requested_velocity.eq(0).all()
    assert env.common_step_counter == 123
    torch.testing.assert_close(robot_state, before)
    assert len(calls) == 5
    for call in calls:
        assert call['update_history'] is True
        torch.testing.assert_close(call['env_ids'], torch.tensor([0]))
