"""按键仅接管所选环境，保留其他机器人的自动指令和历史。"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from mjlab.viewer.base import ViewerAction

from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg
from zbot_rl_mjlab.viewer import FollowCameraViewer


def test_selected_robot_keeps_independent_velocity_mode_and_history():
    env = SimpleNamespace(num_envs=3, device='cpu', step_dt=.02, common_step_counter=0)
    env._omni_play_standing = torch.tensor([True, False, True])
    term = PrivilegedOmniCommandCfg(training=False, mode='source', resampling_time_range=(6., 12.)).build(env)
    term.reset(torch.arange(3))
    calls = []
    env.cfg = SimpleNamespace(commands={'transform': term.cfg})
    env.command_manager = SimpleNamespace(get_term=lambda _: term)
    env.observation_manager = SimpleNamespace(compute=lambda **kw: calls.append(kw))
    viewer = cast(Any, FollowCameraViewer.__new__(FollowCameraViewer))
    viewer.env = SimpleNamespace(unwrapped=env)
    viewer._universal_controls, viewer._snake_controls = True, False
    viewer.env_idx = 1
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_forward')
    assert term.requested_velocity[:, 0].tolist() == pytest.approx([.05, .1, .05])
    assert term.manual_control.tolist() == [False, True, False]
    viewer.env_idx = 0
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_stop')
    assert term.requested_velocity[:, 0].tolist() == pytest.approx([0., .1, .05])
    viewer.env_idx = 1
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'motion_forward')
    assert term.requested_velocity[:, 0].tolist() == pytest.approx([0., .15, .05])
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'transform_stand')
    assert term.standing_mode.tolist() == [True, True, True]
    assert not term.manual_control[2]
    viewer.env_idx = 2
    viewer._handle_custom_action(ViewerAction.CUSTOM, 'transform_snake')
    assert term.standing_mode.tolist() == [True, True, False]
    assert term.requested_velocity[:, 0].tolist() == pytest.approx([0., .15, .05])
    for call, selected in zip(calls, [1, 0, 1, 1, 2], strict=True):
        assert call['update_history'] is True
        torch.testing.assert_close(call['env_ids'], torch.tensor([selected]))
