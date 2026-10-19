"""世界X任务可视化不随蛇头旋转，旧指令控制仅供旧模型。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul
from mjlab.viewer import NativeMujocoViewer
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg
from zbot_rl_mjlab.viewer import FollowCameraViewer


@pytest.mark.parametrize('yaw', [0., math.pi / 2])
def test_snake_forward_and_actual_velocity_arrows_share_site_origin(yaw):
    env = state_env()
    data = env.scene['robot'].data
    rotation = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    data.body_link_quat_w[:, 0] = quat_mul(rotation.expand(2, -1), data.body_link_quat_w[:, 0])
    data.site_pos_w = torch.tensor([[[1., 2., .3]], [[3., 4., .5]]])
    velocity = torch.tensor([.2, -.1, .1])
    data.body_com_lin_vel_w[:] = velocity
    term = SnakeVelocityCommandCfg(resampling_time_range=(5., 9.), debug_vis=True).build(env)
    arrows = []
    term.debug_vis(SimpleNamespace(
        get_env_indices=lambda count: [0], add_sphere=lambda *args, **kwargs: None,
        add_arrow=lambda start, end, **kwargs: arrows.append((torch.as_tensor(start), torch.as_tensor(end))),
    ))
    assert len(arrows) == 2
    origin = data.site_pos_w[0, 0]
    torch.testing.assert_close(arrows[0][0], origin)
    torch.testing.assert_close(arrows[1][0], origin)
    torch.testing.assert_close(arrows[0][1], origin + torch.tensor([.15 * math.cos(yaw), .15 * math.sin(yaw), 0.]))
    torch.testing.assert_close(arrows[1][1], origin + .6 * velocity)


def test_world_task_disables_irrelevant_motion_keys_but_legacy_keeps_them(monkeypatch):
    monkeypatch.setattr(NativeMujocoViewer, '__init__', lambda *args, **kwargs: None)
    env = state_env()
    wrapped = SimpleNamespace(unwrapped=env)
    assert not FollowCameraViewer(wrapped, None)._motion_controls()
    env.cfg.observation_reference = 'command'
    assert FollowCameraViewer(wrapped, None)._motion_controls()
