"""Commands and measured arrows share BASE / head-site origins in Guided play."""
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.debug_visualizer import DebugVisualizer
from test_guided_reference_frames import environment

from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommand
from zbot_rl_mjlab.tasks.universal.guided_observations import GuidedTeacherEnvCfg


@pytest.mark.parametrize('play', [False, True])
def test_measured_velocity_visuals_are_play_only(play):
    cfg = cast(GuidedTeacherEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided', play=play))
    assert cfg.command_visual_actual_velocity == play


@pytest.mark.parametrize('task', [0, 1, 2, 3])
def test_command_and_actual_arrows_use_the_same_correct_reference(task):
    env, term, data = environment((task,))
    env.num_envs = 1
    env.cfg.omni_site_frame = True
    env.cfg.motion_site = 'foot_0_sole'
    env.cfg.command_visual_actual_velocity = True
    data.body_link_pos_w = torch.tensor([[[1., 2., .3]]])
    data.site_pos_w = torch.tensor([[[4., 5., .05]]])
    # Both points move opposite the positive command and yaw in the opposite direction.
    data.body_link_lin_vel_w[:] = torch.tensor([0., -.1, 0.])
    data.site_lin_vel_w[:] = torch.tensor([-.1, 0., 0.])
    data.body_link_ang_vel_w[:] = torch.tensor([0., 0., -.2])
    data.site_ang_vel_w[:] = torch.tensor([0., 0., -.2])
    term._env = env
    term.num_envs = 1
    term.cfg = SimpleNamespace(debug_vis_selected_only=True, velocity_limits=(.2, .2, .4))
    term.velocity_command = torch.tensor([[.1, 0., .2]])
    markers, arrows = [], []
    visualizer = SimpleNamespace(env_idx=0, get_env_indices=lambda _: [0],
        add_sphere=lambda p, **kwargs: markers.append(np.array(p).copy()),
        add_arrow=lambda start, end, **kwargs: arrows.append((np.array(start).copy(), np.array(end).copy(), kwargs)))
    SnakeVelocityCommand._debug_vis_impl(cast(SnakeVelocityCommand, term), cast(DebugVisualizer, visualizer))
    origin = data.site_pos_w[0, 0] if task == 2 else data.body_link_pos_w[0, 0]
    assert len(markers) == 1 and len(arrows) == 4
    np.testing.assert_allclose(markers[0], origin)
    for start, _, _ in arrows:
        np.testing.assert_allclose(start, origin)
    command_delta = arrows[0][1]-arrows[0][0]
    actual_delta = arrows[2][1]-arrows[2][0]
    assert (command_delta*actual_delta).sum() < 0
    assert arrows[1][1][2] > origin[2] and arrows[3][1][2] < origin[2]
    assert arrows[0][2]['color'] != arrows[2][2]['color']
    assert arrows[0][2]['width'] > arrows[2][2]['width']
