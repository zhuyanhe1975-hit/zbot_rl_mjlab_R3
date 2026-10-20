"""Enable the existing snake command visual for selected site-frame teacher robots."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg


@pytest.mark.parametrize('task', [
    'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided',
    'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher',
])
def test_command_visual_enabled_only_for_play(task):
    play = load_env_cfg(task, play=True).commands['transform']
    train = load_env_cfg(task).commands['transform']
    assert play.debug_vis and not train.debug_vis
    assert isinstance(play, PrivilegedOmniCommandCfg) and play.debug_vis_selected_only


def test_existing_visual_uses_effective_site_frame_and_selected_env_only(monkeypatch):
    positions = torch.tensor([[[1., 2., .05]], [[3., 4., .08]]])
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0,
                          cfg=SimpleNamespace(omni_site_frame=True, motion_site='foot_0_sole'),
                          scene={'robot': SimpleNamespace(data=SimpleNamespace(site_pos_w=positions),
                              find_sites=lambda _: ([0], ['foot_0_sole']))})
    term = PrivilegedOmniCommandCfg(training=False, mode='snake', debug_vis=True,
                                    resampling_time_range=(6., 12.)).build(env)
    term.velocity_command[:] = torch.tensor([[.1, 0., 0.], [.125, .2, -.8]])
    term.requested_velocity.zero_()
    monkeypatch.setattr(mdp, 'heading', lambda env: torch.tensor([[1., 0.], [0., 1.]]))
    arrows, markers = [], []
    visualizer: Any = SimpleNamespace(env_idx=1, get_env_indices=lambda _: [0, 1],
                                 add_sphere=lambda p, **kw: markers.append(p),
                                 add_arrow=lambda start, end, **kw: arrows.append((start, end, kw['color'])))
    term.debug_vis(visualizer)
    assert len(markers) == 1 and len(arrows) == 2
    torch.testing.assert_close(torch.as_tensor(markers[0]), positions[1, 0])
    torch.testing.assert_close(torch.as_tensor(arrows[0][1]), positions[1, 0] + torch.tensor([-.15, .0375, 0.]))
    torch.testing.assert_close(torch.as_tensor(arrows[1][1]), positions[1, 0] + torch.tensor([0., 0., -.15]))
    assert arrows[0][2] == (.2, .4, 1., .9)
    assert arrows[1][2] == (1., .55, .1, .9)
    visualizer.env_idx = 0
    arrows.clear()
    markers.clear()
    term.debug_vis(visualizer)
    assert len(markers) == 1 and len(arrows) == 1
    torch.testing.assert_close(torch.as_tensor(markers[0]), positions[0, 0])
