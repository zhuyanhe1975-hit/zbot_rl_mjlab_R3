"""Actual wz shaping penalizes turning in either direction without charging roll/pitch."""
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.snake.site_omni import site_yaw_velocity_penalty

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'


def test_actual_site_wz_cost_is_symmetric_and_excludes_roll_pitch():
    angular = torch.tensor([[2., -3., 0.], [0., 0., 2.], [0., 0., -2.]])
    env = SimpleNamespace(
        scene={'robot': SimpleNamespace(data=SimpleNamespace(
            site_ang_vel_w=torch.stack((torch.full_like(angular, 99.), angular), dim=1)))},
        _omni_motion_site_index=1,
    )
    torch.testing.assert_close(site_yaw_velocity_penalty(env), torch.tensor([0., 4., 4.]))


@pytest.mark.parametrize('play', [False, True])
def test_card_enables_wz_cost_only_for_the_requested_task(play):
    term = load_env_cfg(TASK, play=play).rewards['wz_velocity']
    assert term.func is site_yaw_velocity_penalty
    assert term.weight == -.2  # Preserve the card's existing tuned value.
    assert term.params == {}
    for other in ('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher', 'Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher'):
        assert 'wz_velocity' not in load_env_cfg(other, play=play).rewards
