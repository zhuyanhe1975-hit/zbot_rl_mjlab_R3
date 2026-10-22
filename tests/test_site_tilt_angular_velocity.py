"""Tilt-rate shaping must damp rocking without penalizing commanded yaw."""
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.snake.site_omni import site_tilt_angular_velocity_penalty

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'


def environment(angular, gravity):
    angular = torch.tensor(angular, dtype=torch.float64)
    velocities = torch.stack((torch.full_like(angular, 99.), angular), dim=1)
    return SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(mujoco=SimpleNamespace(gravity=gravity))),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(site_ang_vel_w=velocities))},
        _omni_motion_site_index=1,
    )


def test_roll_pitch_cost_uses_selected_site_and_excludes_yaw():
    env = environment([[0., 0., 5.], [2., 0., 0.], [0., -3., 0.], [2., -3., 5.]], (0., 0., -9.81))
    torch.testing.assert_close(site_tilt_angular_velocity_penalty(env), torch.tensor([0., 4., 9., 13.], dtype=torch.float64))


def test_yaw_exclusion_follows_gravity_instead_of_a_fixed_body_axis():
    env = environment([[0., 5., 0.], [2., 5., 3.]], (0., -9.81, 0.))
    torch.testing.assert_close(site_tilt_angular_velocity_penalty(env), torch.tensor([0., 13.], dtype=torch.float64))


@pytest.mark.parametrize('play', [False, True])
def test_card_enables_tilt_rate_cost_only_for_requested_experiment(play):
    cfg = load_env_cfg(TASK, play=play)
    term = cfg.rewards['site_tilt_angular_velocity']
    assert term.func is site_tilt_angular_velocity_penalty
    assert term.weight == -.05
    assert cfg.rewards['site_tilt'].weight == -1.
    for task in ('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher', 'Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher'):
        assert 'site_tilt_angular_velocity' not in load_env_cfg(task, play=play).rewards
