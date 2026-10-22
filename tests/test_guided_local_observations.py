"""Current teacher uses the selected body's clean local IMU and reference point."""
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from test_guided_reference_frames import environment

from zbot_rl_mjlab.tasks.universal import guided_observations as observations
from zbot_rl_mjlab.teachers import TeacherEnsemble


def test_local_gravity_gyro_and_height_follow_base_or_foot_site(monkeypatch):
    env, _, data = environment()
    env.num_envs = 4
    robot = env.scene['robot']
    # Match the real Scene mapping and origin access without a full simulator.
    class Scene(dict):
        env_origins = torch.zeros(4, 3)
    env.scene = Scene(robot=robot)
    data.body_link_pos_w = torch.tensor([[[0., 0., .3]]]).expand(4, 1, 3).clone()
    data.site_pos_w = torch.tensor([[[0., 0., .05]]]).expand(4, 1, 3).clone()
    baseline = torch.arange(56).float().expand(4, 56).clone()
    monkeypatch.setattr(observations, 'guided_privileged_state', lambda _: baseline.clone())
    actual = observations.local_reference_privileged_state(env)
    quaternion, linear, angular = observations.guided_reference_state(env)
    gravity = torch.tensor([[0., 0., -1.]]).expand(4, 3)
    torch.testing.assert_close(actual[:, 28:31], quat_apply_inverse(quaternion, gravity))
    torch.testing.assert_close(actual[:, 31:34], quat_apply_inverse(quaternion, angular))
    torch.testing.assert_close(actual[:, 27], linear[:, 2])
    torch.testing.assert_close(actual[:, 34], torch.tensor([.3, .3, .05, .3]))
    torch.testing.assert_close(actual[:, :27], baseline[:, :27])
    torch.testing.assert_close(actual[:, 35:], baseline[:, 35:])


def test_legacy_playback_returns_the_original_observation_without_frame_reads(monkeypatch):
    baseline = torch.randn(2, 56)
    monkeypatch.setattr(observations, 'guided_privileged_state', lambda _: baseline)
    env = SimpleNamespace(cfg=SimpleNamespace(guided_local_reference=False))
    assert observations.local_reference_privileged_state(env) is baseline


def test_clean_single_frame_teacher_configuration():
    cfg = cast(observations.GuidedTeacherEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided'))
    assert cfg.guided_local_reference
    group = cfg.observations['actor']
    assert not group.enable_corruption
    assert not getattr(group, 'history_length', None)
    assert group.terms['privileged'].noise is None
    assert group.terms['privileged'].func is observations.local_reference_privileged_state


@pytest.mark.parametrize('modern,expected', [(False, .3), (True, -.3)])
def test_old_walking_expert_receives_its_downward_yaw_convention(monkeypatch, modern, expected):
    from zbot_rl_mjlab import teachers
    env, term, _ = environment((0,))
    env.cfg.guided_local_reference = modern
    term.heading[:] = torch.tensor([1., 0.])
    term.velocity_command = torch.tensor([[.1, .05, .3]])
    monkeypatch.setattr(teachers.walking_mdp, '_forward_world', lambda _: torch.tensor([[1., 0., 0.]]))
    ensemble = TeacherEnsemble.__new__(TeacherEnsemble)
    ensemble.env = env
    torch.testing.assert_close(ensemble._walking_velocity_command(), torch.tensor([[.1, .05, expected]]))
