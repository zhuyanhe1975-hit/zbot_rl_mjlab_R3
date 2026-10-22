"""Head-frame errors scaled by command bounds and head-local gravity; no heading target."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.experiments.site_snake import (
    site_snake_env_cfg as _recipe_site_snake_env_cfg,
)
from zbot_rl_mjlab.tasks.snake import site_omni as task
from zbot_rl_mjlab.tasks.universal.observation_layout import (
    SITE_SNAKE_OBS_DIM,
    teacher_motion_fields,
)


def fixture(monkeypatch, state):
    cfg = _recipe_site_snake_env_cfg()
    cfg.commands['transform'].velocity_limits = (.2, .2, .5)
    quat = state.new_tensor([2**-.5, 0., -2**-.5, 0.]).expand(len(state), 1, 4).clone()
    data = SimpleNamespace(site_quat_w=quat, body_link_quat_w=torch.full_like(quat, float('nan')))
    term = SimpleNamespace(cfg=cfg.commands['transform'])
    env = SimpleNamespace(cfg=cfg, scene={'robot': SimpleNamespace(data=data)}, _omni_motion_site_index=0,
                          command_manager=SimpleNamespace(get_term=lambda _: term))
    monkeypatch.setattr(task, 'privileged_state', lambda _: state.clone())
    return env


def test_site_observation_adds_features_and_preserves_existing_fields(monkeypatch):
    state = torch.arange(112, dtype=torch.float32).reshape(2, 56)
    state[:, 46] = torch.tensor([.7, -2.1])
    state[:, 40:43] = torch.tensor([[.1, 0., .25], [0., -.025, 0.]])
    state[:, 50:53] = torch.tensor([[.05, .025, .5], [-.025, -.05, -.25]])
    env = fixture(monkeypatch, state)
    result = task.site_snake_state(env)
    assert result.shape == (2, SITE_SNAKE_OBS_DIM) == (2, 61)
    torch.testing.assert_close(result[:, :46], state[:, :46])
    torch.testing.assert_close(result[:, 46:55], state[:, 47:56])
    torch.testing.assert_close(result[:, 55:58], torch.tensor([[.25, -.125, -.5], [.125, .125, .5]]))
    torch.testing.assert_close(result[:, 58:61], torch.tensor([[-1., 0., 0.]]).expand(2, -1), atol=1e-6, rtol=0)
    target, actual, requested = teacher_motion_fields(result)
    torch.testing.assert_close(target, state[:, 40:43])
    torch.testing.assert_close(actual, state[:, 50:53])
    torch.testing.assert_close(requested, state[:, 53:56])


def test_velocity_error_uses_each_axis_command_bound_even_at_zero_command(monkeypatch):
    state = torch.zeros(1, 56)
    state[:, 50:53] = torch.tensor([.05, -.025, .5])
    env = fixture(monkeypatch, state)
    env.cfg.commands['transform'].velocity_limits = (.1, .05, 1.)
    result = task.site_snake_state(env)
    torch.testing.assert_close(result[:, 55:58], torch.tensor([[-.5, .5, -.5]]))


def test_removed_heading_slot_cannot_affect_the_observation(monkeypatch):
    state = torch.zeros(2, 56)
    state[:, 46] = torch.tensor([math.pi-1e-5, -math.pi+1e-5])
    env = fixture(monkeypatch, state)
    result = task.site_snake_state(env)
    torch.testing.assert_close(result[0], result[1])


@pytest.mark.parametrize('yaw', [0., .7, -2.1])
def test_head_gravity_uses_site_orientation_and_is_world_yaw_invariant(monkeypatch, yaw):
    angles = torch.deg2rad(torch.tensor([0., 30., 60., 90.]))
    env = fixture(monkeypatch, torch.zeros(4, 56))
    rotation = torch.zeros(4, 4)
    rotation[:, 0], rotation[:, 2] = (angles/2).cos(), (angles/2).sin()
    canonical = rotation.new_tensor([2**-.5, 0., -2**-.5, 0.]).expand_as(rotation)
    yaw_quat = rotation.new_tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)]).expand_as(rotation)
    env.scene['robot'].data.site_quat_w[:, 0] = quat_mul(yaw_quat, quat_mul(rotation, canonical))
    gravity = task.site_snake_state(env)[:, 58:61]
    torch.testing.assert_close(gravity, torch.stack((-angles.cos(), torch.zeros(4), -angles.sin()), -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(gravity.norm(dim=-1), torch.ones(4), atol=1e-6, rtol=0)
    env.cfg.sim.mujoco.gravity = (0., 0., -1.)
    torch.testing.assert_close(task.site_snake_state(env)[:, 58:61], gravity, atol=1e-6, rtol=0)


def test_new_observation_sanitizes_nonfinite_features(monkeypatch):
    state = torch.zeros(2, 56)
    state[0, [0, 46, 50]] = float('nan')
    state[1, [1, 46, 51]] = float('inf')
    env = fixture(monkeypatch, state)
    env.scene['robot'].data.site_quat_w[:] = float('nan')
    observed = task.site_snake_state(env)
    assert torch.isfinite(observed).all()
    assert not observed[:, 58:61].any()


def test_site_observation_rejects_legacy_one_hot_encoding(monkeypatch):
    env = fixture(monkeypatch, torch.zeros(2, 59))
    with pytest.raises(ValueError, match='scalar task encoding'):
        task.site_snake_state(env)
