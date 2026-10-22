"""The velocity frame follows the current snake-head direction without a yaw target."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.experiments.site_snake import (
    site_snake_env_cfg as _recipe_site_snake_env_cfg,
)
from zbot_rl_mjlab.tasks.snake import site_omni as task
from zbot_rl_mjlab.tasks.universal import omni
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommand


def set_yaw(env, yaw):
    yaw = torch.as_tensor(yaw, dtype=torch.float32).expand(env.num_envs)
    rotation = torch.stack(((yaw/2).cos(), torch.zeros_like(yaw),
                            torch.zeros_like(yaw), (yaw/2).sin()), -1)
    env.scene['robot'].data.site_quat_w[:, 0] = quat_mul(
        rotation, rotation.new_tensor([2**-.5, 0., -2**-.5, 0.]).expand_as(rotation))


def fixture(yaw=(.7, -1.2), backend='eager', device='cpu'):
    cfg = _recipe_site_snake_env_cfg(play=True)
    cfg.omni_frame_backend = backend
    env = SimpleNamespace(cfg=cfg, num_envs=len(yaw), device=device,
                          step_dt=.02, common_step_counter=0)
    term = cfg.commands['transform'].build(env)
    term.command[:, 0] = -1
    env.command_manager = SimpleNamespace(get_term=lambda _: term)
    data = SimpleNamespace(site_quat_w=torch.zeros(len(yaw), 1, 4, device=device),
                           site_lin_vel_w=torch.zeros(len(yaw), 1, 3, device=device),
                           site_ang_vel_w=torch.zeros(len(yaw), 1, 3, device=device))
    env.scene = {'robot': SimpleNamespace(data=data, find_sites=lambda _: ([0], ['foot_0_sole']))}
    set_yaw(env, yaw)
    omni.site_heading(env)
    return env, term


@pytest.mark.parametrize('initial_yaw', [0., .7, math.pi-.01, -math.pi+.01])
def test_actual_head_yaw_rotates_velocity_frame_even_without_yaw_command(initial_yaw):
    env, term = fixture((initial_yaw,))
    initial = term.heading.clone()
    term.velocity_command[:, 0] = .1
    env.scene['robot'].data.site_lin_vel_w[:, 0, :2] = .1*initial
    set_yaw(env, (initial_yaw+.8,))
    expected = torch.tensor([[.1*math.cos(.8), -.1*math.sin(.8), 0.]])
    for _ in range(3):
        torch.testing.assert_close(task.mdp.tracking_velocity(env), expected, atol=1e-7, rtol=1e-5)
        assert not hasattr(term, 'reference_heading')
        assert 'heading_deviation' not in env.cfg.rewards
        assert task.mdp.heading_error(env).item() == 0.


@pytest.mark.parametrize('yaw_command', [0., .2, -.2])
def test_yaw_command_does_not_integrate_a_velocity_reference(monkeypatch, yaw_command):
    env, term = fixture()
    monkeypatch.setattr(PrivilegedOmniCommand, '_update_command', lambda self, ids: None)
    initial = term.heading.clone()
    term.requested_velocity[:, 2] = yaw_command
    term.velocity_command[:, 2] = yaw_command
    for step in range(10):
        env.common_step_counter = step
        term._update_command(None)
        torch.testing.assert_close(omni.site_heading(env), initial)
    assert not hasattr(term, 'reference_heading')


def test_partial_reset_retains_actual_head_frame_semantics():
    env, term = fixture()
    previous = term.heading.clone()
    set_yaw(env, (2., -1.2))
    term.reset(torch.tensor([0]))
    actual = omni.site_heading(env)
    torch.testing.assert_close(actual[0], torch.tensor([math.cos(2.), math.sin(2.)]))
    torch.testing.assert_close(actual[1], previous[1])
    assert not hasattr(term, 'reference_heading')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_frame_follows_current_head_orientation():
    env, term = fixture(backend='cuda_graph', device='cuda')
    set_yaw(env, (1.5, .4))
    heading = omni.site_heading(env)
    angles = heading.new_tensor([1.5, .4])
    torch.testing.assert_close(heading, torch.stack((angles.cos(), angles.sin()), -1))
    assert not hasattr(term, 'reference_heading')
