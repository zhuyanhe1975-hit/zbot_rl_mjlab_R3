"""Guided uses the requested physical reference, independently of foot articulation."""
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from zbot_rl_mjlab.tasks.universal import omni


def environment(tasks=(0, 1, 2, 3), device='cpu', backend='eager'):
    n = len(tasks)
    q = torch.tensor([2**-.5, 0., -2**-.5, 0.], device=device).expand(n, 1, 4).clone()
    linear = torch.tensor([1., 2., 9.], device=device).expand(n, 1, 3).clone()
    angular = torch.tensor([0., 0., .3], device=device).expand(n, 1, 3).clone()
    data = SimpleNamespace(body_link_quat_w=q, site_quat_w=q.clone(),
                           body_link_lin_vel_w=linear, site_lin_vel_w=linear * 10,
                           body_link_ang_vel_w=angular, site_ang_vel_w=angular * 10,
                           body_com_lin_vel_w=torch.full_like(linear, 999.))
    term = SimpleNamespace(current_task=torch.tensor(tasks, device=device),
                           standing_mode=torch.tensor(tasks, device=device) == 0,
                           heading=torch.zeros(n, 2, device=device),
                           initial_heading=torch.zeros(n, 2, device=device),
                           heading_initialized=torch.zeros(n, dtype=torch.bool, device=device))
    env = SimpleNamespace(cfg=SimpleNamespace(guided_local_reference=True, omni_frame_backend=backend),
                          scene={'robot': SimpleNamespace(data=data, find_bodies=lambda _: ([0], ['base']),
                                                          find_sites=lambda _: ([0], ['foot_0_sole']))},
                          command_manager=SimpleNamespace(get_term=lambda _: term))
    return env, term, data


def test_base_forward_and_head_forward_are_distinct_and_transformations_use_base():
    env, _, _ = environment()
    torch.testing.assert_close(omni.site_heading(env), torch.tensor([[0., 1.], [0., 1.], [1., 0.], [0., 1.]]))


def test_head_articulation_cannot_rotate_biped_reference_and_base_cannot_rotate_snake_reference():
    env, _, data = environment()
    baseline = omni.site_heading(env).clone()
    yaw = torch.tensor([2**-.5, 0., 0., 2**-.5]).expand(4, -1)
    data.site_quat_w[:] = quat_mul(yaw, data.site_quat_w[:, 0])[:, None]
    moved = omni.site_heading(env).clone()
    torch.testing.assert_close(moved[[0, 1, 3]], baseline[[0, 1, 3]])
    torch.testing.assert_close(moved[2], torch.tensor([0., 1.]), atol=1e-6, rtol=1e-6)
    data.body_link_quat_w[:] = quat_mul(yaw, data.body_link_quat_w[:, 0])[:, None]
    torch.testing.assert_close(omni.site_heading(env)[2], moved[2])


def test_velocity_and_yaw_use_selected_point_including_rewards_not_com():
    env, _, _ = environment()
    expected = torch.tensor([[2., -1., .3], [2., -1., .3], [10., 20., 3.], [2., -1., .3]])
    torch.testing.assert_close(omni.site_velocity(env), expected)
    torch.testing.assert_close(omni.tracking_velocity(env), expected)


def test_mode_changes_select_current_task_not_latched_standing_mode():
    env, term, _ = environment((0,))
    torch.testing.assert_close(omni.site_heading(env), torch.tensor([[0., 1.]]))
    term.current_task.fill_(2)
    torch.testing.assert_close(omni.site_heading(env), torch.tensor([[1., 0.]]))
    term.current_task.fill_(3)
    torch.testing.assert_close(omni.site_heading(env), torch.tensor([[0., 1.]]))


def test_rigid_world_yaw_rotates_reference_and_preserves_local_velocity():
    env, _, data = environment()
    before = omni.site_velocity(env).clone()
    yaw = torch.tensor([[.7, 0., 0., 0.]])
    yaw[:, 0] = torch.cos(torch.tensor(.7 / 2))
    yaw[:, 3] = torch.sin(torch.tensor(.7 / 2))
    for name in ('body_link_quat_w', 'site_quat_w'):
        setattr(data, name, quat_mul(yaw.expand(4, -1), getattr(data, name)[:, 0])[:, None])
    for name in ('body_link_lin_vel_w', 'site_lin_vel_w', 'body_link_ang_vel_w', 'site_ang_vel_w'):
        setattr(data, name, quat_apply(yaw.expand(4, -1), getattr(data, name)[:, 0])[:, None])
    torch.testing.assert_close(omni.site_velocity(env), before)


def test_vertical_axes_keep_previous_heading_and_first_use_has_finite_fallback():
    env, term, data = environment((0, 2))
    previous = omni.site_heading(env).clone()
    data.body_link_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    data.site_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    torch.testing.assert_close(omni.site_heading(env), previous)
    term.heading_initialized[:] = False
    torch.testing.assert_close(omni.site_heading(env), torch.tensor([[1., 0.], [1., 0.]]))


def test_packed_kernel_is_device_only_and_matches_public_eager_path():
    env, term, data = environment()
    packed = torch.cat((data.body_link_quat_w[:, 0], data.site_quat_w[:, 0],
                        (term.current_task == 2)[:, None].float(), term.heading,
                        term.heading_initialized[:, None].float(), term.initial_heading), -1)
    result = omni.packed_guided_frame_step(packed)
    assert result.shape == (4, 5)
    torch.testing.assert_close(result[:, :2], omni.site_heading(env))
    assert result[:, 4].bool().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_graph_matches_eager_and_tracks_mode_changes():
    eager, eager_term, _ = environment(device='cuda')
    graph, graph_term, _ = environment(device='cuda', backend='cuda_graph')
    for tasks in ((0, 1, 2, 3), (2, 2, 0, 1)):
        eager_term.current_task[:] = torch.tensor(tasks, device='cuda')
        graph_term.current_task[:] = eager_term.current_task
        torch.testing.assert_close(omni.site_velocity(graph), omni.site_velocity(eager))
