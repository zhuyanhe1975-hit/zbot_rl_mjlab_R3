"""Omni commands use the same signed, gravity-aligned site frame in both modes."""
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.tasks.universal import omni


def fixture(n=4096, training=True):
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=1)
    term = omni.OmniCommandCfg(resampling_time_range=(6., 12.), training=training).build(env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    data = SimpleNamespace(site_quat_w=torch.tensor([1., 0, 0, 0]).expand(n, 1, 4).clone())
    env.scene = {'robot': SimpleNamespace(data=data, find_sites=lambda name: ([0], [name]))}
    return env, term


@pytest.mark.parametrize('standing', [True, False])
@pytest.mark.parametrize('yaw', [0., .7, -2.1])
def test_frame_rotates_with_world_yaw(standing, yaw):
    env, term = fixture(3)
    term.command[:, 0] = 1 if standing else -1
    yaw_q = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    pose = torch.tensor([1., 0, 0, 0] if standing else [2**-.5, 0, -(2**-.5), 0])
    env.scene['robot'].data.site_quat_w[:, 0] = quat_mul(yaw_q, pose)
    actual = omni.site_heading(env)
    torch.testing.assert_close(actual, torch.tensor([math.cos(yaw), math.sin(yaw)]).expand(3, 2))


def test_vertical_primary_uses_rotated_other_axis_then_retains_memory():
    env, term = fixture(2)
    yaw = .8
    q = quat_mul(torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]), torch.tensor([2**-.5, 0, -(2**-.5), 0]))
    env.scene['robot'].data.site_quat_w[:, 0] = q
    expected = torch.tensor([math.cos(yaw), math.sin(yaw)]).expand(2, 2)
    torch.testing.assert_close(omni.site_heading(env), expected)
    env.scene['robot'].data.site_quat_w[:] = float('nan')
    torch.testing.assert_close(omni.site_heading(env), expected)
    assert term.heading_initialized.all()


@pytest.mark.parametrize('mode', [1., -1.])
def test_every_signed_axis_stop_and_mixed_exist_from_iteration_zero(mode):
    env, term = fixture()
    term.command[:, 0] = mode
    term._sample_velocity(torch.arange(env.num_envs))
    requested = term.requested_velocity
    active = requested.abs() > 1e-6
    assert (active.sum(-1) == 0).any()
    assert (active.sum(-1) == 3).any()
    for axis in range(3):
        single = active.sum(-1) == 1
        assert (single & (requested[:, axis] > 0)).sum() > 80
        assert (single & (requested[:, axis] < 0)).sum() > 80
    assert (requested.abs() <= requested.new_tensor(term.cfg.velocity_limits)).all()


def test_manual_commands_survive_sampling_and_ema_unlocks_limits():
    env, term = fixture()
    term.set_velocity(-.2, .15, -.7, env_ids=torch.tensor([0]))
    term._sample_velocity(torch.arange(env.num_envs))
    torch.testing.assert_close(term.requested_velocity[0], torch.tensor([-.2, .15, -.7]))
    low = term.requested_velocity[1:].abs().amax(0)
    term.tracking_ema[:] = 1.
    term._sample_velocity(torch.arange(env.num_envs))
    assert (term.requested_velocity[1:].abs().amax(0) > low * 2).all()


def test_explicit_reset_partial_resets_manual_but_preserves_other_environments(monkeypatch):
    env, term = fixture(8)
    class Scene(dict):
        env_origins: torch.Tensor
    env.scene = Scene(env.scene)
    env.scene.env_origins = torch.zeros(8, 3)
    env.cfg = SimpleNamespace(commands={'transform': term.cfg}, standing_joint_pos=(.3,) * 6)
    data = env.scene['robot'].data
    data.default_root_state = torch.zeros(8, 13)
    data.default_root_state[:, 3] = 1
    data.joint_pos = torch.full((8, 6), 99.)
    data.root_state = torch.full((8, 13), 99.)
    robot = env.scene['robot']
    robot.write_joint_state_to_sim = lambda q, qd, env_ids: data.joint_pos.__setitem__(env_ids, q)
    robot.write_root_state_to_sim = lambda root, env_ids: data.root_state.__setitem__(env_ids, root)
    monkeypatch.setattr(omni, 'collision_radial_envelope', lambda pose: ((), -.3, .3))
    term.manual_control[3] = True
    omni.reset_omni(env, torch.tensor([1, 3, 5]))
    assert (data.joint_pos[[0, 2, 4, 6, 7]] == 99.).all()
    assert (data.joint_pos[[1, 3, 5]] != 99.).all()
    assert (data.root_state[[1, 5], 7:] == 0).all()
    assert not torch.allclose(data.root_state[1, 3:7], data.root_state[5, 3:7])


def test_tracking_uses_com_not_head_translation_and_site_world_yaw(monkeypatch):
    env, _ = fixture(2)
    data = env.scene['robot'].data
    data.body_com_lin_vel_w = torch.tensor([[[2., 0., 0.], [0., 0., 0.]]]).expand(2, -1, -1)
    data.site_lin_vel_w = torch.tensor([[[9., 8., 0.]]]).expand(2, -1, -1)
    data.site_ang_vel_w = torch.tensor([[[5., 6., -.7]]]).expand(2, -1, -1)
    monkeypatch.setattr(omni.transform, 'body_masses', lambda env: torch.tensor([[1., 3.]]))
    torch.testing.assert_close(omni.tracking_velocity(env), torch.tensor([[.5, 0., -.7]]).expand(2, -1))
    torch.testing.assert_close(omni.site_velocity(env), torch.tensor([[9., 8., -.7]]).expand(2, -1))


def reference_frame_step(packed):
    """Independent scalar reference, including degeneracy and NaN handling."""
    from mjlab.utils.lab_api.math import quat_apply
    rows = []
    for row in packed:
        q = row[:4]
        x = quat_apply(q, row.new_tensor([1., 0., 0.]))[:2]
        minus_z = quat_apply(q, row.new_tensor([0., 0., -1.]))[:2]
        primary, other = (x, minus_z) if row[4] > 0 else (minus_z, x)
        heading, initial, initialized = row[5:7], row[8:10], bool(row[7] > 0)
        valid = bool(torch.isfinite(primary).all() and primary.norm() > .15)
        fallback = not initialized and not valid and bool(torch.isfinite(other).all() and other.norm() > .15)
        if valid or fallback:
            chosen = primary if valid else other
            heading = chosen / chosen.norm()
            if not initialized:
                initial = heading
            initialized = True
        rows.append(torch.cat((heading, initial, row.new_tensor([float(initialized)]))))
    return torch.stack(rows)


def test_packed_frame_step_matches_original_eager_semantics():
    torch.manual_seed(71)
    packed = torch.randn(128, 10)
    packed[:, :4] /= packed[:, :4].norm(dim=-1, keepdim=True)
    packed[:8, :4] = torch.tensor([2**-.5, 0., -(2**-.5), 0.])
    packed[8, :4] = float('nan')
    packed[9, :4] = float('inf')
    packed[:, 7] = (packed[:, 7] > 0).float()
    expected = reference_frame_step(packed)
    torch.testing.assert_close(omni.packed_frame_step(packed), expected)


def test_unknown_frame_backend_rejected():
    env, _ = fixture(2)
    env.cfg = SimpleNamespace(omni_frame_backend='unknown')
    with pytest.raises(ValueError, match='omni_frame_backend'):
        omni.site_heading(env)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_frame_replay_owns_outputs_and_matches_eager():
    from zbot_rl_mjlab.cuda_graph import FixedTensorGraph
    packed = torch.randn(64, 10, device='cuda')
    packed[:, :4] /= packed[:, :4].norm(dim=-1, keepdim=True)
    packed[:, 7] = (packed[:, 7] > 0).float()
    graph = FixedTensorGraph(omni.packed_frame_step, packed)
    output = graph(packed)
    original = output.clone()
    torch.testing.assert_close(output, omni.packed_frame_step(packed))
    following = packed.clone()
    following[:, 5:7] = 0
    following[:, 8:10] = 1
    graph(following)
    torch.testing.assert_close(output, original)


def test_frame_wrapper_preserves_heading_storage_and_rejects_cpu_graph():
    env, term = fixture(2)
    storage = term.heading.data_ptr()
    assert omni.site_heading(env) is term.heading
    assert term.heading.data_ptr() == storage
    env.cfg = SimpleNamespace(omni_frame_backend='cuda_graph')
    with pytest.raises(ValueError, match='CUDA tensor'):
        omni.site_heading(env)
