"""Real Warp loop constraints, passive capture, release and scoped resets."""
import math

import mujoco
import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.robot import get_spec
from zbot_rl_mjlab.tasks.reconfiguration.basic import mdp
from zbot_rl_mjlab.tasks.reconfiguration.basic.model import head_tail_spec

TASK = 'Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Basic'
# Inverse-kinematics witness only; production resets always use a straight chain.
CLOSED_Q = (-2.44898034, 1.5892, -1.28241865, 1.7971567, -2.0315471, .93572539)


def test_new_category_limits_and_unchanged_original_asset():
    original = get_spec().compile()
    model = head_tail_spec().compile()
    assert original.neq == 0 and model.neq == 4
    assert model.nq == original.nq == 13
    assert not model.eq_active0.any()
    np.testing.assert_allclose(model.jnt_range[1:], np.tile([-math.pi, math.pi], (6, 1)))
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        assert cfg.scene.num_envs == (1 if play else 8192)
        assert cfg.actions['joint_pos'].target_limit == 2.
        assert not cfg.commands and not cfg.magnetic_model_calibrated
        assert all(v == 0 for v in cfg.scene.entities['robot'].init_state.joint_pos.values())
        assert ('success' in cfg.terminations) == (not play)


@pytest.fixture(scope='module')
def env():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required for real batched equality verification')
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 3
    cfg.sim.mujoco.gravity = (0., 0., 0.)
    with torch.inference_mode():
        instance = ManagerBasedRlEnv(cfg, device='cuda:0')
        yield instance
        instance.close()


def place_closed(env, ids=(0, 1)):
    s = mdp.state(env)
    ids = torch.tensor(ids, device=env.device)
    q = torch.tensor(CLOSED_Q, device=env.device).expand(len(ids), -1)
    root = s.robot.data.default_root_state[ids].clone()
    root[:, :3] = env.scene.env_origins[ids] + root.new_tensor((0., 0., 1.))
    root[:, 3:7] = root.new_tensor((1., 0., 0., 0.))
    root[:, 7:] = 0.
    s.robot.write_root_state_to_sim(root, env_ids=ids)
    s.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    targets = s.robot.data.default_joint_pos.clone()
    targets[ids] = q
    s.robot.set_joint_position_target(targets)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    return s


def test_straight_reset_is_prone_open_and_observable(env):
    obs, _ = env.reset()
    s = mdp.state(env)
    assert obs['actor'].shape == (3, 59)
    assert torch.isfinite(obs['actor']).all()
    assert not s.enabled().any()
    torch.testing.assert_close(s.geometry()['distance'], torch.full((3,), .66, device=env.device), atol=1e-5, rtol=0)
    torch.testing.assert_close(mdp.alignment_reward(env), torch.zeros(3, device=env.device), atol=1e-6, rtol=0)
    assert (s.robot.data.body_com_pos_w[..., 2] < .12).all()
    env.step(torch.zeros(3, 6, device=env.device))
    assert not s.enabled().any()
    assert not s.previous_wrench.any()


def test_capture_is_dwell_gated_without_teleport_and_adds_real_loop_rows(env):
    env.reset()
    s = place_closed(env)
    assert (s.geometry()['angle'][:2] < .001).all()
    before = env.sim.data.qpos[:].clone()
    for _ in range(5):
        s.substep()
    assert not s.enabled().any()
    s.substep()
    torch.testing.assert_close(env.sim.data.qpos[:], before, rtol=0, atol=0)
    assert s.enabled().sum(-1).tolist() == [1, 1, 0]
    env.sim.forward()
    data = env.sim.data
    rows = torch.arange(data.efc.force.shape[1], device=env.device)[None]
    equality_rows = ((data.efc.type[:] == 0) & (rows < data.nefc[:, None])).sum(-1)
    assert equality_rows.tolist() == [6, 6, 0]
    for _ in range(105):
        s.substep()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.physics_dt)
    assert s.enabled().sum(-1).tolist() == [1, 1, 0]
    assert s.succeeded.tolist() == [True, True, False]
    assert (s.geometry()['distance'][:2] < .001).all()
    assert s.just_succeeded[:2].all()
    s.begin_control_step()
    s.substep()
    assert not s.just_succeeded.any()  # one-shot success, no repeated bonus
    # Reset selected world only; the second world's physical loop persists.
    env._reset_idx(torch.tensor([0], device=env.device))
    assert s.enabled().sum(-1).tolist() == [0, 1, 0]
    assert s.connections.tolist() == [0, 1, 0]
    assert s.succeeded.tolist() == [False, True, False]


def test_real_weld_load_breaks_passively_and_cooldown_blocks_recapture(env):
    env.reset()
    s = place_closed(env, (0,))
    for _ in range(6): s.substep()
    env.sim.forward()
    previous = env.cfg.break_force
    env.cfg.break_force = 1e-8  # measured nonzero constraint reaction exceeds test threshold
    try:
        s.substep()
        assert not s.enabled().any()
        assert s.breaks.tolist() == [1, 0, 0]
        assert s.cooldown[0] > 0
        for _ in range(6): s.substep()
        assert not s.enabled().any()
    finally:
        env.cfg.break_force = previous


def test_attraction_conserves_pair_wrench_and_does_not_accumulate(env):
    env.reset()
    s = place_closed(env, (0,))
    g = s.geometry()
    # Finite-range test geometry: front-facing and 1 cm apart.
    g['delta'] = torch.zeros_like(g['delta'])
    g['delta'][:, 2] = -.01
    g['p1'] = g['p0'] + g['delta']
    g['distance'] = g['delta'].norm(dim=-1)
    g['q0'] = g['q1'] = g['qbest'] = g['q0'].new_tensor((1., 0., 0., 0.)).expand(3, -1)
    g['dv'] = torch.ones_like(g['delta']) * .1
    g['dw'] = torch.ones_like(g['delta']) * .1
    active = torch.zeros(3, dtype=torch.bool, device=env.device)
    s.apply_attraction(g, active)
    first = env.sim.data.xfrc_applied[:].clone()
    s.apply_attraction(g, active)
    torch.testing.assert_close(env.sim.data.xfrc_applied[:], first)
    wrench = s.previous_wrench
    assert wrench.abs().sum() > 0
    torch.testing.assert_close(wrench[:, :, :3].sum(1), torch.zeros(3, 3, device=env.device))
    com = s.robot.data.body_com_pos_w[:, s.bodies]
    moment = torch.cross(com - com[:, :1], wrench[:, :, :3], dim=-1) + wrench[:, :, 3:]
    torch.testing.assert_close(moment.sum(1), torch.zeros(3, 3, device=env.device), atol=1e-6, rtol=0)
    g['distance'][:] = 1.
    s.apply_attraction(g, active)
    assert not s.previous_wrench.any()
    assert not env.sim.data.xfrc_applied[:].any()


def test_invalid_magnetic_assumptions_fail_early(env):
    s = mdp.state(env)
    previous = env.cfg.capture_distance
    env.cfg.capture_distance = env.cfg.release_distance
    try:
        with pytest.raises(ValueError, match='capture_distance'):
            s.validate()
    finally:
        env.cfg.capture_distance = previous
