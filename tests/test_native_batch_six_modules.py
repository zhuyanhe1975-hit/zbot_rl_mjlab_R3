"""Run with ./scripts/python_mjbatch.sh -m pytest -q tests/test_native_batch_six_modules.py."""
import mujoco
import math
import numpy as np
import pytest

pytest.importorskip('mjbatch')

from zbot_mjbatch.native_batch import NativeAdhesionBatch, make_native_batch_model


@pytest.mark.parametrize('options, expected', [({}, 35.), ({'contact_force_n': 5.}, 5.)])
def test_magnet_contact_capacity_and_explicit_override(options, expected):
    model, data = make_native_batch_model(**options)
    for face in range(6):
        for magnet in range(4):
            source = model.geom(f'physical_connector_{face}_adhesion_source_{magnet}').id
            target = model.geom(f'physical_connector_{face}_adhesion_target_{magnet}').id
            assert model.geom_adhesion[source] == pytest.approx(expected)
            assert model.geom_adhesion[target] == 0.
    contacts = [c for c in data.contact if c.adhesion > 0.]
    assert len(contacts) == 20  # five connected faces, four contacts each
    assert all(c.adhesion == pytest.approx(expected) for c in contacts)


def test_four_magnets_per_side_are_37mm_from_connector_center():
    model, data = make_native_batch_model()
    for face in range(6):
        for side in ('source', 'target'):
            sites = [model.site(f'physical_connector_{face}_magnet_{side}_{i}').id for i in range(4)]
            # Equally spaced centers have their mean at the connector center,
            # independent of the module's initial rigid-body orientation.
            positions = data.site_xpos[sites]
            np.testing.assert_allclose(np.linalg.norm(positions-positions.mean(axis=0), axis=1), .037, atol=1e-12)


@pytest.mark.parametrize('quarter', range(4))
def test_each_registration_retain_four_magnetic_pairs(quarter):
    model, _ = make_native_batch_model()
    for face in range(6):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        target = (face + 1) % 6
        inlet = model.site(f'module_{target}/port_in').id
        axis = data.site_xmat[inlet].reshape(3, 3)[:, 2].copy()
        qadr = int(model.jnt_qposadr[model.joint(f'module_{target}/module_root').id])
        rotation = np.empty(4)
        result = np.empty(4)
        mujoco.mju_axisAngle2Quat(rotation, axis, quarter * math.pi / 2)
        mujoco.mju_mulQuat(result, rotation, data.qpos[qadr+3:qadr+7].copy())
        data.qpos[qadr+3:qadr+7] = result
        mujoco.mj_forward(model, data)
        source = {model.geom(f'physical_connector_{face}_adhesion_source_{i}').id for i in range(4)}
        target_geoms = {model.geom(f'physical_connector_{face}_adhesion_target_{i}').id for i in range(4)}
        contacts = [c for c in data.contact if
                    ({int(c.geom1), int(c.geom2)} & source) and
                    ({int(c.geom1), int(c.geom2)} & target_geoms)]
        assert len(contacts) == 4
        assert all(c.adhesion == pytest.approx(35.) for c in contacts)


def test_six_module_batch_matches_independent_serial_rollouts():
    env = NativeAdhesionBatch(num_envs=4, num_threads=2)
    m = env.model
    assert m.neq == 0
    assert m.nu == 6
    assert sum(m.geom_adhesion > 0) == 24
    initial_data = mujoco.MjData(m)
    mujoco.mj_setState(m, initial_data, env.initial_state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, initial_data)
    assert np.linalg.norm(initial_data.qfrc_adhesion) > 0
    np.testing.assert_allclose(env.qfrc_adhesion[0], initial_data.qfrc_adhesion, rtol=0, atol=1e-12)
    # Use distinct small controls so a broadcast/array-index error is visible.
    env.ctrl[:] += np.arange(4)[:, None] * .005
    serial = []
    for i in range(4):
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, env.state[i], mujoco.mjtState.mjSTATE_INTEGRATION)
        for _ in range(100):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        serial.append(d)
    env.step(nstep=100)
    np.testing.assert_allclose(env.qpos, np.stack([d.qpos for d in serial]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(env.qvel, np.stack([d.qvel for d in serial]), rtol=0, atol=1e-12)
    assert not np.any(env.warning[:, :, 1])
    assert all(not np.any(d.warning.number) for d in serial)
    assert np.isfinite(env.qpos).all() and np.isfinite(env.qvel).all()


def test_partial_reset_restores_selected_world_only():
    env = NativeAdhesionBatch(num_envs=2, num_threads=2)
    initial = env.state.copy()
    env.step(nstep=5)
    other = env.state[1].copy()
    env.reset([0])
    np.testing.assert_allclose(env.state[0], initial[0], rtol=0, atol=0)
    np.testing.assert_allclose(env.state[1], other, rtol=0, atol=0)
