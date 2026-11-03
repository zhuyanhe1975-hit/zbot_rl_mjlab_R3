"""Real-asset aggregation checks; no magnetic calibration or physical safety."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.four_magnet_force import FourMagnetParameters, magnetic_qfrc
from zbot_mjbatch.physical_zbot_geometry import make_physical_zbot_model
from zbot_mjbatch.six_interface_magnets import SixInterfaceMagnetForces


@pytest.fixture(scope='module')
def model():
    return make_physical_zbot_model()


@pytest.fixture
def params():
    # The 5 N strength is an illustrative assumption, not a measurement or
    # a derivation from the historical equivalent 100 N break threshold.
    return FourMagnetParameters(5., .002, .001, .006, .015)


def updated(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def roots(model):
    joints = [model.joint(f'module_{i}/module_root').id for i in range(6)]
    return [(int(model.jnt_bodyid[j]), int(model.jnt_dofadr[j])) for j in joints]


def net_free_root_wrench(model, data, force):
    linear = np.zeros(3); angular = np.zeros(3)
    for body, dof in roots(model):
        f = force[dof:dof+3]
        linear += f
        angular += data.xmat[body].reshape(3, 3) @ force[dof+3:dof+6]
        angular += np.cross(data.xpos[body], f)
    return linear, angular


def test_manager_aggregates_six_interfaces_and_preserves_existing_force(model, params):
    data = updated(model)
    data.qfrc_applied[:] = np.linspace(-2., 2., model.nv)
    data.xfrc_applied[:] = .25
    before_q = data.qfrc_applied.copy(); before_x = data.xfrc_applied.copy()
    manager = SixInterfaceMagnetForces(model, data, (params,)*6)
    reference = sum((magnetic_qfrc(model, data, a, b, params) for a, b in manager.sites),
                    np.zeros(model.nv))
    scratch = np.full(model.nv, 99.)
    actual = manager.compute(out=scratch)
    assert actual is scratch
    np.testing.assert_allclose(actual, reference, rtol=0., atol=1e-12)
    np.testing.assert_array_equal(data.qfrc_applied, before_q)
    np.testing.assert_array_equal(data.xfrc_applied, before_x)
    linear, angular = net_free_root_wrench(model, data, actual)
    np.testing.assert_allclose(linear, 0., atol=1e-10)
    np.testing.assert_allclose(angular, 0., atol=1e-10)
    # Resetting output each call prevents hidden accumulation.
    np.testing.assert_allclose(manager.compute(out=scratch), reference, rtol=0., atol=1e-12)
    with pytest.raises(ValueError):
        manager.compute(out=data.qfrc_applied)
    with pytest.raises(ValueError):
        manager.compute(out=data.qvel)


@pytest.mark.parametrize('quarter', range(4))
def test_each_interface_at_four_clockings_conserves_wrench(model, params, quarter):
    for face in range(6):
        data = updated(model)
        a = np.array([model.site(f'physical_connector_{face}_magnet_source_{i}').id for i in range(4)])
        target = (face+1) % 6
        pivot = data.site_xpos[a].mean(axis=0)
        site = model.site(f'module_{target}/port_in').id
        axis = data.site_xmat[site].reshape(3, 3)[:, 2].copy()
        joint = model.joint(f'module_{target}/module_root').id
        adr = int(model.jnt_qposadr[joint])
        quaternion = np.empty(4); transformed = np.empty(4); matrix = np.empty(9)
        mujoco.mju_axisAngle2Quat(quaternion, axis, quarter*math.pi/2)
        mujoco.mju_quat2Mat(matrix, quaternion)
        data.qpos[adr:adr+3] = pivot + matrix.reshape(3, 3) @ (data.qpos[adr:adr+3] - pivot)
        mujoco.mju_mulQuat(transformed, quaternion, data.qpos[adr+3:adr+7].copy())
        data.qpos[adr+3:adr+7] = transformed
        mujoco.mj_forward(model, data)
        entries = [None]*6; entries[face] = params
        manager = SixInterfaceMagnetForces(model, data, entries)
        force = manager.compute()
        dof_source = int(model.jnt_dofadr[model.joint(f'module_{face}/module_root').id])
        dof_target = int(model.jnt_dofadr[joint])
        # Four pairs at the reference center distance, each 5 N. The cross
        # pairs are outside the 15 mm experimental cutoff at this 35 mm ring.
        assert np.linalg.norm(force[dof_source:dof_source+3]) == pytest.approx(20., abs=3e-5)
        np.testing.assert_allclose(force[dof_source:dof_source+3],
                                   -force[dof_target:dof_target+3], atol=1e-10)
        linear, angular = net_free_root_wrench(model, data, force)
        np.testing.assert_allclose(linear, 0., atol=1e-10)
        np.testing.assert_allclose(angular, 0., atol=1e-10)


def test_manager_per_interface_configuration_and_separate_data(model, params):
    a = updated(model); b = updated(model)
    # Separate data with the same shared immutable model has independent state.
    joint = model.joint('module_1/module_root').id
    adr = int(model.jnt_qposadr[joint]); b.qpos[adr] += .01
    mujoco.mj_forward(model, b)
    enabled = [None]*6; enabled[0] = params
    one = SixInterfaceMagnetForces(model, a, enabled)
    other = SixInterfaceMagnetForces(model, b, enabled)
    f1 = one.compute(); f2 = other.compute()
    assert np.linalg.norm(f1-f2) > .1
    np.testing.assert_allclose(one.compute(), f1, atol=1e-12)
    np.testing.assert_allclose(SixInterfaceMagnetForces(model, a, [None]*6).compute(), 0.)
    coupling = np.zeros((4, 4))
    inputs = [None]*6; inputs[0] = coupling
    zero = SixInterfaceMagnetForces(model, a, enabled, inputs)
    coupling[:] = 1.  # Caller mutation cannot silently change the manager.
    np.testing.assert_allclose(zero.compute(), 0.)
    with pytest.raises(ValueError):
        SixInterfaceMagnetForces(model, a, enabled[:5])
    with pytest.raises(ValueError):
        SixInterfaceMagnetForces(model, a, enabled, [np.ones((3, 3))]*6)
    another_model = make_physical_zbot_model()
    with pytest.raises(ValueError):
        SixInterfaceMagnetForces(another_model, a, enabled)
