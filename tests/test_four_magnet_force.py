"""Conservation and numerical checks for the uncalibrated pair-force model."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.four_magnet_force import FourMagnetParameters, four_magnet_forces, magnetic_qfrc


@pytest.fixture
def parameters():
    # Illustrative values only, not inferred from a 100 N break criterion.
    return FourMagnetParameters(5., .002, .001, .006, .015)


def points(z):
    return np.array([[.025, 0., z], [0., .025, z], [-.025, 0., z], [0., -.025, z]])


def test_required_parameters_and_calibration(parameters):
    with pytest.raises(TypeError):
        FourMagnetParameters()
    with pytest.raises(ValueError):
        FourMagnetParameters(5., .008, .001, .006, .015)
    a, b = points(0.), points(.002)
    f1, f2, energy = four_magnet_forces(a, b, parameters, np.eye(4))
    np.testing.assert_allclose(f1, np.tile([0., 0., 5.], (4, 1)), atol=1e-12)
    np.testing.assert_allclose(f1 + f2, 0., atol=1e-12)
    assert energy < 0.


def test_force_is_negative_gradient_of_potential(parameters):
    a, b = points(0.), points(.003)
    b[0, 0] += .0009
    forces, _, _ = four_magnet_forces(a, b, parameters)
    for i in range(4):
        for j in range(3):
            plus, minus = a.copy(), a.copy()
            plus[i, j] += 1e-7
            minus[i, j] -= 1e-7
            up = four_magnet_forces(plus, b, parameters)[2]
            down = four_magnet_forces(minus, b, parameters)[2]
            assert forces[i, j] == pytest.approx(-(up-down)/2e-7, rel=2e-6, abs=1e-8)


def test_force_and_moment_balance_at_arbitrary_poses(parameters):
    a, b = points(.01), points(.013)
    b += np.array([.0013, -.0008, .0006])
    f, g, _ = four_magnet_forces(a, b, parameters)
    np.testing.assert_allclose(f.sum(0)+g.sum(0), 0., atol=1e-12)
    torque = np.cross(a, f).sum(0)+np.cross(b, g).sum(0)
    np.testing.assert_allclose(torque, 0., atol=1e-12)


def test_coincident_centers_and_smooth_cutoff(parameters):
    a, b = points(0.), points(0.)
    f, g, energy = four_magnet_forces(a, b, parameters)
    assert np.isfinite(np.r_[f.ravel(), g.ravel(), energy]).all()
    np.testing.assert_allclose(f, 0., atol=1e-12)
    # Only one pair enabled: verify force and potential go smoothly to zero.
    c = np.zeros((4, 4)); c[0, 0] = 1.
    for r in (parameters.switch_start_m, parameters.cutoff_m):
        mag = []
        potential = []
        for delta in (-1e-8, 0., 1e-8):
            p = a.copy(); q = b.copy(); q[0] = p[0] + [0., 0., r+delta]
            fp, _, u = four_magnet_forces(p, q, parameters, c)
            mag.append(np.linalg.norm(fp[0])); potential.append(u)
        assert max(np.abs(np.diff(mag))) < 1e-3
        assert max(np.abs(np.diff(potential))) < 1e-5
    b[0] = a[0]+[0., 0., parameters.cutoff_m]
    assert np.linalg.norm(four_magnet_forces(a, b, parameters, c)[0]) == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('coupling_sign', [1., -1.])
def test_energy_gradient_at_zero_and_switch_boundaries(parameters, coupling_sign):
    a = points(0.); b = points(0.)
    coupling = np.zeros((4, 4)); coupling[0, 0] = coupling_sign
    h = 1e-9
    for distance in (0., 1e-9, 1e-6, parameters.switch_start_m - h,
                     parameters.switch_start_m, parameters.switch_start_m + h,
                     parameters.cutoff_m - h, parameters.cutoff_m,
                     parameters.cutoff_m + h):
        b[0] = a[0] + [0., 0., distance]
        force = four_magnet_forces(a, b, parameters, coupling)[0][0, 2]
        plus, minus = a.copy(), a.copy()
        plus[0, 2] += h; minus[0, 2] -= h
        up = four_magnet_forces(plus, b, parameters, coupling)[2]
        down = four_magnet_forces(minus, b, parameters, coupling)[2]
        assert force == pytest.approx(-(up-down)/(2*h), rel=1e-5, abs=3e-9)


def test_four_rotation_pairings_without_nearest_index_switch(parameters):
    a = points(0.)
    baseline = four_magnet_forces(a, points(.002), parameters)
    for k in range(4):
        b = np.roll(points(.002), k, axis=0)
        f, _, energy = four_magnet_forces(a, b, parameters)
        np.testing.assert_allclose(f, baseline[0], atol=1e-12)
        assert energy == pytest.approx(baseline[2], abs=1e-12)
    # A caller-specified polarity pattern controls the actual permitted pairs.
    c = -np.eye(4)
    f, _, _ = four_magnet_forces(a, points(.002), parameters, c)
    assert np.all(f[:, 2] < 0.)


def simple_model():
    ring = points(0.)
    sites = ''.join(f'<site name="{{prefix}}{i}" pos="{x} {y} {z}"/>'
                    for i, (x, y, z) in enumerate(ring))
    xml = f'''<mujoco><option gravity="0 0 0"/><worldbody>
      <body name="a"><freejoint/><geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>{sites.replace('{prefix}', 'a')}</body>
      <body name="b" pos=".001 0 .003"><freejoint/><geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>{sites.replace('{prefix}', 'b')}</body>
    </worldbody></mujoco>'''
    return mujoco.MjModel.from_xml_string(xml)


def test_mujoco_generalized_force_does_not_accumulate_or_mutate_other_force(parameters):
    model = simple_model(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    source = [model.site(f'a{i}').id for i in range(4)]
    target = [model.site(f'b{i}').id for i in range(4)]
    data.qfrc_applied[:] = np.arange(model.nv)
    existing = data.qfrc_applied.copy(); scratch = np.full(model.nv, 999.)
    force = magnetic_qfrc(model, data, source, target, parameters, out=scratch)
    assert force is scratch
    np.testing.assert_array_equal(data.qfrc_applied, existing)
    np.testing.assert_allclose(force, magnetic_qfrc(model, data, source, target, parameters), atol=1e-12)
    np.testing.assert_allclose(force[:3]+force[6:9], 0., atol=1e-12)
    total_torque = force[3:6]+force[9:12]+np.cross(data.xpos[model.body('b').id], force[6:9])
    np.testing.assert_allclose(total_torque, 0., atol=1e-12)
    with pytest.raises(ValueError):
        magnetic_qfrc(model, data, source, target, parameters, out=data.qfrc_applied)
    with pytest.raises(ValueError):
        magnetic_qfrc(model, data, source, source, parameters)


def test_rotated_free_body_force_and_world_torque_balance(parameters):
    model = simple_model(); data = mujoco.MjData(model)
    qa = np.empty(4); qb = np.empty(4)
    mujoco.mju_axisAngle2Quat(qa, np.array([0., 1., 0.]), .31)
    mujoco.mju_axisAngle2Quat(qb, np.array([1., 0., 0.]), -.17)
    data.qpos[3:7] = qa; data.qpos[10:14] = qb
    mujoco.mj_forward(model, data)
    source = [model.site(f'a{i}').id for i in range(4)]
    target = [model.site(f'b{i}').id for i in range(4)]
    force = magnetic_qfrc(model, data, source, target, parameters)
    a, b = model.body('a').id, model.body('b').id
    ra = data.xmat[a].reshape(3, 3); rb = data.xmat[b].reshape(3, 3)
    moment = (ra @ force[3:6] + rb @ force[9:12]
              + np.cross(data.xpos[a], force[:3])
              + np.cross(data.xpos[b], force[6:9]))
    np.testing.assert_allclose(force[:3]+force[6:9], 0., atol=1e-12)
    np.testing.assert_allclose(moment, 0., atol=1e-12)
