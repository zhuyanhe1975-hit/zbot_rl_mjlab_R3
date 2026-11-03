"""Real-model checks of the four-magnet head-tail controller."""
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.four_site_jacobian import four_site_kinematics
from scripts.polish_headtail_axial import create_refined_with_landmarks, enable_four_site_axial_docking, _ring_pose
from scripts.run_planned_mjbatch import configure_servo
from scripts.polish_safety import CollisionAudit
from scripts.view_headtail_mjbatch_refined import create_refined
from zbot_mjbatch.docking_sites import DockingLandmarks, correspondence


def test_ring_sites_do_not_change_dynamics_or_leak_into_old_model():
    original = create_refined('forward')
    ring = create_refined_with_landmarks(.05)
    try:
        old_model, new_model = original.model, ring.model
        assert new_model.nsite == old_model.nsite + 48
        assert old_model.nq == new_model.nq and old_model.nv == new_model.nv
        for f in (original, ring):
            configure_servo(f, 15., 1.5)
            f.data.ctrl[f.actuators] = np.linspace(-.02, .02, 6)
            mujoco.mj_step(f.model, f.data, nstep=10)
        np.testing.assert_allclose(ring.data.qpos, original.data.qpos, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ring.data.qvel, original.data.qvel, atol=1e-12, rtol=0)
        assert mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_SITE, 'module_5/port_out_ring_0') == -1
    finally:
        original.close(); ring.close()


def test_four_point_actuator_response_matches_dynamics_difference():
    f = create_refined_with_landmarks(.05)
    try:
        configure_servo(f, 15., 1.5)
        m = f.model
        lm = DockingLandmarks.from_model(m, 5)
        target = lm.target[correspondence(1)]
        def predict(actuator, increment):
            d = mujoco.MjData(m)
            mujoco.mj_copyData(d, m, f.data)
            d.ctrl[f.actuators] = f.qref
            d.ctrl[f.actuators[actuator]] += increment
            mujoco.mj_step(m, d, nstep=50)
            mujoco.mj_fwdPosition(m, d)
            e, jac = four_site_kinematics(m, d, lm.source, target, int(target[0]))
            return d.qpos.copy(), e, jac
        nominal = predict(0, 0.)
        # Verify both an asymptotic finite difference and the actual 0.004 rad
        # perturbation used by the controller, with its small truncation error.
        for epsilon in (.0001, .004):
            for i in range(6):
                plus, minus = predict(i, epsilon), predict(i, -epsilon)
                derivative = np.empty(m.nv)
                mujoco.mj_differentiatePos(m, derivative, 2 * epsilon, minus[0], plus[0])
                analytical = nominal[2] @ derivative
                measured = ((plus[1] - minus[1]) / (2 * epsilon)).ravel()
                np.testing.assert_allclose(analytical, measured, atol=2e-7, rtol=2e-5)
        f.phase = 'CAPTURE_HEADTAIL'; f.target = 5; f.cfg.control_dt = .001
        enable_four_site_axial_docking(f)
        f.approach_feedback_update()
        assert np.isfinite(f.qref).all()
        assert f.axial_metrics['align_updates'] == 1
    finally:
        f.close()


def _place_source_at_weld_target(f, position, rotation_vector):
    m, d = f.model, f.data
    source, target = int(m.eq_obj1id[f.old[5]]), int(m.eq_obj2id[f.old[5]])
    source_rotation = d.site_xmat[source].reshape(3, 3)
    target_rotation = d.site_xmat[target].reshape(3, 3)
    desired_rotation = target_rotation @ Rotation.from_rotvec(rotation_vector).as_matrix()
    change = desired_rotation @ source_rotation.T
    root = f.roots[5]
    root_position = d.qpos[root:root+3].copy()
    root_rotation = Rotation.from_quat(d.qpos[root+3:root+7], scalar_first=True).as_matrix()
    desired_position = d.site_xpos[target] + target_rotation @ position
    d.qpos[root:root+3] = desired_position - change @ (d.site_xpos[source] - root_position)
    d.qpos[root+3:root+7] = Rotation.from_matrix(change @ root_rotation).as_quat(scalar_first=True)
    mujoco.mj_forward(m, d)


def test_capture_requires_both_weld_and_four_point_tolerances():
    f = create_refined_with_landmarks(.05)
    try:
        f.phase = 'CAPTURE_HEADTAIL'; f.target = 5
        lm = DockingLandmarks.from_model(f.model, 5)
        _place_source_at_weld_target(f, [0., 0., 0.], [0., 0., 0.])
        errors, angle = _ring_pose(f.model, f.data, lm, 1)
        assert np.max(np.linalg.norm(errors, axis=1)) < 1e-9
        assert angle < 1e-9
        original_geometry = f.geometry
        enable_four_site_axial_docking(f)
        gap, angle, *_ = f.geometry(5, new_weld=True)
        assert gap < f.cfg.capture_position_tolerance
        assert angle < math.radians(f.cfg.capture_angle_tolerance_deg)
        _place_source_at_weld_target(f, [.0009, 0., 0.], [0., 0., math.radians(1.9)])
        gap, angle, *_ = f.geometry(5, new_weld=True)
        assert gap < f.cfg.capture_position_tolerance
        assert angle < math.radians(f.cfg.capture_angle_tolerance_deg)
        errors, _ = _ring_pose(f.model, f.data, lm, 1)
        # 1 mm center and 2 degrees do not imply 1 mm at each 50 mm point.
        assert np.max(np.linalg.norm(errors, axis=1)) > .001
        _place_source_at_weld_target(f, [0., 0., 0.], [0., 0., math.radians(2.1)])
        assert f.geometry(5, new_weld=True)[1] > math.radians(f.cfg.capture_angle_tolerance_deg)
        _place_source_at_weld_target(f, [.0011, 0., 0.], [0., 0., 0.])
        assert f.geometry(5, new_weld=True)[0] > f.cfg.capture_position_tolerance
        assert callable(original_geometry)
    finally:
        f.close()


def test_old_mating_exception_cannot_bypass_four_site_capture(monkeypatch):
    f = create_refined_with_landmarks(.05)
    try:
        f.phase = 'CAPTURE_HEADTAIL'; f.target = 5
        _place_source_at_weld_target(f, [0., 0., 0.], [0., 0., 0.])
        # Force a failing combined four-point gate while the legacy weld site
        # alone appears aligned; emulate an intersecting mating mesh pair.
        f.geometry = lambda face=None, new_weld=False: (.002, 0., 0., 0.)
        f.four_site_headtail_capture = True
        audit = CollisionAudit(f, allow_aligned_mating=True, allow_old_mating=True)
        audit.pairs = np.asarray([f.pair_geoms[5]], dtype=int)
        audit.radii = np.array([100.])
        monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: 0.)
        audit.sample()
        assert f.phase == 'FAILED'
        assert 'module geometry collision' in f.failure
    finally:
        f.close()
