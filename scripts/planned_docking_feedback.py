"""Site-frame docking correction for the planned forward workflow."""
import math
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def staged_feedback(f, guarded=False):
    """Prioritize orientation and XY before commanding normal Z correction."""
    gap, angle, _, _ = f.geometry(new_weld=True)
    if gap < f.cfg.capture_position_tolerance * .8 and angle < math.radians(f.cfg.capture_angle_tolerance_deg) * .8:
        return
    if f.headtail_pool is None:
        f.headtail_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='zbot-dock')
    m = f.model
    base = f.qref.copy()
    eps = .004
    controls = np.repeat(base[None], 13, axis=0)
    controls[1:7] += np.eye(6) * eps
    controls[7:] -= np.eye(6) * eps
    eq = f.new[f.target]
    src, dst = int(m.eq_obj1id[eq]), int(m.eq_obj2id[eq])
    def sample(index):
        if getattr(f.headtail_worker, 'model', None) is not m:
            f.headtail_worker.model = m
            f.headtail_worker.data = mujoco.MjData(m)
        d = f.headtail_worker.data
        mujoco.mj_copyData(d, m, f.data)
        d.ctrl[f.actuators] = controls[index]
        mujoco.mj_step(m, d, nstep=max(1, round(.05 / f.cfg.dt)))
        r0 = d.site_xmat[src].reshape(3, 3)
        r1 = d.site_xmat[dst].reshape(3, 3)
        return np.r_[r1.T @ (d.site_xpos[src] - d.site_xpos[dst]),
                     Rotation.from_matrix(r1.T @ r0).as_rotvec()]
    errors = np.asarray(list(f.headtail_pool.map(sample, range(13))))
    jac = ((errors[1:7] - errors[7:]) / (2 * eps)).T
    # Rotation weight converts radians to comparable residual units. Before
    # alignment, the zero z target suppresses change along the port normal.
    residual = errors[0].copy()
    aligning = np.linalg.norm(residual[:2]) > .001 or np.linalg.norm(residual[3:]) > math.radians(2.)
    clearance = .002 if aligning else .0003
    if aligning and (not guarded or residual[2] <= -clearance):
        residual[2] = 0.
    elif guarded and residual[2] > -clearance:
        # Source must stay on the negative-z side of the destination site.
        residual[2] += clearance
    weights = np.array([1., 1., .1 if aligning and (not guarded or errors[0, 2] <= -clearance) else 1., .15, .15, .15])
    jac *= weights[:, None]
    residual *= weights
    correction = -jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), residual)
    if not np.isfinite(correction).all():
        raise ValueError('nonfinite docking correction')
    q = f.data.qpos[f.qadr]
    scale = f.cfg.control_dt / .02
    f.qref = np.clip(base + np.clip(.1 * correction, -.0006, .0006) * scale, q - .08, q + .08)
    f.vref[:] = 0.


def enable_staged_feedback(f, guarded=False):
    f.approach_feedback_update = lambda: staged_feedback(f, guarded)


def enable_normal_guard(f, current_geometry=False):
    """Reject normal-plane crossing at 1 ms, near the active docking site."""
    original_measure = f.measure
    def measure():
        original_measure()
        if current_geometry:
            # Constraint forces were sampled by original_measure at the
            # solver state; refresh poses only after that sampling.
            mujoco.mj_kinematics(f.model, f.data)
        if f.phase not in ('CURL', 'CAPTURE_HEADTAIL', 'APPROACH'):
            return
        eq = f.old[5] if f.phase in ('CURL', 'CAPTURE_HEADTAIL') else f.new[f.target]
        src = int(f.model.eq_obj1id[eq])
        dst = int(f.model.eq_obj2id[eq])
        relative = f.data.site_xmat[dst].reshape(3, 3).T @ (f.data.site_xpos[src] - f.data.site_xpos[dst])
        distance = -float(relative[2])
        if np.linalg.norm(relative) <= .03 and distance < 0.:
            f.stop(f'docking normal plane crossed: signed_distance={distance:.9f}m')
    f.measure = measure


def enable_curl_offset(f, offset):
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (6,) or not np.isfinite(offset).all():
        raise ValueError('invalid CURL offset')
    original_apply = f.apply
    first_step = None
    def apply():
        nonlocal first_step
        original_apply()
        if f.phase != 'CURL':
            return
        if first_step is None:
            first_step = f.step_count
        steps = math.ceil(f.curve.duration / f.cfg.control_dt - 1e-10) * round(f.cfg.control_dt / f.cfg.dt)
        i = min(max(f.step_count - first_step, 0), steps - 1)
        f.data.ctrl[f.actuators] += math.sin(math.pi * i / max(steps - 1, 1)) ** 2 * offset
        demand = f.kp * (f.data.ctrl[f.actuators] - f.data.qpos[f.qadr]) - f.kd * f.data.qvel[f.dofs]
        if not np.isfinite(demand).all() or (getattr(f.cfg,'raw_torque_limit_enabled',True) and np.max(np.abs(demand)) > f.cfg.peak_torque):
            f.stop('requested curl torque protection')
    f.apply = apply


def enable_internal_docking_v1(f, early_switch=True, separate_rotation=False):
    """Separate and rotate face 4, then switch early to site feedback."""
    f.cfg.motion_wait_seconds = 30.
    original_prepare = f.prepare_internal
    def prepare():
        f.cfg.separation_gap = .021
        f.planner.gap = .04 if f.target == 3 else .021
        f.cfg.overlap_rotation = (f.target != 3) and not separate_rotation
        f.planner.overlap_rotation = f.cfg.overlap_rotation
        original_prepare()
    f.prepare_internal = prepare
    if not early_switch:
        return
    original_apply = f.apply
    switched = False
    def apply():
        nonlocal switched
        if f.phase == 'APPROACH' and f.target == 3 and not switched:
            eq = f.new[f.target]
            src, dst = int(f.model.eq_obj1id[eq]), int(f.model.eq_obj2id[eq])
            relative = f.data.site_xmat[dst].reshape(3, 3).T @ (f.data.site_xpos[src] - f.data.site_xpos[dst])
            if -float(relative[2]) < .018:
                f.curve.duration = f.phase_time
                f.qref = f.data.qpos[f.qadr].copy()
                f.vref[:] = 0.
                switched = True
        original_apply()
    f.apply = apply


def enable_predictive_docking_v1(f, alignment_clearance=None):
    """Predict servo dynamics and prioritize site XY/orientation on face 4."""
    original_feedback = f.approach_feedback_update
    start_z = None
    aligned = False
    def feedback():
        nonlocal start_z, aligned
        if f.target != 3 or f.phase != 'APPROACH':
            return original_feedback()
        eq = f.new[3]
        src, dst = int(f.model.eq_obj1id[eq]), int(f.model.eq_obj2id[eq])
        rotation = f.data.site_xmat[dst].reshape(3, 3)
        position = rotation.T @ (f.data.site_xpos[src] - f.data.site_xpos[dst])
        angle = Rotation.from_matrix(rotation.T @ f.data.site_xmat[src].reshape(3, 3)).as_rotvec()
        if start_z is None:
            start_z = float(position[2])
        if np.linalg.norm(position[:2]) < .001 and np.linalg.norm(angle) < math.radians(2.):
            aligned = True
        target_z = -.0004 if aligned else (start_z if alignment_clearance is None else -alignment_clearance)
        base = f.qref.copy()
        eps = .005
        controls = np.repeat(base[None], 13, axis=0)
        controls[1:7] += np.eye(6) * eps
        controls[7:] -= np.eye(6) * eps
        model = f.model
        def simulate(control):
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, model, f.data)
            data.ctrl[f.actuators] = control
            mujoco.mj_step(model, data, nstep=100)
            rot = data.site_xmat[dst].reshape(3, 3)
            return np.r_[rot.T @ (data.site_xpos[src] - data.site_xpos[dst]),
                         Rotation.from_matrix(rot.T @ data.site_xmat[src].reshape(3, 3)).as_rotvec()]
        predicted = simulate(controls[0])
        jac = np.stack([(simulate(controls[i+1]) - simulate(controls[i+7])) / (2*eps)
                        for i in range(6)], axis=1)
        weights = np.array([1., 1., 3., .12, .12, .12])
        jac *= weights[:, None]
        error = (predicted - np.array([0., 0., target_z, 0., 0., 0.])) * weights
        delta = -jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), error)
        if not np.isfinite(delta).all():
            raise ValueError('nonfinite predictive docking correction')
        q = f.data.qpos[f.qadr]
        scale = f.cfg.control_dt / .02
        f.qref = np.clip(base + np.clip(delta * .5, -.004, .004) * scale, q - .3, q + .3)
        f.vref[:] = 0.
    f.approach_feedback_update = feedback
    original_apply = f.apply
    switched = False
    def apply():
        nonlocal switched
        if f.phase == 'APPROACH' and f.target == 3 and not switched:
            f.curve.duration = 0.
            f.qref = f.data.qpos[f.qadr].copy()
            f.vref[:] = 0.
            switched = True
        original_apply()
    f.apply = apply
