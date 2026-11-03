"""Five-dimensional alignment and axial approach for the head-tail sites."""
import math
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import inverse


def create_refined_with_landmarks(radius):
    """Compile a four-landmark model for new results only.

    The existing factory installs its own connector hook; extend it only for
    this construction and always restore it. Sites add no dynamics or forces.
    """
    from scripts import view_headtail_mjbatch_refined as refined
    import zbot_rl_mjlab.tasks.reconfiguration.full_cpu as full
    from zbot_mjbatch.docking_sites import add_connector_ring_sites
    original = full.add_full_cpu_connectors
    def with_landmarks(spec):
        original(spec)
        add_connector_ring_sites(spec, radius=radius)
    full.add_full_cpu_connectors = with_landmarks
    try:
        return refined.create_refined('forward')
    finally:
        full.add_full_cpu_connectors = original


def _ring_pose(model, data, landmarks, quarter):
    """Four-point residuals, center offset and rigid orientation difference."""
    from zbot_mjbatch.docking_sites import correspondence
    target = landmarks.target[correspondence(quarter)]
    rotation = data.site_xmat[target[0]].reshape(3, 3)
    source_points = data.site_xpos[landmarks.source] @ rotation
    target_points = data.site_xpos[target] @ rotation
    error = source_points - target_points
    source_centered = source_points - source_points.mean(axis=0)
    target_centered = target_points - target_points.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    reflection = np.eye(3)
    reflection[2, 2] = np.linalg.det(vt.T @ u.T)
    relative_rotation = vt.T @ reflection @ u.T
    angle = Rotation.from_matrix(relative_rotation).magnitude()
    return error, float(angle)


def enable_four_site_axial_docking(f, pre_gap=.03, rate=6., quarter=1):
    """Four-point Jacobian + servo-dynamics prediction for head-tail capture.

    Command sensitivity is measured by simulating the actual six servos. The
    complete generalized displacement from each prediction is multiplied by
    the full site Jacobian; floating-body DOFs are never treated as actuators.
    """
    from scripts.four_site_jacobian import four_site_kinematics
    from zbot_mjbatch.docking_sites import DockingLandmarks, correspondence
    landmarks = DockingLandmarks.from_model(f.model, 5)
    matched = landmarks.target[correspondence(quarter)]
    f.four_site_headtail_capture = True
    previous = f.approach_feedback_update
    previous_geometry = f.geometry
    def geometry(face=None, new_weld=False):
        gap, angle, linear_speed, angular_speed = previous_geometry(face, new_weld)
        if f.phase == 'CAPTURE_HEADTAIL' and (face is None or face == 5) and new_weld:
            ring_error, ring_angle = _ring_pose(f.model, f.data, landmarks, quarter)
            # Both original weld-frame and four-point geometry must satisfy
            # capture tolerances. A 2-degree angular tolerance is not a 1-mm
            # bound on every point at a 50-mm radius.
            gap = max(gap, float(np.linalg.norm(ring_error.mean(axis=0))))
            angle = max(angle, ring_angle)
        return gap, angle, linear_speed, angular_speed
    f.geometry = geometry
    stage = 'align'
    f.axial_metrics = {'started': False, 'start_time_s': None, 'align_updates': 0,
                       'axial_updates': 0, 'paused_updates': 0,
                       'max_axial_xy_m': 0., 'max_axial_angle_deg': 0.,
                       'rate_per_s': rate, 'measurement_period_s': f.cfg.control_dt,
                       'landmark_quarter': int(quarter)}

    def feedback():
        nonlocal stage
        if f.phase != 'CAPTURE_HEADTAIL' or f.target != 5:
            return previous()
        m = f.model
        current, angle = _ring_pose(m, f.data, landmarks, quarter)
        center = current.mean(axis=0)
        xy = float(np.linalg.norm(center[:2]))
        if stage == 'align' and xy < .0002 and angle < math.radians(.5):
            stage = 'axial'
            f.axial_metrics['started'] = True
            f.axial_metrics['start_time_s'] = float(f.data.time)
            print('HEADTAIL_AXIAL_START', float(f.data.time), 'z', float(center[2]), flush=True)
        near = center[2] > -.005
        axial_enabled = (stage == 'axial' and xy < (.0002 if near else .0015)
                         and angle < math.radians(.5 if near else 2.))
        if stage == 'align':
            f.axial_metrics['align_updates'] += 1
        else:
            f.axial_metrics['axial_updates'] += 1
            f.axial_metrics['paused_updates'] += int(not axial_enabled)
            f.axial_metrics['max_axial_xy_m'] = max(f.axial_metrics['max_axial_xy_m'], xy)
            f.axial_metrics['max_axial_angle_deg'] = max(f.axial_metrics['max_axial_angle_deg'], math.degrees(angle))
        target = current * math.exp(-rate * .05)
        target[:, 2] += ((min(-.0004, center[2] + .001) if axial_enabled
                          else (center[2] if stage == 'axial' else -pre_gap))
                         - target[:, 2].mean())
        base = f.qref.copy(); eps = .004
        controls = np.repeat(base[None], 13, axis=0)
        controls[1:7] += np.eye(6) * eps
        controls[7:] -= np.eye(6) * eps
        if f.headtail_pool is None:
            f.headtail_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='zbot-ring')

        def simulate(index):
            if getattr(f.headtail_worker, 'model', None) is not m:
                f.headtail_worker.model = m
                f.headtail_worker.data = mujoco.MjData(m)
            d = f.headtail_worker.data
            mujoco.mj_copyData(d, m, f.data)
            d.ctrl[f.actuators] = controls[index]
            mujoco.mj_step(m, d, nstep=50)
            if index == 0:
                mujoco.mj_fwdPosition(m, d)
                residual, jacobian = four_site_kinematics(m, d, landmarks.source, matched, int(matched[0]))
                return d.qpos.copy(), residual, jacobian
            return d.qpos.copy(), None, None

        samples = list(f.headtail_pool.map(simulate, range(13)))
        configuration_response = np.empty((m.nv, 6))
        for i in range(6):
            column = np.empty(m.nv)
            mujoco.mj_differentiatePos(m, column, 2 * eps, samples[i + 7][0], samples[i + 1][0])
            configuration_response[:, i] = column
        response = samples[0][2] @ configuration_response
        weights = np.tile([1., 1., 2.], 4)
        response *= weights[:, None]
        error = (samples[0][1] - target).ravel() * weights
        correction = -response.T @ np.linalg.solve(response @ response.T + 1e-6 * np.eye(12), error)
        if not np.isfinite(correction).all():
            raise ValueError('nonfinite four-point head-tail correction')
        q = f.data.qpos[f.qadr]
        scale = f.cfg.control_dt / .02
        f.qref = np.clip(base + np.clip(.5 * correction, -.003, .003) * scale, q - .15, q + .15)
        f.vref[:] = 0.
    f.approach_feedback_update = feedback


def prealign_goal(f, closure_goal, distance=.03):
    k = f.planner.kin
    anchor = f.planner.sync(f.shim(), 5)
    _, source, destination = k.fk(np.asarray(closure_goal), 5, anchor)
    target = inverse(destination) @ source
    k.q = np.asarray(closure_goal).copy()
    pos = target[:3, 3] + k.inlet[0][:3, :3] @ np.array([0., 0., -distance])
    k.solve(5, anchor, target[:3, :3], position=pos)
    return k.q.copy()


def enable_axial_docking(f, closure_goal, pre_gap=.03, rate=6.):
    """Exponential XY/rotation error targets and gated local-Z approach.

    The update uses a finite-difference 50 ms dynamics prediction. The
    exponential rule applies to the target, not a guarantee for real motion.
    """
    previous = f.approach_feedback_update
    stage = 'align'
    f.axial_metrics={'started':False,'start_time_s':None,'align_updates':0,'axial_updates':0,
                     'paused_updates':0,'max_axial_xy_m':0.,'max_axial_angle_deg':0.,
                     'rate_per_s':rate,'measurement_period_s':f.cfg.control_dt}
    def feedback():
        nonlocal stage
        if f.phase != 'CAPTURE_HEADTAIL' or f.target != 5:
            return previous()
        m=f.model
        eq=f.old[5]; src=int(m.eq_obj1id[eq]); dst=int(m.eq_obj2id[eq])
        def pose(data):
            rot=data.site_xmat[dst].reshape(3,3)
            return np.r_[rot.T @ (data.site_xpos[src]-data.site_xpos[dst]),
                         Rotation.from_matrix(rot.T@data.site_xmat[src].reshape(3,3)).as_rotvec()]
        current=pose(f.data)
        xy=float(np.linalg.norm(current[:2])); angle=float(np.linalg.norm(current[3:]))
        if stage == 'align' and xy < .0002 and angle < math.radians(.5):
            stage='axial'
            f.axial_metrics['started']=True
            f.axial_metrics['start_time_s']=float(f.data.time)
            print('HEADTAIL_AXIAL_START',float(f.data.time),'z',float(current[2]),flush=True)
        # Hysteresis: keep correcting XY/rotation but pause axial advance if
        # these depart too far from alignment.
        near = current[2] > -.005
        axial_enabled=(stage=='axial' and xy < (.0002 if near else .0015)
                       and angle < math.radians(.5 if near else 2.))
        if stage=='align':
            f.axial_metrics['align_updates']+=1
        else:
            f.axial_metrics['axial_updates']+=1
            f.axial_metrics['paused_updates']+=int(not axial_enabled)
            f.axial_metrics['max_axial_xy_m']=max(f.axial_metrics['max_axial_xy_m'],xy)
            f.axial_metrics['max_axial_angle_deg']=max(f.axial_metrics['max_axial_angle_deg'],math.degrees(angle))
        target=current.copy()
        target[:2] *= math.exp(-rate * .05)
        target[3:] *= math.exp(-rate * .05)
        target[2]=(min(-.0004,current[2]+.001) if axial_enabled
                   else (current[2] if stage=='axial' else -pre_gap))
        base=f.qref.copy(); eps=.004
        controls=np.repeat(base[None],13,axis=0)
        controls[1:7]+=np.eye(6)*eps; controls[7:]-=np.eye(6)*eps
        if f.headtail_pool is None:
            f.headtail_pool=ThreadPoolExecutor(max_workers=8,thread_name_prefix='zbot-axial')
        def simulate(index):
            if getattr(f.headtail_worker,'model',None) is not m:
                f.headtail_worker.model=m;f.headtail_worker.data=mujoco.MjData(m)
            data=f.headtail_worker.data
            mujoco.mj_copyData(data,m,f.data);data.ctrl[f.actuators]=controls[index]
            mujoco.mj_step(m,data,nstep=50)
            return pose(data)
        samples=np.asarray(list(f.headtail_pool.map(simulate,range(13))))
        jac=((samples[1:7]-samples[7:])/(2*eps)).T
        weights=np.array([1.,1.,2.,.15,.15,.15]); jac*=weights[:,None]
        error=(samples[0]-target)*weights
        correction=-jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(6),error)
        if not np.isfinite(correction).all():
            raise ValueError('nonfinite head-tail axial correction')
        q=f.data.qpos[f.qadr]
        scale=f.cfg.control_dt/.02
        f.qref=np.clip(base+np.clip(.5*correction,-.003,.003)*scale,q-.15,q+.15)
        f.vref[:]=0.
    f.approach_feedback_update=feedback
