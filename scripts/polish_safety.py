"""Discrete module-geometry collision and joint-speed auditing for polish."""
import math
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from pathlib import Path


def source_joint_limits():
    """Read the six hinge limits specified by the supplied source asset."""
    xml = ET.parse(Path('assets/zbot_6/ZBOT6.xml')).getroot()
    joint = xml.find("./default/default[@class='module_joint']/joint")
    if joint is None or joint.get('limited') != 'true':
        raise ValueError('source XML does not declare enabled module joint limits')
    limits = np.fromstring(joint.get('range', ''), sep=' ')
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError('invalid source module joint limits')
    return limits


class CollisionAudit:
    def __init__(self, f, max_speed=2 * math.pi, allow_aligned_mating=False, allow_old_mating=False,
                 enforce_source_limits=False):
        self.f = f
        self.max_speed = max_speed
        self.allow_aligned_mating = allow_aligned_mating
        self.allow_old_mating = allow_old_mating
        self.position_limits = source_joint_limits() if enforce_source_limits else None
        ids = np.flatnonzero(f.geom_module >= 0)
        pairs = [(int(a), int(b)) for i, a in enumerate(ids) for b in ids[i+1:]
                 if f.geom_module[a] != f.geom_module[b]]
        self.pairs = np.asarray(pairs, dtype=int)
        self.radii = f.model.geom_rbound[self.pairs].sum(axis=1)
        self.mating = {tuple(sorted(map(int, pair))): face for face, pair in enumerate(f.pair_geoms)}
        self.min_distance_m = None
        self.collision = None
        self.peak_joint_speed_rad_s = 0.
        self.peak_abs_joint_position_rad = 0.
        self.samples = 0

    def sample(self):
        f = self.f
        if f.phase in ('DONE', 'FAILED'):
            return
        speed = float(np.max(np.abs(f.data.qvel[f.dofs])))
        q = f.data.qpos[f.qadr]
        self.peak_abs_joint_position_rad = max(self.peak_abs_joint_position_rad, float(np.max(np.abs(q))))
        if self.position_limits is not None and (np.any(q < self.position_limits[0]) or np.any(q > self.position_limits[1])):
            f.stop(f'source joint position limit: {q.tolist()} outside {self.position_limits.tolist()} rad')
            return
        self.peak_joint_speed_rad_s = max(self.peak_joint_speed_rad_s, speed)
        if speed > self.max_speed:
            f.stop(f'joint speed limit: {speed:.6f} > {self.max_speed:.6f} rad/s')
            return
        connected = f.connected()
        positions = f.data.geom_xpos
        delta = positions[self.pairs[:, 0]] - positions[self.pairs[:, 1]]
        broad = np.einsum('ij,ij->i', delta, delta) <= (self.radii + .001) ** 2
        for index in np.flatnonzero(broad):
            a, b = self.pairs[index]
            face = self.mating.get((int(a), int(b)))
            pose = None
            # Mating surfaces may touch once their sites have reached the
            # existing docking pose tolerance, including before the weld is
            # activated. Other geometry pairs remain protected.
            if face is not None and connected[face]:
                continue
            if face is not None and self.allow_aligned_mating:
                gap, angle, _, _ = f.geometry(face, new_weld=True)
                pose = {'face': face + 1, 'site_gap_m': float(gap), 'site_angle_deg': math.degrees(angle)}
                if gap < f.cfg.capture_position_tolerance and angle < math.radians(f.cfg.capture_angle_tolerance_deg):
                    continue
                # During release an old weld can just have opened while its
                # mating sites are still aligned in the old registration.
                if (self.allow_old_mating and not
                        (getattr(f, 'four_site_headtail_capture', False)
                         and f.phase == 'CAPTURE_HEADTAIL' and face == 5)):
                    old_eq = int(f.old[face])
                    s = int(f.model.eq_obj1id[old_eq]); t = int(f.model.eq_obj2id[old_eq])
                    old_gap = float(np.linalg.norm(f.data.site_xpos[s] - f.data.site_xpos[t]))
                    r0 = f.data.site_xmat[s].reshape(3, 3); r1 = f.data.site_xmat[t].reshape(3, 3)
                    old_angle = float(np.linalg.norm(Rotation.from_matrix(r1.T @ r0).as_rotvec()))
                    pose.update(old_site_gap_m=old_gap, old_site_angle_deg=math.degrees(old_angle))
                    if old_gap < f.cfg.capture_position_tolerance and old_angle < math.radians(f.cfg.capture_angle_tolerance_deg):
                        continue
            distance = float(mujoco.mj_geomDistance(f.model, f.data, int(a), int(b), .001, None))
            self.samples += 1
            if self.min_distance_m is None or distance < self.min_distance_m:
                self.min_distance_m = distance
            if distance <= 0.:
                self.collision = {'time_s': float(f.data.time), 'phase': f.phase,
                                  'geom1': f.model.geom(int(a)).name, 'geom2': f.model.geom(int(b)).name,
                                  'distance_m': distance, 'mating_pose': pose}
                f.stop(f'module geometry collision: {self.collision}')
                return

    def report(self):
        return {'min_module_distance_m': self.min_distance_m, 'collision': self.collision,
                'geometry_samples': self.samples, 'peak_joint_speed_rad_s': self.peak_joint_speed_rad_s,
                'joint_speed_limit_rad_s': self.max_speed,
                'peak_abs_joint_position_rad':self.peak_abs_joint_position_rad,
                'source_joint_limits_rad':None if self.position_limits is None else self.position_limits.tolist(),
                'scope': 'all module geometry pairs except connected mating pair' +
                         (' or aligned mating pair' if self.allow_aligned_mating else '') +
                         '; ground support allowed; 1ms samples'}
