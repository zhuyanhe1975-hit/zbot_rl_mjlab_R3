"""Per-physics-step audit for the experimental six-interface native model.

This uses a separate MuJoCo data object to recompute contacts at the *current*
qpos after integration.  The simulator's just-solved actuator forces and warning
counters are read from the original data, not from the recomputation. Contact
sampling is discrete; filtered pairs, tunnelling and geometry/model errors are
not ruled out. In particular this does not use mj_geomDistance: on the installed
engine mesh/mesh queries have returned a false zero for separated objects.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


_CONNECTOR = re.compile(r'^physical_connector_(\d+)_(.+)$')
_MODULE_SHELL = re.compile(r'^module_(\d+)/a\d+_geom_clipped_shell$')


def _limits_from_source():
    path = Path(__file__).resolve().parents[2] / 'assets/zbot_6/ZBOT6.xml'
    joint = ET.parse(path).getroot().find("./default/default[@class='module_joint']/joint")
    if joint is None or joint.get('limited') != 'true':
        raise ValueError('source asset has no enabled module joint position limits')
    limits = np.fromstring(joint.get('range', ''), sep=' ')
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError('invalid source asset joint limits')
    return limits


def classify_contact(name1, name2):
    """Classify only explicitly paired geoms, independent of numerical IDs.

    Pin/backing contact has zero designed clearance and is never accepted as
    normal pin/socket contact. The name and actual body association are checked
    at audit construction and same-module contacts are not audited by design.
    """
    if 'terrain' in (name1, name2):
        return 'ground', None
    matches = [_CONNECTOR.fullmatch(n) for n in (name1, name2)]
    if all(matches) and matches[0].group(1) == matches[1].group(1):
        face = int(matches[0].group(1)); parts = {m.group(2) for m in matches}
        if (any(x.startswith('adhesion_source_') for x in parts) and
                any(x.startswith('adhesion_target_') for x in parts)):
            return 'magnetic_proxy', face
        if any(x.startswith('socket_piece_') for x in parts):
            if 'pin_plate_disc' in parts:
                return 'mating_face', face
            if any(re.fullmatch(r'pin_\d+', x) for x in parts):
                return 'pin_wall', face
    for pin, shell in ((name1, name2), (name2, name1)):
        match = _CONNECTOR.fullmatch(pin); shell_match = _MODULE_SHELL.fullmatch(shell)
        if match and shell_match and re.fullmatch(r'pin_\d+', match.group(2)):
            face = int(match.group(1))
            if int(shell_match.group(1)) == (face + 1) % 6:
                return 'pin_backing', face
    return 'other', None


def _measured_pin_mesh_dimensions(model, data):
    """Read the common tapered-pin mesh in its physical connector frame.

    MuJoCo may recenter/rotate meshes at compilation, so transform vertices
    through the compiled geom pose instead of trusting the original scale.
    All 24 pins must use the same compiled mesh for a single gate to apply.
    """
    ids=[model.geom(f'physical_connector_{face}_pin_{pin}').id
         for face in range(6) for pin in range(4)]
    if any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH for g in ids):
        raise ValueError('four-pin gate requires tapered mesh pins')
    mesh_ids=np.asarray(model.geom_dataid[ids],dtype=int)
    if np.any(mesh_ids != mesh_ids[0]):
        raise ValueError('four-pin gate requires a shared pin mesh')
    g=ids[0]; mesh=mesh_ids[0]
    first=int(model.mesh_vertadr[mesh]); count=int(model.mesh_vertnum[mesh])
    vertices=model.mesh_vert[first:first+count] @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
    site=model.site('module_0/physical_out_frame').id
    local=(vertices-data.site_xpos[site]) @ data.site_xmat[site].reshape(3,3)
    z=local[:,2]; low=float(z.min()); high=float(z.max())
    root=local[np.isclose(z,high,rtol=0.,atol=1e-7),:2]
    tip=local[np.isclose(z,low,rtol=0.,atol=1e-7),:2]
    if high-low <= 0. or len(root) < 4 or len(tip) < 4:
        raise ValueError('invalid tapered pin mesh')
    root_center=root.mean(axis=0); tip_center=tip.mean(axis=0)
    if np.linalg.norm(root_center-tip_center)>1e-7:
        raise ValueError('pin mesh axis does not align with connector frame')
    base=float(np.linalg.norm(root-root_center,axis=1).max())
    tip_radius=float(np.linalg.norm(tip-tip_center,axis=1).max())
    return high-low, base, tip_radius


class NativeSafetyAudit:
    """Track 1 ms samples and return a list of violations from each sample.

    ``sample()`` must be called once initially and after *every* mj_step,
    including the last step. Physical contacts and penetration depths are
    diagnostic only: they do not create a violation or stop the simulation.
    Robot shells are deliberately collision-disabled, so pin-backing and
    other shell contact cannot be detected. Ground support remains enabled.
    """

    def __init__(self, model, data, *, max_speed_rad_s=2 * math.pi,
                 rated_torque_nm=14., peak_torque_nm=20., overload_duration_s=1.,
                 mating_position_m=.001, mating_angle_deg=2.,
                 mating_penetration_m=.0001, ground_penetration_m=.003,
                 pin_center_radius_m=.035, pin_height_m=None,
                 aperture_clearance_m=.0001, pin_base_radius_m=None,
                 pin_tip_radius_m=None, numerical_epsilon_m=1e-8):
        self.model, self.data = model, data
        self.scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, self.scratch)
        observed = _measured_pin_mesh_dimensions(model, self.scratch)
        supplied=(pin_height_m,pin_base_radius_m,pin_tip_radius_m)
        for value, actual in zip(supplied, observed):
            if value is not None and not math.isclose(float(value),actual,rel_tol=1e-4,abs_tol=1e-8):
                raise ValueError('pin gate dimensions do not match compiled pin mesh')
        self.speed_limit = float(max_speed_rad_s)
        self.rated = float(rated_torque_nm)
        self.peak_limit = float(peak_torque_nm)
        self.overload_duration = float(overload_duration_s)
        self.mating_position = float(mating_position_m)
        self.mating_angle = math.radians(mating_angle_deg)
        self.mating_penetration = float(mating_penetration_m)
        self.ground_penetration = float(ground_penetration_m)
        self.pin_center_radius = float(pin_center_radius_m)
        self.pin_height = observed[0]
        self.aperture_clearance = float(aperture_clearance_m)
        self.pin_base_radius = observed[1]
        self.pin_tip_radius = observed[2]
        self.epsilon = float(numerical_epsilon_m)
        vals=(self.speed_limit,self.rated,self.peak_limit,self.overload_duration,
              self.mating_position,self.mating_angle,self.mating_penetration,
              self.ground_penetration,self.pin_center_radius,self.pin_height,
              self.aperture_clearance,self.pin_base_radius,self.pin_tip_radius,self.epsilon)
        if not all(math.isfinite(x) and x > 0 for x in vals):
            raise ValueError('audit limits must all be finite and positive')
        if self.peak_limit < self.rated:
            raise ValueError('peak torque must be at least rated torque')
        if self.pin_tip_radius > self.pin_base_radius:
            raise ValueError('tapered pin tip must not exceed base radius')
        if not math.isclose(float(model.opt.timestep), .001, rel_tol=0, abs_tol=1e-12):
            raise ValueError('audit requires 1 ms MuJoCo timestep')
        if model.nu != 6:
            raise ValueError('audit requires six joint actuators')
        if model.nexclude or model.npair or (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)):
            raise ValueError('audit requires native contacts enabled and no custom pair/exclusion filters')
        if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT) or not np.allclose(model.actuator_gear, np.tile([1.,0.,0.,0.,0.,0.],(6,1))):
            raise ValueError('audit requires direct unit-gear joint actuators')
        self.position_limits = _limits_from_source()
        joints = np.asarray(model.actuator_trnid[:, 0], dtype=int)
        if (len(set(map(int, joints))) != 6 or np.any(joints < 0) or
                np.any(model.jnt_type[joints] != mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError('expected six distinct hinge joint actuators')
        if np.any(model.actuator_forcelimited == 0) or np.any(model.actuator_forcerange[:,0] < -self.peak_limit) or np.any(model.actuator_forcerange[:,1] > self.peak_limit):
            raise ValueError('actuator force saturation must be enabled within peak torque')
        self.qadr = model.jnt_qposadr[joints].astype(int)
        self.dadr = model.jnt_dofadr[joints].astype(int)
        self.geom_names = [model.geom(i).name or '' for i in range(model.ngeom)]
        self.module = []
        for i in range(model.ngeom):
            name = model.body(int(model.geom_bodyid[i])).name or ''
            m = re.match(r'^module_(\d+)/', name)
            self.module.append(int(m.group(1)) if m else None)
        self.frames=[]
        enabled_connector_geoms=set()
        for face in range(6):
            src = model.site(f'module_{face}/physical_out_frame').id
            dst = model.site(f'module_{(face+1)%6}/physical_in_frame').id
            disc = model.geom(f'physical_connector_{face}_pin_plate_disc').id
            # The same connector-local offset is passed to both frames at build.
            rotation = self.scratch.site_xmat[src].reshape(3,3)
            geom_rotation = self.scratch.geom_xmat[disc].reshape(3,3)
            surface = self.scratch.geom_xpos[disc] - geom_rotation[:,2] * model.geom_size[disc,1]
            offset = rotation.T @ (surface - self.scratch.site_xpos[src])
            self.frames.append((src, dst, offset))
            # Fail closed if required geometry is missing or attached to wrong modules.
            expected = [('pin_plate_disc',face)]
            expected += [(f'pin_{i}',face) for i in range(4)]
            expected += [(f'socket_piece_{i}',(face+1)%6) for i in range(14)]
            expected += [(f'adhesion_source_{i}',face) for i in range(4)]
            expected += [(f'adhesion_target_{i}',(face+1)%6) for i in range(4)]
            for kind, owner in expected:
                geom = model.geom(f'physical_connector_{face}_{kind}').id
                enabled_connector_geoms.add(geom)
                if self.module[geom] != owner:
                    raise ValueError('connector geometry attached to unexpected module')
                contype=int(model.geom_contype[geom]); affinity=int(model.geom_conaffinity[geom])
                bit=1 << (face+1)
                if kind.startswith('adhesion_source_'):
                    valid=contype == bit and affinity == 0
                elif kind.startswith('adhesion_target_'):
                    valid=contype == 0 and affinity == bit
                else:
                    valid=contype == 1 and affinity == 1
                if not valid:
                    raise ValueError('connector collision mask changed or disabled')
        for module in range(6):
            for half in ('a','b'):
                geom=model.geom(f'module_{module}/{half}{module+1}_geom_clipped_shell').id
                if self.module[geom] != module or model.geom_contype[geom] or model.geom_conaffinity[geom]:
                    raise ValueError('robot shell collision must be disabled for connector-only audit')
        terrain=model.geom('terrain').id
        if int(model.geom_contype[terrain]) != 1 or int(model.geom_conaffinity[terrain]) != 1:
            raise ValueError('terrain contact disabled')
        for geom in range(model.ngeom):
            if geom not in enabled_connector_geoms and geom != terrain:
                if model.geom_contype[geom] or model.geom_conaffinity[geom]:
                    raise ValueError('unexpected non-connector collision geom enabled')
        for body in range(1, model.nbody):
            parent=int(model.body_parentid[body])
            if parent and parent != body:
                child_name=model.body(body).name or ''
                parent_name=model.body(parent).name or ''
                child=re.match(r'^module_(\d+)/',child_name)
                ancestor=re.match(r'^module_(\d+)/',parent_name)
                if child and ancestor and child.group(1) != ancestor.group(1):
                    raise ValueError('different modules share parent-child filtering')
        self.overload_s = np.zeros(6)
        self.last_time = None
        self.samples = 0
        self.peak_speed = 0.
        self.peak_torque = 0.
        self.max_penetration = {}
        self.worst_contact = {}
        self.contact_counts = {}
        self.first_violation = None
        self.total_violations = 0

    def mating_pose(self, face):
        src, dst, offset = self.frames[face]
        d = self.scratch
        r0 = d.site_xmat[src].reshape(3,3); r1 = d.site_xmat[dst].reshape(3,3)
        p0 = d.site_xpos[src] + r0 @ offset; p1 = d.site_xpos[dst] + r1 @ offset
        gap = float(np.linalg.norm(p1-p0))
        relative = r0.T @ r1
        # Four registration orientations are equivalent, including their full
        # face normal and in-plane rotation. No scipy dependency is required.
        angles=[]
        for quarter in range(4):
            a=quarter*math.pi/2; c=math.cos(a); s=math.sin(a)
            zrot=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            cosine=np.clip((np.trace(relative @ zrot)-1.)/2.,-1.,1.)
            angles.append(math.acos(float(cosine)))
        return gap, min(angles)

    def four_pin_clearance_error(self, face):
        """Conservative pin-axis offset at both ends, in square-hole axes.

        Each axis is checked at the pin base and tip, against each 90 degree
        registration. Requiring <= (hole half-width - base radius) ensures a
        circular cross-section fits in the square for the assumed geometry.
        This gate is conservative, not proof of a collision-free approach.
        """
        src,dst,offset=self.frames[face]; d=self.scratch
        r0=d.site_xmat[src].reshape(3,3); r1=d.site_xmat[dst].reshape(3,3)
        p0=d.site_xpos[src]+r0@offset; p1=d.site_xpos[dst]+r1@offset
        radius=self.pin_center_radius
        centers=np.array([[radius,0.],[0.,radius],[-radius,0.],[0.,-radius]])
        best=math.inf
        for shift in range(4):
            worst=0.
            for j in range(4):
                for depth in (0.,self.pin_height):
                    base=np.array([*centers[j],-depth])
                    in_socket=r1.T @ (p0+r0@base-p1)
                    target=centers[(j+shift)%4]
                    worst=max(worst,float(np.max(np.abs(in_socket[:2]-target))))
            best=min(best,worst)
        return best

    def four_pin_gate_excess(self, face):
        """Conservative signed excess beyond the assumed pin/socket envelope.

        The 4.2 mm square-hole width is represented by the assumed base radius
        plus the radial clearance. The pin's dimensions are read from the
        compiled mesh; its radius decreases linearly from root to tip. For
        each of four registrations, bound
        both ends in each square-hole axis, conservatively inflating the
        projected radius for a tilted pin. Along the axis, both center position
        and taper radius are linear, so the endpoint bounds cover intermediate
        sections. A positive result means the assumed geometric envelope is
        exceeded; it is reported for diagnosis and never flags a contact.

        This is a geometry gate, not a substitute for the native contact depth
        audit. It conservatively considers the full pin length, including the
        portion beyond the 2 mm through-hole plate; there is no modeled backing
        contact there. The base diameter and soft-contact tolerance are
        assumptions, while the gate reads the pin dimensions from the mesh.
        """
        src,dst,offset=self.frames[face]; d=self.scratch
        r0=d.site_xmat[src].reshape(3,3); r1=d.site_xmat[dst].reshape(3,3)
        p0=d.site_xpos[src]+r0@offset; p1=d.site_xpos[dst]+r1@offset
        axis_cos=abs(float(np.dot(r0[:,2],r1[:,2])))
        if axis_cos < 1e-6:
            return math.inf
        hole_half_width=self.pin_base_radius+self.aperture_clearance
        radius=self.pin_center_radius
        centers=np.array([[radius,0.],[0.,radius],[-radius,0.],[0.,-radius]])
        best=math.inf
        for shift in range(4):
            worst=-math.inf
            for j in range(4):
                target=centers[(j+shift)%4]
                for depth, pin_radius in ((0.,self.pin_base_radius),
                                          (self.pin_height,self.pin_tip_radius)):
                    point=np.array([*centers[j],-depth])
                    in_socket=r1.T @ (p0+r0@point-p1)
                    excess=(float(np.max(np.abs(in_socket[:2]-target))) +
                            pin_radius/axis_cos - hole_half_width - self.mating_penetration)
                    worst=max(worst,excess)
            best=min(best,worst)
        return best

    def sample(self):
        m, d = self.model, self.data
        issues=[]
        def bad(kind, **extra):
            issues.append({'kind':kind,'time_s':float(d.time),**extra})
        t=float(d.time)
        dt=0. if self.last_time is None else t-self.last_time
        if self.last_time is not None and not math.isclose(dt, .001, abs_tol=1e-9, rel_tol=0):
            bad('sample_interval', interval_s=dt)
        self.last_time=t
        self.samples+=1
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and
                np.isfinite(d.actuator_force).all() and math.isfinite(t)):
            bad('nonfinite_state')
        warning_indices=np.flatnonzero(d.warning.number)
        if len(warning_indices):
            bad('mujoco_warning', counters=d.warning.number.tolist())
        q=d.qpos[self.qadr]; speed=np.abs(d.qvel[self.dadr]); force=np.abs(d.actuator_force)
        if np.isfinite(q).all() and (np.any(q < self.position_limits[0]) or np.any(q > self.position_limits[1])):
            bad('joint_position', values_rad=q.tolist(), limits_rad=self.position_limits.tolist())
        if np.isfinite(speed).all():
            self.peak_speed=max(self.peak_speed,float(speed.max()))
            if speed.max() > self.speed_limit:
                bad('joint_speed', peak_rad_s=float(speed.max()), limit_rad_s=self.speed_limit)
        if np.isfinite(force).all():
            self.peak_torque=max(self.peak_torque,float(force.max()))
            if force.max() > self.peak_limit + 1e-9:
                bad('joint_peak_torque', peak_nm=float(force.max()), limit_nm=self.peak_limit)
            if self.last_time is not None and dt > 0:
                self.overload_s = np.where(force > self.rated, self.overload_s + dt, 0.)
                if np.any(self.overload_s > self.overload_duration + 1e-10):
                    bad('joint_continuous_overload', durations_s=self.overload_s.tolist(), rated_nm=self.rated)
        # mj_forward on the original data would overwrite the last solver force.
        # Copy only integration state into a separate data buffer for collision.
        s=self.scratch
        s.qpos[:]=d.qpos; s.qvel[:]=d.qvel; s.ctrl[:]=d.ctrl; s.time=d.time
        if m.na: s.act[:]=d.act
        mujoco.mj_forward(m,s)
        if np.any(s.warning.number):
            bad('audit_mujoco_warning', counters=s.warning.number.tolist())
        for c in s.contact:
            a,b=int(c.geom1),int(c.geom2)
            if self.module[a] is not None and self.module[a] == self.module[b]:
                continue
            n1,n2=self.geom_names[a],self.geom_names[b]
            category,face=classify_contact(n1,n2)
            distance=float(c.dist)
            if not math.isfinite(distance):
                bad('nonfinite_contact', geom1=n1,geom2=n2);continue
            self.contact_counts[category]=self.contact_counts.get(category,0)+1
            penetration=max(0., -distance)
            if penetration > self.max_penetration.get(category,0.):
                self.max_penetration[category]=penetration
                self.worst_contact[category]={'time_s':t,'geom1':n1,'geom2':n2,
                                              'distance_m':distance,'face':face}
            else:
                self.max_penetration.setdefault(category,0.)
        if issues:
            self.total_violations+=len(issues)
            if self.first_violation is None:self.first_violation=issues[0]
        return issues

    def report(self):
        return {'samples':self.samples,'first_violation':self.first_violation,
                'total_violations':self.total_violations,'peak_speed_rad_s':self.peak_speed,
                'peak_actuator_torque_nm':self.peak_torque,'joint_position_limits_rad':self.position_limits.tolist(),
                'contact_counts':self.contact_counts.copy(),
                'max_penetration_m':self.max_penetration.copy(),
                'worst_contact_by_category':self.worst_contact.copy(),
                'mating_position_tolerance_m':self.mating_position,
                'mating_angle_tolerance_deg':math.degrees(self.mating_angle),
                'assumed_mating_penetration_reference_m':self.mating_penetration,
                'ground_penetration_reference_m':self.ground_penetration,
                'assumed_four_pin_clearance_m':self.aperture_clearance,
                'assumed_four_pin_radius_m':self.pin_center_radius,
                'assumed_pin_height_m':self.pin_height,
                'assumed_pin_base_radius_m':self.pin_base_radius,
                'assumed_pin_tip_radius_m':self.pin_tip_radius,
                'pin_dimensions_source':'compiled pin mesh; base diameter remains a modeling assumption',
                'assumed_square_hole_half_width_m':self.pin_base_radius+self.aperture_clearance,
                'four_pin_gate':'root and tip taper envelope including allowed mating penetration; '
                                'projected radius conservatively inflated for tilt',
                'scope':'1 ms discrete native contacts; contact and penetration are diagnostic only '
                        'and never cause a violation or stop; only connector discs, pins, sockets, magnetic proxies '
                        'and ground enabled; other robot body collisions are disabled and not audited; '
                        'ground support and connector contact remain physically simulated; '
                        'pin-backing is not modeled because shell collisions are disabled; '
                        'filtered pairs and tunnelling not covered; '
                        'no mj_geomDistance or calibrated material/contact limits'}
