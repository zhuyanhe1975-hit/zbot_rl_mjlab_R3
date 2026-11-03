"""Opt-in Zbot collision geometry experiment, not a complete robot controller.

Replace each end of the six modules by a full 100 mm disc with four square
holes or pins, while retaining a clipped approximation of the original convex
half-module shell. There are no connector welds, permanent magnets, state
machine, physical safety audit or automatic contact-pair exceptions here.
Dimensions are adjustable assumptions. Use only for isolated diagnostics.
"""
from __future__ import annotations

import math
import numpy as np
import mujoco

from zbot_mjbatch.connector_geometry import inject_connector_collision
from zbot_mjbatch.full_square_connector import FullSquareConnector
from zbot_mjbatch.shell_collision import ShellCut, add_clipped_collision_shells


def _site(spec, name):
    # Avoid MjSpec cached name views after model element insertions.
    site = next((x for x in spec.sites if x.name == name), None)
    if site is None:
        raise ValueError(f'missing site {name}')
    return site


def _flipped_site(spec, name, new_name):
    original = _site(spec, name)
    quat = np.empty(4)
    # Source body is in port -Z and inlet body in port +Z; the injector's
    # pin-body +Z and socket-body -Z conventions require flipping both frames.
    mujoco.mju_mulQuat(quat, np.asarray(original.quat), np.array([0., 1., 0., 0.]))
    original.parent.add_site(name=new_name, pos=np.asarray(original.pos).copy(),
                             quat=quat, size=(.001,)*3, rgba=(0.,0.,0.,0.))


def _add_magnet_sites(spec, face, pin_frame_name, socket_frame_name, offset,
                      *, radius, embed_depth, phase):
    """Place passive landmarks inside the plates; these sites add no force.

    Both frames point +Z into the pin plate; the socket extends in -Z. The
    four points are evenly spaced around the disk, with a configurable phase
    independent of peg locations. Name and ordering are stable per interface.
    """
    pin_frame=_site(spec,pin_frame_name)
    socket_frame=_site(spec,socket_frame_name)
    frames=[]
    for frame in (pin_frame,socket_frame):
        quat=np.asarray(frame.quat,dtype=float).copy()
        rot=np.empty(9);mujoco.mju_quat2Mat(rot,quat)
        frames.append((frame.parent, np.asarray(frame.pos,dtype=float).copy(), rot.reshape(3,3), quat))
    names=[]
    for label, sign, (body, pos, rot, quat) in zip(('source','target'), (1.,-1.), frames):
        side=[]
        for j in range(4):
            angle=phase+j*math.pi/2
            local=np.asarray(offset,dtype=float)+np.array([radius*math.cos(angle),radius*math.sin(angle),sign*embed_depth])
            name=f'physical_connector_{face}_magnet_{label}_{j}'
            if any(s.name==name for s in spec.sites):
                raise ValueError(f'magnet site already exists: {name}')
            body.add_site(name=name, pos=pos+rot@local, quat=quat,
                          size=(.001,)*3, rgba=(1.,.25,.12,.85) if label=='source' else (.15,.7,1.,.85))
            side.append(name)
        names.append(tuple(side))
    return {'source':names[0], 'target':names[1]}


def add_physical_zbot_geometry(spec, config=None, *, pin_shape='taper',
                               pin_tip_radius_m=.0018, pin_sides=16,
                               magnet_radius_m=.035, magnet_embed_depth_m=.001,
                               magnet_phase_rad=math.pi/4,
                               inlet_shell_clearance_m=0.):
    """Decorate fresh six-module spec, return collision geometry metadata.

    Does not add any connectivity/equality or remove preexisting exclusions or
    explicit collision pairs. To prevent old hulls or exclusions blocking the
    result callers must pass a fresh spec without preexisting contacts/welds.
    The inlet-frame CAD mounting offset is inherited from existing project
    conventions; actual magnet locations and peg dimensions remain unmeasured.
    """
    config = config or FullSquareConnector(hole_segments=4)
    if config.hole_segments != 4:
        raise ValueError('selected experiment requires square holes')
    if pin_shape not in ('taper', 'cylinder'):
        raise ValueError('pin_shape must be taper or cylinder')
    if (not math.isfinite(pin_tip_radius_m) or not 0 <= pin_tip_radius_m <= config.pin_radius_m
            or not isinstance(pin_sides, int) or pin_sides < 8 or pin_sides % 4):
        raise ValueError('invalid tapered pin tip radius or polygon side count')
    if (not math.isfinite(magnet_radius_m) or not 0 < magnet_radius_m < config.disk_radius_m
            or not math.isfinite(magnet_embed_depth_m)
            or not 0 <= magnet_embed_depth_m <= config.disk_thickness_m
            or not math.isfinite(magnet_phase_rad)):
        raise ValueError('invalid magnet radius, embed depth, or phase')
    if len(spec.pairs) or len(spec.excludes) or len(spec.equalities):
        raise ValueError('fresh spec without legacy pairs, excludes or welds required')
    if not math.isfinite(inlet_shell_clearance_m) or inlet_shell_clearance_m < 0:
        raise ValueError('inlet shell clearance must be finite and nonnegative')
    cuts=[]
    for i in range(6):
        cuts.extend((ShellCut(f'module_{i}/a{i+1}_geom', f'module_{i}/port_in', 1,
                              config.disk_thickness_m + inlet_shell_clearance_m),
                     ShellCut(f'module_{i}/b{i+1}_geom', f'module_{i}/port_out', -1,
                              config.disk_thickness_m)))
    shell_metadata = add_clipped_collision_shells(spec, cuts)
    for i in range(6):
        _flipped_site(spec, f'module_{i}/port_out', f'module_{i}/physical_out_frame')
        _flipped_site(spec, f'module_{i}/port_in', f'module_{i}/physical_in_frame')
    # Rotation by pi about X maps old site-local coordinates to flipped frame.
    flip = np.diag([1., -1., -1.])
    meshname='physical_taper_pin_shared_mesh'
    if pin_shape == 'taper':
        mesh=spec.add_mesh(name=meshname,
            scale=(config.pin_radius_m,config.pin_radius_m,config.pin_height_m/2))
        # Built-in cone radius is the ratio between its tip and base. Its base
        # points -Z; rotate each geom 180 deg about its own local X so the tip
        # projects into the opposite module, matching the cylinder prototype.
        mesh.make_cone(pin_sides, pin_tip_radius_m/config.pin_radius_m)
    result=[]
    for i in range(6):
        inlet = _site(spec, f'module_{(i+1)%6}/port_in')
        if not np.allclose(inlet.quat, [1.,0.,0.,0.], atol=1e-9):
            raise ValueError('unexpected inlet frame; cannot apply CAD offset')
        offset=flip @ -np.asarray(inlet.pos, dtype=float)
        metadata=inject_connector_collision(spec, f'module_{i}/physical_out_frame',
            f'module_{(i+1)%6}/physical_in_frame', config,
            pin_center_site=offset, socket_center_site=offset,
            prefix=f'physical_connector_{i}')
        if pin_shape == 'taper':
            for name in metadata['pins']:
                geom=next((item for item in spec.geoms if item.name==name), None)
                if geom is None:
                    raise ValueError(f'missing injected pin {name}')
                quat=np.empty(4)
                mujoco.mju_mulQuat(quat,np.asarray(geom.quat,dtype=float).copy(),np.array([0.,1.,0.,0.]))
                geom.type=mujoco.mjtGeom.mjGEOM_MESH
                geom.meshname=meshname
                geom.quat=quat
        metadata['magnets']=_add_magnet_sites(spec, i, f'module_{i}/physical_out_frame',
            f'module_{(i+1)%6}/physical_in_frame', offset, radius=magnet_radius_m,
            embed_depth=magnet_embed_depth_m, phase=magnet_phase_rad)
        result.append(metadata)
    return {'shells':shell_metadata, 'connectors':result, 'config':config,
            'pin_shape':pin_shape, 'pin_tip_radius_m':pin_tip_radius_m,
            'pin_sides':pin_sides, 'magnet_radius_m':magnet_radius_m,
            'magnet_embed_depth_m':magnet_embed_depth_m, 'magnet_phase_rad':magnet_phase_rad,
            'inlet_shell_clearance_m':inlet_shell_clearance_m}


def make_physical_zbot_model(config=None, *, pin_shape='taper',
                             pin_tip_radius_m=.0018, pin_sides=16,
                             magnet_radius_m=.035, magnet_embed_depth_m=.001,
                             magnet_phase_rad=math.pi/4):
    """Construct independent, unconstrained MuJoCo scene for diagnostics."""
    from mjlab.scene import Scene
    from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg
    cfg=reference_env_cfg().scene
    cfg.spec_fn=lambda spec: add_physical_zbot_geometry(spec, config,
        pin_shape=pin_shape, pin_tip_radius_m=pin_tip_radius_m, pin_sides=pin_sides,
        magnet_radius_m=magnet_radius_m, magnet_embed_depth_m=magnet_embed_depth_m,
        magnet_phase_rad=magnet_phase_rad)
    scene=Scene(cfg, device='cpu')
    model=scene.compile()
    model.opt.timestep=.001
    return model
