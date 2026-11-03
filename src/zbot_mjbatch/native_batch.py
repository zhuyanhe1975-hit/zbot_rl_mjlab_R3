"""Independent six-module native-adhesion MuJoCo / mjbatch model.

This construction uses no MJLab.  Connector dimensions and adhesion are
assumptions; the native geom adhesion is not a calibrated magnetic field.
"""
from __future__ import annotations

import json
import math

import mujoco
import numpy as np
from mjbatch import Batch

from .model import SNAPSHOT, START, _module_spec
from .native_adhesion import add_native_adhesion_proxies
from .physical_zbot_geometry import add_physical_zbot_geometry
from .full_square_connector import FullSquareConnector

DEFAULT_CONTACT_FORCE_N = 35.
DEFAULT_MAGNET_RADIUS_M = .037
PIN_HEIGHT_M = .004
PIN_BASE_RADIUS_M = .002  # existing assumed 4 mm base diameter
PIN_TAPER_ANGLE_RAD = math.radians(15.)
PIN_TIP_RADIUS_M = PIN_BASE_RADIUS_M - PIN_HEIGHT_M * math.tan(PIN_TAPER_ANGLE_RAD)


def make_native_batch_model(*, timestep=.001, contact_force_n=DEFAULT_CONTACT_FORCE_N, interaction_gap_m=.003,
                            inlet_shell_clearance_m=.0001, magnet_radius_m=DEFAULT_MAGNET_RADIUS_M):
    """Build the physical connectors and passive native adhesion without welds.

    The MuJoCo default qpos is the saved closed pose.  The returned data holds
    the independently saved straight starting configuration instead.
    """
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError('timestep must be finite and positive')
    snapshot = json.loads(SNAPSHOT.read_text())
    spec = mujoco.MjSpec()
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    spec.option.iterations = 50
    spec.option.ls_iterations = 20
    spec.option.ccd_iterations = 50
    spec.worldbody.add_geom(name='terrain', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=(10, 10, .1), friction=(.6, .005, .0001))
    for i in range(6):
        spec.attach(_module_spec(i, snapshot), prefix=f'module_{i}/',
                    frame=spec.worldbody.add_frame())
    # Assumed clearance behind the socket: deeper inlet-shell clipping removes
    # collision volume over the whole end slab, not just beneath each hole.
    # This is not a measured physical clearance or a safety certification.
    # The 4 mm peg extends 2 mm past the back of the existing 2 mm
    # through-hole plate. Module shells are collision-disabled below; this
    # model does not represent a measured 4 mm deep socket or its backing.
    connector=FullSquareConnector(hole_segments=4, pin_radius_m=PIN_BASE_RADIUS_M,
                                  pin_height_m=PIN_HEIGHT_M)
    add_physical_zbot_geometry(spec, config=connector, pin_tip_radius_m=PIN_TIP_RADIUS_M,
                               inlet_shell_clearance_m=inlet_shell_clearance_m,
                               magnet_radius_m=magnet_radius_m)
    add_native_adhesion_proxies(spec, contact_force_n=contact_force_n,
                                interaction_gap_m=interaction_gap_m)
    model = spec.compile()
    # For this independent experiment only, the user selected the connector
    # discs, pins, socket pieces and magnetic proxies as the collision geometry.
    # Module shells and original visual meshes consequently also cannot contact
    # the floor. The floor remains enabled for the retained connector geometry.
    # No explicit geom pairs are present to override these masks.
    if model.npair:
        raise ValueError('explicit contact pairs could bypass the geometry mask')
    for geom in range(model.ngeom):
        if (model.geom(geom).name or '').startswith('module_'):
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0
    data = mujoco.MjData(model)
    start = json.loads(START.read_text())
    for i in range(6):
        root = int(model.joint(f'module_{i}/module_root').qposadr[0])
        joint = int(model.joint(f'module_{i}/joint{i+1}').qposadr[0])
        data.qpos[root:root+7] = start['roots'][i]
        data.qpos[joint] = start['joint_angles'][i]
        data.ctrl[model.actuator(f'module_{i}/joint{i+1}').id] = start['joint_angles'][i]
    mujoco.mj_forward(model, data)
    return model, data


class NativeAdhesionBatch:
    """Parallel independent worlds, exposing mjbatch's native bound arrays."""

    def __init__(self, num_envs=1, num_threads=1, **model_options):
        if num_envs < 1 or num_threads < 1:
            raise ValueError('num_envs and num_threads must be positive')
        self.model, initial = make_native_batch_model(**model_options)
        self.batch = Batch(self.model, num_envs, min(num_threads, num_envs), forward=True)
        self.state = self.batch.bind('state')
        self.qpos = self.batch.bind('qpos')
        self.qvel = self.batch.bind('qvel')
        self.ctrl = self.batch.bind('ctrl')
        self.site_xpos = self.batch.bind('site_xpos')
        self.site_xmat = self.batch.bind('site_xmat')
        self.actuator_force = self.batch.bind('actuator_force')
        # Derived slots must be bound before forward/reset, otherwise mjbatch
        # leaves a newly bound slot zero until the next forward or step.
        self.qfrc_adhesion = self.batch.bind('qfrc_adhesion')
        self.qfrc_constraint = self.batch.bind('qfrc_constraint')
        self.warning = self.batch.bind('warning')
        self.initial_state = np.empty(self.batch.nstate)
        mujoco.mj_getState(self.model, initial, self.initial_state,
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        self.reset()

    def reset(self, ids=None):
        ids = np.arange(len(self.state), dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        self.batch.reset(ids)
        self.state[ids] = self.initial_state
        self.batch.forward(ids)

    def step(self, ids=None, nstep=1):
        if ids is None:
            self.batch.step(nstep=nstep)
        else:
            self.batch.step(np.asarray(ids, dtype=np.int64), nstep=nstep)
