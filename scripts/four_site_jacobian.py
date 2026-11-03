"""Four-point relative kinematics for MuJoCo docking.

All Jacobians include every generalized velocity, including floating roots.
The caller must refresh MuJoCo position-dependent state after changing qpos.
"""

from __future__ import annotations

import mujoco
import numpy as np


def _cross_matrix(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def four_site_kinematics(model, data, source_sites, target_sites, frame_site):
    """Return four paired point errors and Jacobian in a moving site frame.

    ``source_sites`` and ``target_sites`` are four ordered MuJoCo site ids;
    their ordering specifies the chosen 90-degree clocking permutation.
    ``frame_site`` determines the target-aligned coordinate axes. For each pair,
    e_i = R_frame.T @ (p_source - p_target), and the corresponding three rows
    of J satisfy e_dot_i = J_i @ qvel, including rotation of the frame itself.

    Return (errors[4, 3], J[12, nv]). The caller must ensure ``data`` already
    contains position quantities corresponding to its current qpos, e.g. with
    mj_fwdPosition on scratch data. This routine does not advance simulation.
    """
    source_sites = tuple(int(i) for i in source_sites)
    target_sites = tuple(int(i) for i in target_sites)
    if len(source_sites) != 4 or len(target_sites) != 4:
        raise ValueError("exactly four source and target site ids are required")
    ids = (*source_sites, *target_sites, int(frame_site))
    if any(i < 0 or i >= model.nsite for i in ids):
        raise ValueError("invalid MuJoCo site id")
    rotation = data.site_xmat[frame_site].reshape(3, 3)
    frame_jacp = np.zeros((3, model.nv))
    frame_jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, frame_jacp, frame_jacr, frame_site)
    frame_angular_jacobian = rotation.T @ frame_jacr
    errors = np.empty((4, 3))
    jacobian = np.empty((12, model.nv))
    for i, (source, target) in enumerate(zip(source_sites, target_sites)):
        source_jac = np.zeros((3, model.nv))
        target_jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, source_jac, None, source)
        mujoco.mj_jacSite(model, data, target_jac, None, target)
        error = rotation.T @ (data.site_xpos[source] - data.site_xpos[target])
        errors[i] = error
        jacobian[3*i:3*i+3] = (
            rotation.T @ (source_jac - target_jac)
            + _cross_matrix(error) @ frame_angular_jacobian
        )
    return errors, jacobian


def equality_constraint_jacobian(model, data):
    """Return velocity Jacobian of active equality rows from a fresh mj_forward.

    Contact and joint-limit rows are intentionally excluded. They require their
    own inequality-aware dynamics/validation, not this kinematic projection.
    """
    if mujoco.mj_isSparse(model):
        raise ValueError("dense MuJoCo constraint Jacobian required")
    rows = np.asarray(data.efc_type)[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    return np.asarray(data.efc_J).reshape(data.nefc, model.nv)[rows].copy()


def equality_nullspace(constraint_jacobian, rcond=1e-10):
    """Orthonormal basis for instantaneous equality-consistent velocities.

    This is kinematic only; selecting generalized velocities this way is not a
    substitute for actuator dynamics, torque limits or collision validation.
    """
    c = np.asarray(constraint_jacobian, dtype=float)
    if c.ndim != 2:
        raise ValueError("constraint Jacobian must be a matrix")
    _, singular_values, vh = np.linalg.svd(c, full_matrices=True)
    tolerance = rcond * singular_values[0] if singular_values.size else 0.0
    rank = int(np.sum(singular_values > tolerance))
    return vh[rank:].T
