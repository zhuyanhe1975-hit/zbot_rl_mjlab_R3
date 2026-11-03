"""Registered body welds with no runtime site dependency.

Registration frame geometry is baked into model equality data at compilation.
This is intentionally opt-in: legacy site welds and FSMs remain unchanged.
The weld is a compliant mechanical connection approximation, not four
independent magnet forces, and does not implement magnetic rupture. Existing
load monitoring and the default safety break rule remain the caller's duty.
"""
from __future__ import annotations

import mujoco
import numpy as np


def body_weld_data(site1_pos, site1_quat, site2_pos, site2_quat, *, torque_scale=1.0):
    """Return MuJoCo weld data for coincident local registered connector frames.

    The body2 pose relative to body1 is T_body1_site1 @ inv(T_body2_site2).
    The anchor is the second connector origin in body2 coordinates. Thus both
    force and moment act at the original connector, including any CAD offset.
    MuJoCo expresses angular residuals in body2's frame, whereas site welds
    use site2's frame. Non-isotropic or residual-dependent solver behavior may
    differ despite having identical zero-residual pose and anchors.
    """
    p1, p2 = np.asarray(site1_pos, dtype=float), np.asarray(site2_pos, dtype=float)
    q1, q2 = np.asarray(site1_quat, dtype=float), np.asarray(site2_quat, dtype=float)
    if (p1.shape != (3,) or p2.shape != (3,) or q1.shape != (4,) or q2.shape != (4,)
            or not np.isfinite(np.r_[p1, p2, q1, q2, torque_scale]).all()
            or abs(np.linalg.norm(q1) - 1) > 1e-6 or abs(np.linalg.norm(q2) - 1) > 1e-6
            or torque_scale <= 0):
        raise ValueError('Invalid connector pose or torque scale')
    inverse = np.empty(4); relative = np.empty(4)
    mujoco.mju_negQuat(inverse, q2)
    mujoco.mju_mulQuat(relative, q1, inverse)
    # MjSpec.add_equality takes the *compiled* layout: two local anchor
    # points, not the XML relpose translation. XML compilation transforms the
    # latter into anchor1; supplying relpose translation directly is incorrect.
    return np.r_[p2, p1, relative, torque_scale]


def add_body_weld_from_sites(spec, name, site1, site2, *, active=False,
                             torque_scale=1., solref=(.005, 1.),
                             solimp=(.999, .9999, .001, .5, 2.)):
    """Add a body equality from existing registered frames at model-build time.

    Do not mutate these frames later: body weld data is independent after the
    model is built. Pass the actual registered frame, including its translation,
    separately for every allowed orientation. The returned metadata lets
    audit and wrench code identify the original sites, since eq_obj1id and
    eq_obj2id on the resulting equality are body IDs, not site IDs.
    """
    source, destination = spec.site(site1), spec.site(site2)
    if source is None or destination is None:
        raise ValueError(f'Unknown connector site: {site1}, {site2}')
    if source.parent.name == destination.parent.name:
        raise ValueError('A connector must join two different bodies')
    data = body_weld_data(source.pos, source.quat, destination.pos, destination.quat,
                          torque_scale=torque_scale)
    spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_BODY,
                      name1=source.parent.name, name2=destination.parent.name,
                      active=active, data=data, solref=solref, solimp=solimp)
    return {'weld': name, 'source_site': site1, 'destination_site': site2,
            'body1': source.parent.name, 'body2': destination.parent.name}


def activate_registration(model, data, equality_names, selected=None):
    """Disable all mutually exclusive registrations, optionally activate one.

    The caller must check capture position, orientation, speed and collision
    before activation. MuJoCo otherwise snaps mismatched frames together.
    Changes affect this MjData only; never edit shared MjModel.eq_data per world.
    """
    names = tuple(equality_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected distinct registration weld names')
    ids = []
    for name in names:
        eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq < 0 or model.eq_type[eq] != mujoco.mjtEq.mjEQ_WELD or model.eq_objtype[eq] != mujoco.mjtObj.mjOBJ_BODY:
            raise ValueError(f'Expected body weld: {name}')
        ids.append(eq)
    if selected is not None and selected not in names:
        raise ValueError(f'Unknown registration: {selected}')
    data.eq_active[ids] = False
    if selected is not None:
        data.eq_active[ids[names.index(selected)]] = True
