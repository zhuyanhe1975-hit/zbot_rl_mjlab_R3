"""Isolated pin-shape comparison; no robot, magnets, or weld changes.

The disk face is local z=0 and the pin projects towards -z. A 4 mm
diameter pin projecting 2 mm cannot have a full exposed capsule cylinder:
the exposed portion is a hemisphere. MuJoCo rejects a zero-length capsule;
we therefore compare an exact half-embedded sphere and a positive-length
capsule. The latter extends beyond the 2 mm-thick disk's rear face.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np

from zbot_mjbatch.full_square_connector import FullSquareConnector, make_full_square_spec


def pin_shape_spec(shape: str, capsule_half_length_m: float = .0001):
    config = FullSquareConnector(hole_segments=4)
    spec = make_full_square_spec(config)
    if shape not in ('cylinder', 'sphere', 'capsule'):
        raise ValueError('shape must be cylinder, sphere, or capsule')
    if not math.isfinite(capsule_half_length_m) or capsule_half_length_m <= 0:
        raise ValueError('capsule half-cylinder length must be positive')
    for i in range(4):
        geom = next(g for g in spec.geoms if g.name == f'pin_{i}')
        if shape == 'sphere':
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size = (config.pin_radius_m, 0., 0.)
            geom.pos = (config.centers[i, 0], config.centers[i, 1], 0.)
        elif shape == 'capsule':
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.size = (config.pin_radius_m, capsule_half_length_m, 0.)
            # Tip remains z=-radius; the cylinder and rear hemisphere extend
            # inside and beyond the disk. The back overhang is 2*half-length.
            geom.pos = (config.centers[i, 0], config.centers[i, 1], capsule_half_length_m)
    return spec


def pin_shape_model(shape: str, capsule_half_length_m: float = .0001):
    return pin_shape_spec(shape, capsule_half_length_m).compile()
