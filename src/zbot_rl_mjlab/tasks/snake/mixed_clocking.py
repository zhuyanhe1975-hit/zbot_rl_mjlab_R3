"""Six-module open snake with fixed mixed connection clocking."""
import math

import mujoco
import numpy as np

from ...robot import get_spec
from .rz270 import CONNECTION_BODIES

CLOCKING_DEG = (270, 270, 180, 90, 90)


def mixed_clocking_spec():
    """Apply each clocking about its mounting normal relative to the 180° model."""
    spec = get_spec()
    for name, angle in zip(CONNECTION_BODIES, CLOCKING_DEG, strict=True):
        body = spec.body(name)
        delta = math.radians(angle - 180) / 2
        q = np.empty(4)
        mujoco.mju_mulQuat(q, body.quat, np.array((math.cos(delta), 0., 0., math.sin(delta))))
        body.quat = q
    return spec
