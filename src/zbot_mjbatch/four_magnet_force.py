"""Uncalibrated, finite-range, conservative point-magnet approximation.

This module supplies a *model*, not measured magnetic properties. Each of the
four physical magnet centers on one body interacts with each of the four on
the other body. It is neither a dipole field nor a weld or a collision model.
The optional 4x4 coupling matrix must be supplied explicitly when polarities
or unavailable pairs differ. Site positions are physical magnet centers, not
surface projections, and must be refreshed for the current state by the caller.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class FourMagnetParameters:
    """Illustrative per-pair parameters; every value must be supplied by caller.

    ``contact_force_n`` is the magnitude of *one* attractive pair at
    ``reference_distance_m`` with unit coupling, not a hard force cap or a
    measurement. The reference distance is the physical distance between the
    magnet-center sites, including their depths inside the facing plates.
    ``softening_m`` regularizes the potential at zero; the quintic switch
    starts at ``switch_start_m`` and reaches zero at ``cutoff_m``. A reference
    outside the flat part would make its intended calibration ambiguous.
    """
    contact_force_n: float
    reference_distance_m: float
    softening_m: float
    switch_start_m: float
    cutoff_m: float

    def __post_init__(self):
        values = (self.contact_force_n, self.reference_distance_m, self.softening_m,
                  self.switch_start_m, self.cutoff_m)
        if not all(math.isfinite(v) and v > 0. for v in values):
            raise ValueError('all magnetic parameters must be positive and finite')
        if not self.reference_distance_m <= self.switch_start_m < self.cutoff_m:
            raise ValueError('require reference distance <= switch start < cutoff')

    @property
    def strength(self):
        """Potential scale chosen to match the force at the reference distance."""
        r = self.reference_distance_m
        return self.contact_force_n * (r*r + self.softening_m**2)**1.5 / r


def _switch(distance, parameters):
    """Quintic switch and derivative: continuous through both boundaries."""
    t = np.clip((distance - parameters.switch_start_m)
                / (parameters.cutoff_m - parameters.switch_start_m), 0., 1.)
    w = 1. - t**3 * (10. - 15.*t + 6.*t*t)
    dw = -30. * t*t * (1.-t)**2 / (parameters.cutoff_m - parameters.switch_start_m)
    return w, dw


def four_magnet_forces(source_positions, target_positions, parameters, coupling=None):
    """Return (source forces, target forces, potential energy) in world frame.

    Positions must have shape (4, 3), one per actual magnet center. Every one
    of the 16 pairs contributes a softened radial potential with a smooth
    cutoff. Unit positive coupling is attraction; negative values model a
    chosen repulsive pair. The 4x4 coupling must be fixed by the caller, not
    switched to the nearest point on each step. Default all-positive coupling
    is a symmetric *assumption*, not known hardware magnet polarity.

    Each pair is equal and opposite, collinear with the two applied points;
    hence it conserves total linear and angular momentum. Magnetic torques,
    off-axis field orientation, hysteresis and changing material properties are
    deliberately absent. With these parameters the force is finite and zero
    at coincident centers, but not necessarily bounded by contact_force_n.
    """
    a = np.asarray(source_positions, dtype=float)
    b = np.asarray(target_positions, dtype=float)
    if a.shape != (4, 3) or b.shape != (4, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('magnet positions must be finite arrays with shape (4, 3)')
    if not isinstance(parameters, FourMagnetParameters):
        raise TypeError('parameters must be FourMagnetParameters')
    c = np.ones((4, 4)) if coupling is None else np.asarray(coupling, dtype=float)
    if c.shape != (4, 4) or not np.isfinite(c).all() or np.any(np.abs(c) > 1.):
        raise ValueError('coupling must be a finite 4x4 array with values in [-1, 1]')
    delta = b[None, :, :] - a[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    w, dw = _switch(distance, parameters)
    soft_r = np.sqrt(distance**2 + parameters.softening_m**2)
    magnitude = parameters.strength * c * (w * distance / soft_r**3 - dw / soft_r)
    direction = np.divide(delta, distance[..., None], out=np.zeros_like(delta),
                          where=distance[..., None] > 0.)
    pair = magnitude[..., None] * direction
    potential = float(np.sum(-parameters.strength * c * w / soft_r))
    return pair.sum(axis=1), -pair.sum(axis=0), potential


def magnetic_qfrc(model, data, source_site_ids, target_site_ids, parameters,
                  coupling=None, out=None):
    """Compute this step's force in a separate MuJoCo generalized-force array.

    The caller must update the MuJoCo position state before calling, then add
    the returned vector to its own qfrc_applied each physics step. If supplied,
    ``out`` is reset and reused; it may not alias ``data.qfrc_applied`` so
    unrelated forces cannot be accidentally removed. This function never
    modifies the model, data state, or another producer's applied force.
    """
    source = np.asarray(source_site_ids, dtype=int)
    target = np.asarray(target_site_ids, dtype=int)
    if (source.shape != (4,) or target.shape != (4,)
            or np.any(source < 0) or np.any(source >= model.nsite)
            or np.any(target < 0) or np.any(target >= model.nsite)
            or len(set(source.tolist())) != 4 or len(set(target.tolist())) != 4
            or set(source.tolist()) & set(target.tolist())):
        raise ValueError('expected four distinct valid site IDs on each side')
    body_a = np.asarray(model.site_bodyid)[source]
    body_b = np.asarray(model.site_bodyid)[target]
    if set(body_a.tolist()) & set(body_b.tolist()):
        raise ValueError('magnet sites on opposite sides must belong to different bodies')
    if out is None:
        result = np.zeros(model.nv)
    else:
        result = np.asarray(out)
        if (result.shape != (model.nv,) or not np.issubdtype(result.dtype, np.floating)
                or not result.flags.writeable or np.shares_memory(result, data.qfrc_applied)):
            raise ValueError('out must be a separate writable floating vector of shape (nv,)')
        result.fill(0.)
    force_a, force_b, _ = four_magnet_forces(data.site_xpos[source], data.site_xpos[target],
                                             parameters, coupling)
    zero_torque = np.zeros(3)
    for sites, bodies, forces in ((source, body_a, force_a), (target, body_b, force_b)):
        for site, body, force in zip(sites, bodies, forces):
            mujoco.mj_applyFT(model, data, force, zero_torque, data.site_xpos[site], int(body), result)
    return result
