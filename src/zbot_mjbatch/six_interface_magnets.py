"""Per-data six-interface aggregation for the experimental four-point model.

Only the isolated physical Zbot model currently supplies these magnet sites.
The manager does not advance physics, activate constraints, select topology,
install a callback, or change externally applied force buffers. Callers must
invoke it for each physical substep on the appropriate MuJoCo data instance.
"""
from __future__ import annotations

import mujoco
import numpy as np

from zbot_mjbatch.four_magnet_force import FourMagnetParameters, magnetic_qfrc


class SixInterfaceMagnetForces:
    """Aggregate six uncalibrated interfaces on one model/data pair.

    ``parameters`` has six entries of :class:`FourMagnetParameters` or None;
    None disables that interface. Each interface has four source and four
    destination sites. ``couplings`` can contain six 4x4 arrays, or None for
    an all-attractive assumption; changing that assumption requires actual
    magnet polarity information. Parameters and couplings are fixed for this
    manager; use independent managers for separate simulation data instances.

    Before *every* call, the caller must update position-dependent MuJoCo data
    for the current qpos (e.g. mj_fwdPosition), including quantities needed by
    mj_applyFT. Do not assume the site state after mj_step is the integrated
    qpos. Add the returned vector once to the caller-owned qfrc_applied for that
    substep; the caller must reset or rebuild its applied force buffer each
    step rather than accumulate the same magnetic contribution repeatedly.
    """

    def __init__(self, model, data, parameters, couplings=None):
        if data.model is not model:
            raise ValueError('data must be created from the supplied model')
        params = tuple(parameters)
        if len(params) != 6 or any(x is not None and not isinstance(x, FourMagnetParameters) for x in params):
            raise ValueError('parameters must have six FourMagnetParameters or None entries')
        if couplings is None:
            matrix = (None,) * 6
        else:
            matrix = tuple(couplings)
            if len(matrix) != 6:
                raise ValueError('couplings must contain six 4x4 arrays or None')
        sanitized = []
        for c in matrix:
            if c is None:
                sanitized.append(None)
                continue
            c = np.array(c, dtype=float, copy=True)
            if c.shape != (4, 4) or not np.isfinite(c).all() or np.any(np.abs(c) > 1.):
                raise ValueError('each coupling must be a finite 4x4 array in [-1, 1]')
            c.flags.writeable = False
            sanitized.append(c)
        sites = []
        for face in range(6):
            groups = []
            for side in ('source', 'target'):
                ids = []
                for j in range(4):
                    name = f'physical_connector_{face}_magnet_{side}_{j}'
                    ident = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                    if ident < 0:
                        raise ValueError(f'model does not contain magnet site {name}')
                    ids.append(ident)
                groups.append(np.asarray(ids, dtype=int))
            if (len(set(groups[0])) != 4 or len(set(groups[1])) != 4
                    or set(groups[0]) & set(groups[1])
                    or set(model.site_bodyid[groups[0]]) & set(model.site_bodyid[groups[1]])):
                raise ValueError(f'interface {face} has invalid magnet site bodies')
            sites.append(tuple(groups))
        self.model = model
        self.data = data
        self.parameters = params
        self.couplings = tuple(sanitized)
        self.sites = tuple(sites)
        self._scratch = np.empty(model.nv, dtype=float)

    def compute(self, out=None):
        """Return only this step's net magnetic generalized force, shape (nv,).

        ``out`` if provided is cleared; it must be a writable floating vector
        that does not overlap any exposed simulation state / applied forces.
        Without ``out`` a fresh array is returned. Data and non-magnetic forces
        remain untouched, including if some interfaces are disabled.
        """
        if out is None:
            total = np.zeros(self.model.nv)
        else:
            total = np.asarray(out)
            state = (self.data.qpos, self.data.qvel, self.data.act, self.data.ctrl,
                     self.data.qfrc_applied, self.data.xfrc_applied, self._scratch)
            if (total.shape != (self.model.nv,) or not np.issubdtype(total.dtype, np.floating)
                    or not total.flags.writeable or any(np.shares_memory(total, arr) for arr in state)):
                raise ValueError('out must be a separate writable floating vector of shape (nv,)')
            total.fill(0.)
        for (source, target), params, coupling in zip(self.sites, self.parameters, self.couplings):
            if params is None:
                continue
            magnetic_qfrc(self.model, self.data, source, target, params, coupling=coupling, out=self._scratch)
            total += self._scratch
        return total
