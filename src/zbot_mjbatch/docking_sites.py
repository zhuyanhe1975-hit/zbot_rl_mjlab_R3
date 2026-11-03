"""Four virtual docking landmarks and MuJoCo point Jacobians.

Landmarks are visual/kinematic sites, not collision geoms or force elements.
The old connector sites remain available to the MuJoCo weld and load sensor.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np


def _rotation(quat):
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def site_names(port_name):
    return tuple(f'{port_name}_ring_{i}' for i in range(4))


def add_connector_ring_sites(spec, radius=0.05):
    """Add four sites to each side of each of the six original connectors.

    The circle radius is a virtual guidance parameter, not a calibrated hardware
    landmark or collision bound. The old port site is offset from the CAD
    mounting origin by the inlet
    translation. Use the mounting origin as the circle center on *both* sides:
    this is the same offset used to create the existing RZ270 weld variant.
    Source port i mates with inlet (i+1) % 6.  The inlet site orientation must
    be identity, as in the original model; fail rather than guessing otherwise.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    for i in range(6):
        inlet = spec.site(f'module_{(i + 1) % 6}/port_in')
        source = spec.site(f'module_{i}/port_out')
        if inlet is None or source is None:
            raise ValueError(f'missing connector {i} source or inlet site')
        inlet_rotation = _rotation(inlet.quat)
        if not np.allclose(inlet_rotation, np.eye(3), atol=1e-9):
            raise ValueError('inlet site must use the original identity orientation')
        offset = np.asarray(inlet.pos, dtype=float).copy()
        for port_name, port in ((f'module_{i}/port_out', source),
                                (f'module_{(i + 1) % 6}/port_in', inlet)):
            rotation = _rotation(port.quat)
            center = np.asarray(port.pos, dtype=float) - rotation @ offset
            for index, name in enumerate(site_names(port_name)):
                if spec.site(name) is not None:
                    raise ValueError(f'docking landmark {name} already exists')
                angle = index * math.pi / 2
                point = center + rotation @ np.array([radius * math.cos(angle),
                                                       radius * math.sin(angle), 0.])
                port.parent.add_site(name=name, pos=point, size=(.003,)*3,
                                     rgba=(0.2, 0.9, 0.3, 1.))


def correspondence(quarter_turn):
    """Index of destination point matching source i for +quarter_turn*90°.

    Positive angle means source basis relative to the destination basis; source
    and destination are both viewed in their existing +Z site coordinates.
    """
    if not isinstance(quarter_turn, (int, np.integer)):
        raise TypeError('quarter_turn must be an integer')
    return (np.arange(4) + int(quarter_turn)) % 4


@dataclass(frozen=True)
class DockingLandmarks:
    source: np.ndarray
    target: np.ndarray

    @classmethod
    def from_model(cls, model, connector):
        if not isinstance(connector, (int, np.integer)) or not 0 <= connector < 6:
            raise ValueError('connector must be an integer in [0, 5]')
        source = [model.site(name).id for name in site_names(f'module_{connector}/port_out')]
        target = [model.site(name).id for name in site_names(f'module_{(connector + 1) % 6}/port_in')]
        return cls(np.asarray(source, dtype=int), np.asarray(target, dtype=int))

    def errors(self, data, quarter_turn):
        """World-coordinate point residuals, source minus matched target."""
        return data.site_xpos[self.source] - data.site_xpos[self.target[correspondence(quarter_turn)]]

    def jacobian(self, model, data, quarter_turn, dofs=None):
        """Jacobian of the 12 stacked world residuals against qvel DOFs.

        Call mj_forward or the appropriate kinematic update before this method.
        The optional dofs selects columns in the caller's joint order.
        """
        out = np.empty((4, 3, model.nv), dtype=float)
        for i, target in enumerate(self.target[correspondence(quarter_turn)]):
            ja = np.zeros((3, model.nv)); jb = np.zeros_like(ja)
            mujoco.mj_jacSite(model, data, ja, None, int(self.source[i]))
            mujoco.mj_jacSite(model, data, jb, None, int(target))
            out[i] = ja - jb
        result = out.reshape(12, model.nv)
        return result if dofs is None else result[:, np.asarray(dofs, dtype=int)]
