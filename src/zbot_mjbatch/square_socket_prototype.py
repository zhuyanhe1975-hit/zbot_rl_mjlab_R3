"""Experimental four-wall square sockets with a central support disc.

Isolated two-body comparison model, not a full robot or a magnetic model.
The outer annulus of the socket-side 100 mm face is deliberately absent: it
cannot detect impacts there or reproduce its contact pressure and moment.
All dimensions are provisional, not hardware measurements.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np


@dataclass(frozen=True)
class SquareSocketGeometry:
    plate_radius_m: float = .05
    plate_thickness_m: float = .002
    pin_radius_m: float = .002
    pin_height_m: float = .002
    bolt_radius_m: float = .035
    hole_half_width_m: float = .0021
    wall_thickness_m: float = .001
    support_radius_m: float = .0315

    def __post_init__(self):
        numbers = tuple(self.__dict__.values())
        if not all(math.isfinite(v) and v > 0 for v in numbers):
            raise ValueError('all dimensions must be positive and finite')
        if self.hole_half_width_m <= self.pin_radius_m:
            raise ValueError('square opening must fit the cylindrical pin')
        if self.support_radius_m >= self.bolt_radius_m - self.hole_half_width_m - self.wall_thickness_m:
            raise ValueError('support disc must not overlap a hole wall')
        if self.bolt_radius_m + self.hole_half_width_m + self.wall_thickness_m >= self.plate_radius_m:
            raise ValueError('hole walls must fit within the nominal disc')

    @property
    def centers(self):
        return np.array([[self.bolt_radius_m, 0], [0, self.bolt_radius_m],
                         [-self.bolt_radius_m, 0], [0, -self.bolt_radius_m]])


def make_square_spec(config=SquareSocketGeometry()):
    """Build a mechanically isolated equivalent with four square apertures.

    Socket face is z=0 with support and walls occupying [-thickness, 0].
    Male disc lies above z=0 and four 2 mm pins extend downwards. Each wall
    fits edge-to-edge without blocking a square hole. The circular support disc
    and square walls do not form a complete 100 mm socket face.
    """
    spec = mujoco.MjSpec.from_string('''<mujoco model="square_socket_comparison">
      <option timestep="0.001" gravity="0 0 0"/>
      <worldbody>
        <body name="socket"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia="0.001 0.001 0.001"/></body>
        <body name="pin_plate"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia="0.001 0.001 0.001"/></body>
      </worldbody></mujoco>''')
    socket, pins = spec.body('socket'), spec.body('pin_plate')
    h = config.plate_thickness_m
    pins.add_geom(name='pin_plate_disc', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  pos=(0, 0, h / 2), size=(config.plate_radius_m, h / 2, 0))
    socket.add_geom(name='support_disc', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=(0, 0, -h / 2), size=(config.support_radius_m, h / 2, 0))
    half, thickness = config.hole_half_width_m, config.wall_thickness_m
    walls = [(-half-thickness/2, 0., thickness/2, half+thickness),
             ( half+thickness/2, 0., thickness/2, half+thickness),
             (0., -half-thickness/2, half, thickness/2),
             (0.,  half+thickness/2, half, thickness/2)]
    for i, (x, y) in enumerate(config.centers):
        pins.add_geom(name=f'pin_{i}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=(x, y, -config.pin_height_m/2),
                      size=(config.pin_radius_m, config.pin_height_m/2, 0))
        for j, (dx, dy, sx, sy) in enumerate(walls):
            socket.add_geom(name=f'socket_wall_{i}_{j}', type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=(x+dx, y+dy, -h/2), size=(sx, sy, h/2))
    return spec


def make_square_model(config=SquareSocketGeometry()):
    return make_square_spec(config).compile()
