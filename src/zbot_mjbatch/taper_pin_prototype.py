"""Independent MuJoCo built-in cone-mesh pin and square-hole prototype.

The 3.6 mm nominal tip is an adjustable approximation, not a measurement.
Four convex mesh geoms share one built-in mesh. No production scene, magnetic
forces, contact filtering or state machine is altered.
"""
from __future__ import annotations

import math
import mujoco

from zbot_mjbatch.full_square_connector import FullSquareConnector, make_full_square_spec


def tapered_pin_spec(*, tip_radius_m=.0018, sides=16):
    """Build 4 mm base / 2 mm tall tapered pins; tip points along local -Z.

    The built-in cone generator's radius parameter is the tip/base ratio.
    Polygon circumradii are specified; inradii are slightly smaller. MuJoCo
    compiles and recenters the mesh, so use compiled transforms to examine it.
    """
    config=FullSquareConnector(hole_segments=4)
    if not math.isfinite(tip_radius_m) or not 0 <= tip_radius_m <= config.pin_radius_m:
        raise ValueError('tip radius must be finite and between zero and base radius')
    if not isinstance(sides,int) or sides < 8 or sides % 4:
        raise ValueError('mesh sides must be a multiple of four, at least eight')
    spec=make_full_square_spec(config)
    mesh=spec.add_mesh(name='taper_pin_shared_mesh',
                       scale=(config.pin_radius_m,config.pin_radius_m,config.pin_height_m/2))
    mesh.make_cone(sides,tip_radius_m/config.pin_radius_m)
    for i in range(4):
        geom=next(g for g in spec.geoms if g.name==f'pin_{i}')
        geom.type=mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname=mesh.name
        # Built-in base is -Z, tip +Z. Rotate pi around X; existing center
        # z=-1mm then gives base z=0 and tip z=-2mm in the parent body.
        geom.quat=(0.,1.,0.,0.)
    return spec


def tapered_pin_model(*, tip_radius_m=.0018, sides=16):
    return tapered_pin_spec(tip_radius_m=tip_radius_m,sides=sides).compile()
