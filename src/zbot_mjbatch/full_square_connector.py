"""Standalone full-disc collision prototype with four polygonal apertures.

The socket is a 64-sided convex polygonal approximation of the disc minus four
square or octagonal through-holes, decomposed into non-overlapping convex pieces.
The default geometric dimensions are assumptions, not measurements.  No robot
geometry, connector magnet forces, or old experiment is changed by this module.
"""
from dataclasses import dataclass
import math

import numpy as np

from zbot_mjbatch.connector_geometry import make_standalone_model, make_standalone_spec


@dataclass(frozen=True)
class FullSquareConnector:
    pin_radius_m: float = .002
    bolt_radius_m: float = .035
    radial_clearance_m: float = .0001
    disk_radius_m: float = .05
    disk_thickness_m: float = .002
    pin_height_m: float = .002
    outer_segments: int = 64
    hole_segments: int = 8

    def __post_init__(self):
        vals = (self.pin_radius_m, self.bolt_radius_m, self.radial_clearance_m,
                self.disk_radius_m, self.disk_thickness_m, self.pin_height_m)
        if any(not math.isfinite(v) or v <= 0 for v in vals):
            raise ValueError('all geometric dimensions must be positive and finite')
        if not isinstance(self.outer_segments, int) or self.outer_segments < 16 or self.outer_segments % 4:
            raise ValueError('outer polygon must have a multiple of four sides, at least 16')
        if self.hole_segments not in (4, 8, 16):
            raise ValueError('supported hole polygon resolutions are 4, 8 and 16')
        corner_radius = self.hole_radius_m / math.cos(math.pi / self.hole_segments)
        if self.bolt_radius_m + corner_radius >= self.disk_radius_m * math.cos(math.pi / self.outer_segments):
            raise ValueError('square holes must fit inside the outer polygon')
        if math.sqrt(2.) * self.bolt_radius_m <= 2. * corner_radius:
            raise ValueError('square holes must not overlap')

    @property
    def hole_radius_m(self):
        return self.pin_radius_m + self.radial_clearance_m

    @property
    def centers(self):
        a = np.arange(4) * math.pi / 2.
        return self.bolt_radius_m * np.column_stack((np.cos(a), np.sin(a)))


def make_full_square_model(config=FullSquareConnector()):
    """Create isolated two-body MuJoCo model with full socket support."""
    return make_standalone_model(config)


def make_full_square_spec(config=FullSquareConnector()):
    """Create editable isolated spec; physical parameters match round prototype."""
    return make_standalone_spec(config)
