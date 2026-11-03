"""An offline reference distinguishes separated and overlapping hulls."""
import mujoco
import pytest

from scripts.diagnose_taper_distance import nonnegative_convex_distance_mm
from zbot_mjbatch.taper_pin_prototype import tapered_pin_model


def test_reference_hull_distance_and_overlap_are_not_signed():
    model = tapered_pin_model()
    data = mujoco.MjData(model)
    adr = int(model.jnt_qposadr[model.body_jntadr[model.body('pin_plate').id]])
    pin, socket = model.geom('pin_0').id, model.geom('socket_piece_8').id
    data.qpos[adr+2] = .0005
    mujoco.mj_forward(model, data)
    # Independent expected distance from the validated pin and hole geometry.
    assert nonnegative_convex_distance_mm(model, data, pin, socket) == pytest.approx(.149254, abs=2e-4)
    data.qpos[adr] = .0003
    mujoco.mj_forward(model, data)
    # Hull intersection is zero, not a signed penetration depth.
    assert nonnegative_convex_distance_mm(model, data, pin, socket) < 1e-5
