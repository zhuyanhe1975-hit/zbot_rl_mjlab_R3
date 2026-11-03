"""Coverage and contact tests for isolated full-face polygonal sockets."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.connector_geometry import merge_convex_triangles, socket_triangles
from zbot_mjbatch.full_square_connector import FullSquareConnector, make_full_square_model


def in_convex(poly, point):
    edge = np.roll(poly, -1, axis=0) - poly
    delta = np.asarray(point) - poly
    cross = edge[:, 0]*delta[:, 1] - edge[:, 1]*delta[:, 0]
    return bool(np.all(cross >= -1e-12) or np.all(cross <= 1e-12))


@pytest.mark.parametrize('sides', [4, 8])
def test_face_remains_covered_and_apertures_are_empty(sides):
    c = FullSquareConnector(hole_segments=sides)
    pieces = merge_convex_triangles(socket_triangles(c))
    assert len(pieces) < 53
    for center in c.centers:
        assert not any(in_convex(p, center) for p in pieces)
    # The perimeter near all pins and between the pins is represented, unlike
    # a smaller central disk plus four isolated walls.
    for angle in np.arange(16) * math.pi / 8:
        point = .047 * np.array([math.cos(angle), math.sin(angle)])
        assert any(in_convex(p, point) for p in pieces)


@pytest.mark.parametrize('sides', [4, 8])
@pytest.mark.parametrize('quarter', range(4))
def test_matched_pins_fit_all_four_registrations(sides, quarter):
    c = FullSquareConnector(hole_segments=sides)
    m = make_full_square_model(c)
    d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.body_jntadr[m.body('pin_plate').id]]
    d.qpos[qadr+2] = .0005
    a = quarter * math.pi/2
    d.qpos[qadr+3:qadr+7] = [math.cos(a/2), 0, 0, math.sin(a/2)]
    mujoco.mj_forward(m, d)
    socket_geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith('socket_piece_')]
    for pin in range(4):
        pin_id = m.geom(f'pin_{pin}').id
        closest = min(mujoco.mj_geomDistance(m, d, pin_id, g, .01, None) for g in socket_geoms)
        assert closest == pytest.approx(.0001, abs=2e-8)
    assert d.ncon == 0


@pytest.mark.parametrize('sides', [4, 8])
def test_wall_reacts_to_lateral_overtravel(sides):
    m = make_full_square_model(FullSquareConnector(hole_segments=sides))
    d = mujoco.MjData(m)
    qadr = m.jnt_qposadr[m.body_jntadr[m.body('pin_plate').id]]
    dof = m.jnt_dofadr[m.body_jntadr[m.body('pin_plate').id]]
    d.qpos[qadr] = .0003
    d.qpos[qadr+2] = .0005
    mujoco.mj_forward(m, d)
    assert d.ncon > 0
    assert d.qfrc_constraint[dof] < 0


@pytest.mark.parametrize('sides', [4, 8])
def test_pressed_face_has_contact_without_warning(sides):
    m = make_full_square_model(FullSquareConnector(hole_segments=sides))
    d = mujoco.MjData(m)
    dof = m.jnt_dofadr[m.body_jntadr[m.body('pin_plate').id]]
    d.qfrc_applied[dof+2] = -1.
    max_contacts = 0
    for _ in range(100):
        mujoco.mj_step(m, d)
        max_contacts = max(max_contacts, d.ncon)
    assert max_contacts > 0 and np.isfinite(d.qpos).all()
    assert not np.any(d.warning.number)
