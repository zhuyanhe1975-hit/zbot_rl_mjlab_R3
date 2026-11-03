"""Independent geometry checks; dimensions below are illustrative only."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.connector_geometry import (ConnectorGeometry, make_standalone_model, make_standalone_spec,
    merge_convex_triangles, socket_triangles)


@pytest.fixture(scope='module')
def model():
    return make_standalone_model(ConnectorGeometry(.002, .035, .0001))


def pose(model, angle=0., x=0., z=.0005):
    data = mujoco.MjData(model)
    bid = model.body('pin_plate').id
    adr = model.jnt_qposadr[model.body_jntadr[bid]]
    data.qpos[adr:adr+3] = [x, 0., z]
    data.qpos[adr+3:adr+7] = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
    mujoco.mj_forward(model, data)
    return data


def distances(model, data, pin):
    pinid = model.geom(f'pin_{pin}').id
    return np.array([mujoco.mj_geomDistance(model, data, pinid, g, .02, None)
                     for g in range(model.ngeom)
                     if model.geom(g).name.startswith('socket_piece_')])


@pytest.mark.parametrize('quarter', range(4))
def test_four_registrations_have_open_holes(model, quarter):
    data = pose(model, quarter * math.pi/2)
    for pin in range(4):
        assert distances(model, data, pin).min() == pytest.approx(.0001, abs=2e-8)
    assert data.ncon == 0


def test_wall_contact_follows_clearance(model):
    data = pose(model, x=.001)
    for pin in range(4):
        assert distances(model, data, pin).min() < -.0005
    assert any(model.geom(c.geom1).name.startswith('pin_') or
               model.geom(c.geom2).name.startswith('pin_') for c in data.contact)


def test_wall_contact_resists_translation_and_yaw(model):
    displaced = pose(model, x=.0003)
    assert displaced.qfrc_constraint[6] < 0
    rotated = pose(model, angle=.006)
    assert rotated.qfrc_constraint[11] < 0
    for data in (displaced, rotated):
        mujoco.mj_step(model, data, nstep=10)
        assert not np.any(data.warning.number)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_plates_support_each_other(model):
    data = pose(model, z=-.0001)
    pid = model.geom('pin_plate_disc').id
    assert min(mujoco.mj_geomDistance(model, data, pid, g, .01, None)
               for g in range(model.ngeom) if model.geom(g).name.startswith('socket_piece_')) < 0


def test_supported_contact_stays_finite_under_one_newton(model):
    data = pose(model)
    dof = model.jnt_dofadr[model.body_jntadr[model.body('pin_plate').id]]
    data.qfrc_applied[dof+2] = -1.
    peak_contacts = 0
    for _ in range(1000):
        mujoco.mj_step(model, data)
        peak_contacts = max(peak_contacts, data.ncon)
    assert peak_contacts > 0
    assert not np.any(data.warning.number)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_hole_disk_and_pin_dimensions():
    config = ConnectorGeometry(.002, .035, .0001)
    assert config.disk_radius_m == .05 and config.disk_thickness_m == .002
    assert config.pin_height_m == .002 and config.hole_radius_m == pytest.approx(.0021)
    triangles = socket_triangles(config)
    pieces = merge_convex_triangles(triangles)
    def area(poly):
        return abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                                - poly[:, 1] * np.roll(poly[:, 0], -1)))) / 2
    assert len(pieces) < len(triangles)
    assert sum(map(area, pieces)) == pytest.approx(sum(map(area, triangles)), rel=1e-10)
    with pytest.raises(ValueError):
        ConnectorGeometry(.02, .04, .002)


def test_exported_xml_has_the_same_socket_gap(tmp_path):
    spec = make_standalone_spec(ConnectorGeometry(.002, .035, .0001))
    spec.compile()
    xml = tmp_path / 'connector.xml'
    xml.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = pose(model)
    for pin in range(4):
        assert distances(model, data, pin).min() == pytest.approx(.0001, abs=2e-8)
