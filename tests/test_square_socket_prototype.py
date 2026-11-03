"""Geometric behaviour of the isolated square-socket approximation."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.square_socket_prototype import SquareSocketGeometry, make_square_model


@pytest.fixture(scope='module')
def model():
    return make_square_model()


def pose(model, angle=0., x=0., y=0., z=.0005):
    data = mujoco.MjData(model)
    body = model.body('pin_plate').id
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[adr:adr+3] = [x, y, z]
    data.qpos[adr+3:adr+7] = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
    mujoco.mj_forward(model, data)
    return data


def pin_to_socket(model, data, pin):
    source = model.geom(f'pin_{pin}').id
    return [mujoco.mj_geomDistance(model, data, source, g, .02, None)
            for g in range(model.ngeom)
            if model.geom(g).name == 'support_disc' or model.geom(g).name.startswith('socket_wall_')]


@pytest.mark.parametrize('quarter', range(4))
def test_four_registered_positions_insert_freely(model, quarter):
    data = pose(model, angle=quarter*math.pi/2)
    for pin in range(4):
        assert min(pin_to_socket(model, data, pin)) == pytest.approx(.0001, abs=2e-8)
    assert data.ncon == 0


def test_offset_and_yaw_have_restoring_contact(model):
    shifted = pose(model, x=.0003)
    rotated = pose(model, angle=.006)
    assert shifted.qfrc_constraint[6] < 0
    assert rotated.qfrc_constraint[11] < 0
    assert min(pin_to_socket(model, shifted, 0)) < 0
    for data in (shifted, rotated):
        mujoco.mj_step(model, data, nstep=10)
        assert not np.any(data.warning.number)


def test_support_disc_contact(model):
    data = pose(model, z=-.0001)
    distance = mujoco.mj_geomDistance(model, data, model.geom('pin_plate_disc').id,
                                     model.geom('support_disc').id, .01, None)
    assert distance < 0


def test_center_disc_does_not_close_holes_or_overlap_walls(model):
    geometry = SquareSocketGeometry()
    assert geometry.support_radius_m < geometry.bolt_radius_m-geometry.hole_half_width_m-geometry.wall_thickness_m
    data = pose(model)
    support = model.geom('support_disc').id
    for i in range(4):
        assert mujoco.mj_geomDistance(model, data, model.geom(f'pin_{i}').id, support, .01, None) > 0
    assert model.ngeom == 22
    with pytest.raises(ValueError):
        SquareSocketGeometry(support_radius_m=.032)
