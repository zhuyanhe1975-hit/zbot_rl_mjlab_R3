"""Independent tests for tentative spherical and capsule pin geometry."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.capsule_pin_prototype import pin_shape_model


@pytest.mark.parametrize('shape, expected_back', [
    ('cylinder', 0.), ('sphere', .002), ('capsule', .0022),
])
def test_local_pin_height_and_back_extent(shape, expected_back):
    model = pin_shape_model(shape)
    pin = model.geom('pin_0').id
    radius = model.geom_size[pin, 0]
    assert radius == pytest.approx(.002)
    if shape == 'cylinder':
        zmin, zmax = model.geom_pos[pin, 2] - model.geom_size[pin, 1], model.geom_pos[pin, 2] + model.geom_size[pin, 1]
    elif shape == 'sphere':
        zmin, zmax = model.geom_pos[pin, 2] - radius, model.geom_pos[pin, 2] + radius
    else:
        extent = radius + model.geom_size[pin, 1]
        zmin, zmax = model.geom_pos[pin, 2] - extent, model.geom_pos[pin, 2] + extent
    assert zmin == pytest.approx(-.002)
    assert zmax == pytest.approx(expected_back)


@pytest.mark.parametrize('shape', ['cylinder', 'sphere', 'capsule'])
@pytest.mark.parametrize('quarter', range(4))
def test_hole_fits_at_90_degree_registrations(shape, quarter):
    model = pin_shape_model(shape)
    data = mujoco.MjData(model)
    body = model.body('pin_plate').id
    q = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[q+2] = .0005
    angle = quarter * math.pi / 2
    data.qpos[q+3:q+7] = [math.cos(angle/2), 0., 0., math.sin(angle/2)]
    mujoco.mj_forward(model, data)
    sockets = [g for g in range(model.ngeom) if model.geom(g).name.startswith('socket_piece_')]
    for i in range(4):
        pin = model.geom(f'pin_{i}').id
        gap = min(mujoco.mj_geomDistance(model, data, pin, g, .01, None) for g in sockets)
        assert gap > .00009
    assert data.ncon == 0


@pytest.mark.parametrize('shape', ['cylinder', 'sphere', 'capsule'])
def test_wall_resists_lateral_overtravel(shape):
    model = pin_shape_model(shape)
    data = mujoco.MjData(model)
    body = model.body('pin_plate').id
    joint = model.body_jntadr[body]
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[q] = .0003
    data.qpos[q+2] = .0005
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    assert data.qfrc_constraint[v] < 0
    mujoco.mj_step(model, data, nstep=10)
    assert np.isfinite(data.qpos).all()
    assert not np.any(data.warning.number)


def test_capsule_needs_positive_half_length():
    with pytest.raises(ValueError):
        pin_shape_model('capsule', 0.)
