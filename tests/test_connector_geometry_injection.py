"""Injection respects explicit site-local frames, offsets and body inertia."""
import math

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from zbot_mjbatch.connector_geometry import ConnectorGeometry, inject_connector_collision


def make_spec():
    return mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="p" pos="0.1 0.2 0.3"><freejoint/>
        <inertial pos="0.01 0 0" mass="2" diaginertia="0.02 0.03 0.04"/>
        <site name="p/port" pos="0.02 0.01 -0.03" quat="0.9238795325 0 0 0.3826834324"/>
      </body>
      <body name="s" pos="-0.1 0.2 0.3"><freejoint/>
        <inertial pos="0 0.02 0" mass="3" diaginertia="0.03 0.04 0.05"/>
        <site name="s/port" pos="-0.01 0.02 0.03" quat="0.9659258263 0.2588190451 0 0"/>
      </body>
    </worldbody></mujoco>''')


def test_injection_matches_site_frame_and_preserves_inertia():
    spec = make_spec()
    before = spec.compile()
    pin_offset = np.array([.004, -.003, .002])
    socket_offset = np.array([-.005, .004, -.003])
    cfg = ConnectorGeometry(.002, .035, .0001)
    names = inject_connector_collision(spec, 'p/port', 's/port', cfg,
        pin_center_site=pin_offset, socket_center_site=socket_offset, prefix='test')
    model = spec.compile()
    assert model.nq == before.nq and model.nv == before.nv
    np.testing.assert_array_equal(model.body_mass, before.body_mass)
    np.testing.assert_allclose(model.body_inertia, before.body_inertia, atol=1e-15)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    p = model.site('p/port').id
    rotation = d.site_xmat[p].reshape(3, 3)
    center = d.site_xpos[p] + rotation @ pin_offset
    disc = model.geom(names['pin_disc']).id
    np.testing.assert_allclose(d.geom_xpos[disc], center + rotation @ [0, 0, .001], atol=1e-12)
    np.testing.assert_allclose(d.geom_xmat[disc].reshape(3, 3), rotation, atol=1e-12)
    for i, (x, y) in enumerate(cfg.centers):
        np.testing.assert_allclose(d.geom_xpos[model.geom(names['pins'][i]).id],
            center + rotation @ [x, y, -.001], atol=1e-12)
    s = model.site('s/port').id
    srot = d.site_xmat[s].reshape(3, 3)
    sorigin = d.site_xpos[s] + srot @ socket_offset
    for name in names['socket_pieces']:
        gid = model.geom(name).id
        mid = model.geom_dataid[gid]
        # MuJoCo recentres mesh data at compilation; mesh_pos is that offset.
        np.testing.assert_allclose(d.geom_xpos[gid], sorigin + srot @ model.mesh_pos[mid], atol=1e-10)


@pytest.mark.parametrize('quarter', range(4))
def test_injected_holes_mate_with_rotated_site_and_offsets(quarter):
    spec = make_spec()
    pin_offset = np.array([.004, -.003, .002])
    socket_offset = np.array([-.005, .004, -.003])
    cfg = ConnectorGeometry(.002, .035, .0001)
    names = inject_connector_collision(spec, 'p/port', 's/port', cfg,
        pin_center_site=pin_offset, socket_center_site=socket_offset, prefix='test')
    model = spec.compile()
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    s = model.site('s/port').id
    srot = d.site_xmat[s].reshape(3, 3)
    origin = d.site_xpos[s] + srot @ socket_offset
    p = model.site('p/port').id
    pid = model.body('p').id
    adr = model.jnt_qposadr[model.body_jntadr[pid]]
    local_site_rot = Rotation.from_quat(model.site_quat[p], scalar_first=True).as_matrix()
    desired_site_rot = srot @ Rotation.from_euler('z', quarter*math.pi/2).as_matrix()
    body_rot = desired_site_rot @ local_site_rot.T
    desired_body_pos = origin + srot @ [0, 0, .0005] - body_rot @ (model.site_pos[p] + local_site_rot @ pin_offset)
    d.qpos[adr:adr+3] = desired_body_pos
    d.qpos[adr+3:adr+7] = Rotation.from_matrix(body_rot).as_quat(scalar_first=True)
    mujoco.mj_forward(model, d)
    pieces = [model.geom(name).id for name in names['socket_pieces']]
    for name in names['pins']:
        pin = model.geom(name).id
        distance = min(mujoco.mj_geomDistance(model, d, pin, g, .02, None) for g in pieces)
        assert distance == pytest.approx(.0001, abs=2e-8)


def test_offsets_and_existing_sites_are_required():
    spec = make_spec()
    cfg = ConnectorGeometry(.002, .035, .0001)
    with pytest.raises(ValueError):
        inject_connector_collision(spec, 'missing', 's/port', cfg,
            pin_center_site=(0, 0, 0), socket_center_site=(0, 0, 0), prefix='test')
    with pytest.raises(ValueError):
        inject_connector_collision(spec, 'p/port', 's/port', cfg,
            pin_center_site=(0, 0), socket_center_site=(0, 0, 0), prefix='test')


def test_injection_preserves_inferred_mass_and_inertia():
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
      <body name="a"><freejoint/><geom type="sphere" size=".02" density="1000"/><site name="a/port"/></body>
      <body name="b"><freejoint/><geom type="box" size=".02 .02 .02" density="1000"/><site name="b/port"/></body>
    </worldbody></mujoco>''')
    before = spec.compile()
    inject_connector_collision(spec, 'a/port', 'b/port', ConnectorGeometry(.002, .035, .0001),
        pin_center_site=(0, 0, 0), socket_center_site=(0, 0, 0), prefix='test')
    after = spec.compile()
    np.testing.assert_array_equal(after.body_mass, before.body_mass)
    np.testing.assert_allclose(after.body_inertia, before.body_inertia, atol=1e-15)
