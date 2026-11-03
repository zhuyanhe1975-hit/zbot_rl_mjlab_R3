"""Registered body welds reproduce site welds and can be switched at runtime."""
import math

import mujoco
import numpy as np
import pytest

from zbot_rl_mjlab.tasks.reconfiguration.body_weld import (
    activate_registration, add_body_weld_from_sites,
)


def make_model():
    xml = '''<mujoco><option timestep="0.001" gravity="0 0 0"/>
    <worldbody>
      <body name="a"><freejoint/><geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/></body>
      <body name="b" pos=".5 0 0"><freejoint/><geom type="sphere" size=".02" mass="1" contype="0" conaffinity="0"/>
      <site name="target" pos=".017 -.023 .031" quat=".9238795325 0 .3826834324 0"/></body>
    </worldbody></mujoco>'''
    spec = mujoco.MjSpec.from_string(xml)
    for k in range(4):
        angle = k * math.pi / 2
        # An offset registration origin ensures the CAD translation is preserved.
        spec.body('a').add_site(name=f'source{k}',
            pos=(.05 + .003 * k, -.02 + .002 * k, .04),
            quat=(math.cos(angle / 2), 0, 0, math.sin(angle / 2)))
        spec.add_equality(name=f'site{k}', type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE, name1=f'source{k}', name2='target',
            active=False, data=(*([0.] * 10), 1.), solref=(.005, 1.),
            solimp=(.999, .9999, .001, .5, 2.))
        add_body_weld_from_sites(spec, f'body{k}', f'source{k}', 'target')
    return spec.compile()


def result(model, qpos, qvel, name):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.eq_active[model.equality(name).id] = True
    mujoco.mj_forward(model, data)
    return data


@pytest.mark.parametrize('k', range(4))
def test_site_and_body_weld_dynamics_are_equivalent(k):
    model = make_model()
    qpos = model.qpos0.copy()
    # Away from the exact goal, compare the full solver and force at an offset.
    qpos[:3] = [.03, -.08, .2]
    qpos[3:7] = [.965925826289, 0, .258819045103, 0]
    qpos[7:10] = [.12, .10, .14]
    qpos[10:14] = [.923879532511, .382683432365, 0, 0]
    qvel = np.linspace(-.3, .4, model.nv)
    a = result(model, qpos, qvel, f'site{k}')
    b = result(model, qpos, qvel, f'body{k}')
    assert a.nefc == b.nefc == 6
    frame = np.empty(9)
    mujoco.mju_quat2Mat(frame, model.site_quat[model.site('target').id])
    frame = frame.reshape(3, 3)
    np.testing.assert_allclose(a.efc_pos[:3], b.efc_pos[:3], atol=1e-10)
    np.testing.assert_allclose(a.efc_pos[3:], frame.T @ b.efc_pos[3:], atol=1e-10)
    ja = a.efc_J.reshape(6, model.nv)
    jb = b.efc_J.reshape(6, model.nv)
    np.testing.assert_allclose(ja[:3], jb[:3], atol=1e-10)
    np.testing.assert_allclose(ja[3:], frame.T @ jb[3:], atol=1e-10)
    np.testing.assert_allclose(a.qfrc_constraint, b.qfrc_constraint, atol=1e-7, rtol=1e-9)
    # The source body's generalized force determines its wrench. Convert the
    # root moment to world, then shift it to the original source connector.
    def connector_wrench(d):
        body = model.body('a').id
        site = model.site(f'source{k}').id
        force = d.qfrc_constraint[:3]
        moment = d.xmat[body].reshape(3, 3) @ d.qfrc_constraint[3:6]
        moment -= np.cross(d.site_xpos[site] - d.xpos[body], force)
        return np.r_[force, moment]
    np.testing.assert_allclose(connector_wrench(a), connector_wrench(b), atol=1e-7, rtol=1e-9)


def test_switch_disconnect_reconnect():
    model = make_model()
    data = mujoco.MjData(model)
    names = [f'body{k}' for k in range(4)]
    for k, name in enumerate(names):
        activate_registration(model, data, names, name)
        mujoco.mj_forward(model, data)
        assert data.nefc == 6
        assert [bool(data.eq_active[model.equality(n).id]) for n in names] == [i == k for i in range(4)]
        assert np.linalg.norm(data.qfrc_constraint) > 0
        activate_registration(model, data, names)
        mujoco.mj_forward(model, data)
        assert data.nefc == 0
        assert np.linalg.norm(data.qfrc_constraint) == 0
    with pytest.raises(ValueError):
        activate_registration(model, data, names, 'absent')


@pytest.mark.parametrize('k', range(4))
def test_weld_carries_load_disconnects_and_reconnects(k):
    model = make_model()
    data = mujoco.MjData(model)
    name = f'body{k}'
    names = [f'body{i}' for i in range(4)]
    eq = model.equality(name).id
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, model.eq_data[eq, 6:10])
    # Exactly realize the registered body2 pose relative to the fixed initial a.
    data.qpos[7:10] = model.eq_data[eq, 3:6] - rot.reshape(3, 3) @ model.eq_data[eq, :3]
    data.qpos[10:14] = model.eq_data[eq, 6:10]
    initial = data.qpos.copy()
    activate_registration(model, data, names, name)
    data.xfrc_applied[model.body('b').id, 0] = 5.
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.qfrc_constraint) > 1.
    source = model.site(f'source{k}').id
    target = model.site('target').id
    assert np.linalg.norm(data.site_xpos[source] - data.site_xpos[target]) < 1e-3
    activate_registration(model, data, names)
    mujoco.mj_forward(model, data)
    assert data.nefc == 0
    for _ in range(50):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.site_xpos[source] - data.site_xpos[target]) > 1e-3
    # Simulate safe re-approach to the known registration before reconnection.
    data.qpos[:] = initial
    data.qvel[:] = 0
    activate_registration(model, data, names, name)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.qfrc_constraint) > 1.
    assert np.linalg.norm(data.site_xpos[source] - data.site_xpos[target]) < 1e-3
