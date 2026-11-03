import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.docking_sites import (
    DockingLandmarks, add_connector_ring_sites, correspondence, site_names,
)


def model_with_connectors():
    bodies = []
    for i in range(6):
        bodies.append(f'''<body name="module_{i}">
            <freejoint name="free_{i}"/>
            <inertial mass="1" diaginertia=".1 .1 .1" pos="0 0 0"/>
            <site name="module_{i}/port_in" pos="-.00022282 -.00010463 0"/>
            <site name="module_{i}/port_out" pos="-.00022282 -.00010463 0"/>
        </body>''')
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody>' + ''.join(bodies) + '</worldbody></mujoco>')
    add_connector_ring_sites(spec)
    return spec.compile()


@pytest.mark.parametrize('quarter', range(4))
def test_quarter_turn_matches_real_sites(quarter):
    model = model_with_connectors()
    data = mujoco.MjData(model)
    q = model.jnt_qposadr[model.joint('free_0').id]
    angle = quarter * math.pi / 2
    data.qpos[q+3:q+7] = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
    mujoco.mj_forward(model, data)
    points = DockingLandmarks.from_model(model, 0)
    assert np.max(np.abs(points.errors(data, quarter))) < 1e-12
    assert np.linalg.norm(points.errors(data, (quarter + 1) % 4)) > .1
    assert model.ngeom == 0
    assert len(set(site_names('module_0/port_out'))) == 4


def test_actual_jacobian_matches_integrated_finite_difference():
    model = model_with_connectors()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(12)
    for i in range(2):
        q = model.jnt_qposadr[model.joint(f'free_{i}').id]
        data.qpos[q:q+3] = rng.normal(size=3)
        quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
        data.qpos[q+3:q+7] = quat
    mujoco.mj_forward(model, data)
    points = DockingLandmarks.from_model(model, 0)
    jac = points.jacobian(model, data, 1)
    residual = points.errors(data, 1).reshape(-1).copy()
    eps = 1e-7
    original = data.qpos.copy()
    for col in range(model.nv):
        direction = np.zeros(model.nv); direction[col] = 1
        data.qpos[:] = original
        mujoco.mj_integratePos(model, data.qpos, direction, eps)
        mujoco.mj_forward(model, data)
        derivative = (points.errors(data, 1).reshape(-1) - residual) / eps
        np.testing.assert_allclose(jac[:, col], derivative, atol=2e-8)
    np.testing.assert_array_equal(points.jacobian(model, data, 1, [2, 0]).shape, (12, 2))


def test_bad_values_are_rejected():
    with pytest.raises(TypeError):
        correspondence(1.0)
    with pytest.raises(ValueError):
        add_connector_ring_sites(mujoco.MjSpec(), radius=0)


def test_real_model_registration_offsets_and_mixed_variants():
    # Test against the actual mixed model and its CAD-offset weld definitions.
    from mjlab.scene import Scene
    from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import (
        add_reference_connectors, reference_env_cfg,
    )

    def build(spec):
        add_reference_connectors(spec)
        spec.add_exclude(bodyname1='module_5/foot_1', bodyname2='module_0/foot_0')
        original = spec.site('module_5/port_out')
        original.parent.add_site(name='module_5/search_port', pos=original.pos,
                                 quat=original.quat, size=(.004,)*3)
        add_connector_ring_sites(spec)

    cfg = reference_env_cfg().scene
    cfg.spec_fn = build
    model = Scene(cfg, device='cpu').compile()

    def move_weld_sites_together(connector, source_name, quarter):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        a = model.site(source_name).id
        b = model.site(f'module_{(connector + 1) % 6}/port_in').id
        rotation = data.site_xmat[b].reshape(3, 3) @ data.site_xmat[a].reshape(3, 3).T
        translation = data.site_xpos[b] - rotation @ data.site_xpos[a]
        joint = model.joint(f'module_{connector}/module_root').id
        root = model.jnt_bodyid[joint]
        q = model.jnt_qposadr[joint]
        data.qpos[q:q+3] = rotation @ data.xpos[root] + translation
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (rotation @ data.xmat[root].reshape(3, 3)).reshape(9))
        data.qpos[q+3:q+7] = quat
        mujoco.mj_forward(model, data)
        assert np.max(np.abs(DockingLandmarks.from_model(model, connector).errors(data, quarter))) < 1e-9

    for i in range(5):
        # mixed model interfaces 3 and 4 reverse the quarter-turn sign.
        move_weld_sites_together(i, f'module_{i}/port_out_rz270', 3 if i < 3 else 1)

    old = model.site('module_5/port_out').id
    search = model.site('module_5/search_port').id
    inlet = model.site_pos[model.site('module_0/port_in').id]
    old_rotation = np.empty(9)
    mujoco.mju_quat2Mat(old_rotation, model.site_quat[old])
    for angle in (0, 90, 180, 270):
        delta = math.radians(angle - 180)
        turn = np.array([[math.cos(delta), -math.sin(delta), 0],
                         [math.sin(delta), math.cos(delta), 0], [0, 0, 1]])
        model.site_pos[search] = model.site_pos[old] + old_rotation.reshape(3, 3) @ (turn @ inlet - inlet)
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, model.site_quat[old],
                           np.array([math.cos(delta/2), 0., 0., math.sin(delta/2)]))
        model.site_quat[search] = quat
        move_weld_sites_together(5, 'module_5/search_port', (-(angle - 180) // 90) % 4)
