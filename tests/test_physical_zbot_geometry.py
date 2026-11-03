"""Independent robot-asset geometry checks; no magnets or FSM are implied."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.physical_zbot_geometry import make_physical_zbot_model


@pytest.fixture(scope='module')
def model():
    # Preserve the cylinder's legacy geometry and distance checks explicitly.
    return make_physical_zbot_model(pin_shape='cylinder')


@pytest.fixture(scope='module')
def tapered_model():
    return make_physical_zbot_model()


def _ids(model, face):
    pins=[model.geom(f'physical_connector_{face}_pin_{i}').id for i in range(4)]
    pieces=[model.geom(f'physical_connector_{face}_socket_piece_{i}').id for i in range(14)]
    shell=model.geom(f'module_{(face+1)%6}/a{(face+1)%6+1}_geom_clipped_shell').id
    disc=model.geom(f'physical_connector_{face}_pin_plate_disc').id
    return pins,pieces,shell,disc


def test_scene_contains_six_pairs_without_legacy_filters(model):
    assert model.neq == model.npair == model.nexclude == 0
    for i in range(6):
        pins, pieces, shell, disc = _ids(model, i)
        assert len(pins)==4 and len(pieces)==14
        source=model.body(f'module_{i}/b{i+1}' if i<5 else 'module_5/foot_1').id
        target=model.body(f'module_{(i+1)%6}/'+('foot_0' if i==5 else ('base' if i==2 else f'a{i+2}'))).id
        assert all(model.geom_bodyid[g]==source for g in [disc,*pins])
        assert all(model.geom_bodyid[g]==target for g in [shell,*pieces])
        for side in 'ab':
            visual=model.geom(f'module_{i}/{side}{i+1}_geom').id
            assert model.geom_contype[visual] == model.geom_conaffinity[visual] == 0


def test_module_mass_and_inertia_remain_unchanged(model,tapered_model):
    from mjlab.scene import Scene
    from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg
    config=reference_env_cfg().scene
    config.spec_fn=lambda spec: None
    baseline=Scene(config,device='cpu').compile()
    for body in (baseline.body(i).name for i in range(baseline.nbody)):
        if body and body.startswith('module_'):
            old=baseline.body(body).id
            for candidate in (model,tapered_model):
                new=candidate.body(body).id
                assert candidate.body_mass[new] == pytest.approx(baseline.body_mass[old],abs=1e-12)
                np.testing.assert_allclose(candidate.body_inertia[new],baseline.body_inertia[old],rtol=0,atol=1e-12)


def test_peg_gap_and_backing_shell_at_original_registration(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for face in range(6):
        pins,pieces,shell,_=_ids(model,face)
        gaps=[min(mujoco.mj_geomDistance(model,data,p,g,.01,None) for g in pieces) for p in pins]
        np.testing.assert_allclose(gaps, .0001, atol=2e-8)
        depths=[mujoco.mj_geomDistance(model,data,p,shell,.01,None) for p in pins]
        # Three pins touch the unmodified remaining shell at full insertion;
        # the fourth has some clearance because the original shell is shaped.
        assert sum(abs(x)<1e-7 for x in depths)>=3
        assert all(x>=-1e-7 for x in depths)


@pytest.mark.parametrize('quarter', range(4))
def test_each_square_pair_allows_quarter_turns(model,quarter):
    for face in range(6):
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        target=(face+1)%6
        site=model.site(f'module_{target}/port_in').id
        axis=data.site_xmat[site].reshape(3,3)[:,2].copy()
        qadr=int(model.jnt_qposadr[model.joint(f'module_{target}/module_root').id])
        qworld=np.empty(4);qresult=np.empty(4)
        mujoco.mju_axisAngle2Quat(qworld,axis,quarter*math.pi/2)
        mujoco.mju_mulQuat(qresult,qworld,data.qpos[qadr+3:qadr+7].copy())
        data.qpos[qadr+3:qadr+7]=qresult
        mujoco.mj_forward(model,data)
        pins,pieces,_,_=_ids(model,face)
        gaps=[min(mujoco.mj_geomDistance(model,data,p,g,.01,None) for g in pieces) for p in pins]
        np.testing.assert_allclose(gaps,.0001,atol=3e-8)


def test_face_and_side_contact_are_not_filtered(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    site=model.site('module_1/port_in').id
    rot=data.site_xmat[site].reshape(3,3).copy()
    qadr=int(model.jnt_qposadr[model.joint('module_0/module_root').id])
    data.qpos[qadr:qadr+3]+=rot[:,2]*.0001
    mujoco.mj_forward(model,data)
    pins,pieces,_,disc=_ids(model,0)
    assert min(mujoco.mj_geomDistance(model,data,disc,g,.01,None) for g in pieces)<-1e-6
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    data.qpos[qadr:qadr+3]+=rot[:,0]*.0003
    mujoco.mj_forward(model,data)
    assert min(mujoco.mj_geomDistance(model,data,p,g,.01,None) for p in pins for g in pieces)<-1e-5
    assert any(({int(c.geom1),int(c.geom2)}&set(pins)) and
               ({int(c.geom1),int(c.geom2)}&set(pieces)) for c in data.contact)


def test_tapered_pins_share_mesh_and_follow_each_interface(tapered_model):
    model=tapered_model
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    pins=[model.geom(f'physical_connector_{face}_pin_{i}').id
          for face in range(6) for i in range(4)]
    assert set(model.geom_type[pins])=={mujoco.mjtGeom.mjGEOM_MESH}
    assert len(set(model.geom_dataid[pins]))==1
    for face in range(6):
        frame=model.site(f'module_{face}/physical_out_frame').id
        frame_rotation=data.site_xmat[frame].reshape(3,3)
        for i in range(4):
            geom=model.geom(f'physical_connector_{face}_pin_{i}').id
            mesh=int(model.geom_dataid[geom]); adr=model.mesh_vertadr[mesh]; n=model.mesh_vertnum[mesh]
            vertices=model.mesh_vert[adr:adr+n] @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]
            local=(vertices-data.site_xpos[frame]) @ frame_rotation
            xy_center=(local[:,:2].max(axis=0)+local[:,:2].min(axis=0))/2
            radius=np.linalg.norm(local[:,:2]-xy_center,axis=1)
            tip=np.isclose(local[:,2],-.002,atol=1e-8)
            base=np.isclose(local[:,2],0.,atol=1e-8)
            assert tip.any() and base.any()
            np.testing.assert_allclose(radius[tip],.0018,atol=2e-9)
            np.testing.assert_allclose(radius[base],.002,atol=2e-9)


@pytest.mark.parametrize('quarter', range(4))
def test_tapered_pins_clear_holes_at_all_quarter_turns(tapered_model,quarter):
    model=tapered_model
    for face in range(6):
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        target=(face+1)%6
        frame=model.site(f'module_{target}/port_in').id
        axis=data.site_xmat[frame].reshape(3,3)[:,2].copy()
        qadr=int(model.jnt_qposadr[model.joint(f'module_{target}/module_root').id])
        qworld=np.empty(4);qresult=np.empty(4)
        mujoco.mju_axisAngle2Quat(qworld,axis,quarter*math.pi/2)
        mujoco.mju_mulQuat(qresult,qworld,data.qpos[qadr+3:qadr+7].copy())
        data.qpos[qadr+3:qadr+7]=qresult
        mujoco.mj_forward(model,data)
        pins,pieces,_,_=_ids(model,face)
        assert not any(({int(c.geom1),int(c.geom2)} & set(pins)) and
                       ({int(c.geom1),int(c.geom2)} & set(pieces))
                       for c in data.contact)


def test_tapered_nearest_distance_may_report_false_zero(tapered_model):
    model=tapered_model;data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    target=model.site('module_1/port_in').id
    axis=data.site_xmat[target].reshape(3,3)[:,2].copy()
    adr=int(model.jnt_qposadr[model.joint('module_0/module_root').id])
    data.qpos[adr:adr+3]-=axis*.0005
    mujoco.mj_forward(model,data)
    pins,pieces,_,_=_ids(model,0)
    raw=[min(mujoco.mj_geomDistance(model,data,p,g,.01,None) for g in pieces) for p in pins]
    assert not any(({int(c.geom1),int(c.geom2)} & set(pins)) and
                   ({int(c.geom1),int(c.geom2)} & set(pieces)) for c in data.contact)
    # MuJoCo 3.11's mesh/mesh query returns a false zero for this known case;
    # a fixed backend may instead return the physically positive gap.
    assert all(value>=-1e-9 for value in raw)
    assert any(value>.0001 for value in raw)


def test_taper_profile_validation():
    with pytest.raises(ValueError):make_physical_zbot_model(pin_tip_radius_m=.003)
    with pytest.raises(ValueError):make_physical_zbot_model(pin_sides=10)


def test_taper_profile_parameters_are_respected():
    model=make_physical_zbot_model(pin_tip_radius_m=.001, pin_sides=8)
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    geom=model.geom('physical_connector_0_pin_0').id
    mesh=int(model.geom_dataid[geom]);adr=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh]
    vertices=model.mesh_vert[adr:adr+n] @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]
    frame=model.site('module_0/physical_out_frame').id
    local=(vertices-data.site_xpos[frame]) @ data.site_xmat[frame].reshape(3,3)
    tip=local[np.isclose(local[:,2],-.002,atol=1e-8)]
    center=(tip[:,:2].max(axis=0)+tip[:,:2].min(axis=0))/2
    np.testing.assert_allclose(np.linalg.norm(tip[:,:2]-center,axis=1),.001,atol=2e-9)
