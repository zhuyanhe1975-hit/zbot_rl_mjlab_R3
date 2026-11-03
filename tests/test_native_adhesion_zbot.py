"""Geom adhesion isolation and contact semantics; not magnet calibration."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.native_adhesion import make_native_adhesion_zbot_model
from zbot_mjbatch.physical_zbot_geometry import make_physical_zbot_model


@pytest.fixture(scope='module')
def model():
    return make_native_adhesion_zbot_model()


def _ids(model,face,side):
    return [model.geom(f'physical_connector_{face}_adhesion_{side}_{j}').id for j in range(4)]


def _allowed(model,a,b):
    return bool((model.geom_contype[a]&model.geom_conaffinity[b]) or
                (model.geom_contype[b]&model.geom_conaffinity[a]))


def _contact_pairs(data):
    return [(int(c.geom1),int(c.geom2),c) for c in data.contact]


def test_proxy_masks_isolate_all_other_geometries(model):
    proxies={g for i in range(6) for side in ('source','target') for g in _ids(model,i,side)}
    assert len(proxies)==48 and model.flg_adhesion
    for face in range(6):
        source=_ids(model,face,'source');target=_ids(model,face,'target')
        for a in source:
            assert model.geom_adhesion[a] == pytest.approx(5.)
            assert model.geom_gap[a] == pytest.approx(.003)
            assert model.geom_condim[a] == 1
            for b in range(model.ngeom):
                assert _allowed(model,a,b) == (b in target)
        for b in target:
            assert model.geom_adhesion[b] == 0.
            for a in range(model.ngeom):
                assert _allowed(model,a,b) == (a in source)
        np.testing.assert_allclose(model.geom_size[source+target,0],.001)


@pytest.mark.parametrize('quarter',range(4))
def test_all_four_registered_proxy_pairs_detected(model,quarter):
    for face in range(6):
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        target=(face+1)%6;site=model.site(f'module_{target}/port_in').id
        axis=data.site_xmat[site].reshape(3,3)[:,2].copy()
        qadr=int(model.jnt_qposadr[model.joint(f'module_{target}/module_root').id])
        turn=np.empty(4);quat=np.empty(4)
        mujoco.mju_axisAngle2Quat(turn,axis,quarter*math.pi/2)
        mujoco.mju_mulQuat(quat,turn,data.qpos[qadr+3:qadr+7].copy())
        data.qpos[qadr+3:qadr+7]=quat
        mujoco.mj_forward(model,data)
        src=set(_ids(model,face,'source'));dst=set(_ids(model,face,'target'))
        matched=[c for a,b,c in _contact_pairs(data) if ({a,b}&src) and ({a,b}&dst)]
        assert len(matched)==4
        assert all(c.adhesion==pytest.approx(5.) for c in matched)
        assert all(c.dim==1 for c in matched)
        assert all(abs(c.dist)<1e-8 for c in matched)


def test_gap_attraction_is_capped_per_contact_and_never_custom_force():
    # Disable other geometry and actuation only in this isolated experiment;
    # this is not a validated whole-robot state.
    model=make_native_adhesion_zbot_model();model.opt.gravity[:]=0
    model.actuator_gainprm[:]=0;model.actuator_biasprm[:]=0
    src=set(_ids(model,0,'source'));dst=set(_ids(model,0,'target'))
    for g in range(model.ngeom):
        if g not in src|dst:model.geom_contype[g]=model.geom_conaffinity[g]=0
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    axis=data.site_xmat[model.site('module_1/port_in').id].reshape(3,3)[:,2].copy()
    qadr=int(model.jnt_qposadr[model.joint('module_0/module_root').id])
    data.qpos[qadr:qadr+3]-=.002*axis
    mujoco.mj_forward(model,data)
    matches=[(k,c) for k,c in enumerate(data.contact) if
             ({int(c.geom1),int(c.geom2)}&src) and ({int(c.geom1),int(c.geom2)}&dst)]
    assert len(matches)==4
    for k,c in matches:
        assert c.adhesion==pytest.approx(5.) and c.dist==pytest.approx(.002,abs=1e-8)
        wrench=np.empty(6);mujoco.mj_contactForce(model,data,k,wrench)
        assert -5.000001<=wrench[0]<0
    assert not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied)
    # Move 4 mm from the face; the 3 mm interaction band is left.
    data.qpos[qadr:qadr+3]-=.002*axis;mujoco.mj_forward(model,data)
    assert not any(({int(c.geom1),int(c.geom2)}&src) and
                   ({int(c.geom1),int(c.geom2)}&dst) for c in data.contact)


def test_no_other_proxy_contacts_in_original_state(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    all_proxies={g for i in range(6) for side in ('source','target') for g in _ids(model,i,side)}
    for a,b,_ in _contact_pairs(data):
        if a not in all_proxies and b not in all_proxies:continue
        assert any((a in _ids(model,face,'source') and b in _ids(model,face,'target')) or
                   (b in _ids(model,face,'source') and a in _ids(model,face,'target')) for face in range(6))


def test_adhesion_proxies_do_not_change_inertia(model):
    reference=make_physical_zbot_model()
    for body in (reference.body(i).name for i in range(reference.nbody)):
        if body and body.startswith('module_'):
            a=reference.body(body).id;b=model.body(body).id
            assert model.body_mass[b]==pytest.approx(reference.body_mass[a],abs=1e-12)
            np.testing.assert_allclose(model.body_inertia[b],reference.body_inertia[a],atol=1e-12,rtol=0)
