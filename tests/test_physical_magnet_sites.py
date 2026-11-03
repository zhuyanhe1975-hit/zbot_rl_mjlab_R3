"""Actual-asset ring landmarks; no magnetic force or calibration is implied."""
import math

import mujoco
import numpy as np
import pytest

from zbot_mjbatch.physical_zbot_geometry import make_physical_zbot_model


@pytest.fixture(scope='module')
def model():
    return make_physical_zbot_model()


def _sites(model,face,label):
    return np.array([model.site(f'physical_connector_{face}_magnet_{label}_{j}').id for j in range(4)])


def test_six_interfaces_own_distinct_magnets_not_pegs(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    all_ids=[]
    for face in range(6):
        src=_sites(model,face,'source');dst=_sites(model,face,'target')
        all_ids.extend(src);all_ids.extend(dst)
        expected_source=model.geom(f'physical_connector_{face}_pin_0').id
        expected_target=model.geom(f'physical_connector_{face}_socket_piece_0').id
        assert np.all(model.site_bodyid[src] == model.geom_bodyid[expected_source])
        assert np.all(model.site_bodyid[dst] == model.geom_bodyid[expected_target])
        np.testing.assert_allclose(np.linalg.norm(data.site_xpos[src]-data.site_xpos[dst],axis=1),.002,atol=2e-10)
        # Magnet landmarks are on a 35 mm ring, halfway between the four peg
        # positions, with 1 mm of embedded depth, not coincident with a pin.
        frame=model.site(f'module_{face}/physical_out_frame').id
        inlet=model.site(f'module_{(face+1)%6}/port_in').id
        offset=np.diag([1.,-1.,-1.]) @ -model.site_pos[inlet]
        local=(data.site_xpos[src]-data.site_xpos[frame]) @ data.site_xmat[frame].reshape(3,3)-offset
        angles=np.arange(4)*math.pi/2+math.pi/4
        expected=np.c_[.035*np.cos(angles),.035*np.sin(angles),np.full(4,.001)]
        np.testing.assert_allclose(local,expected,atol=2e-10)
        source_geom_positions=np.array([data.geom_xpos[model.geom(f'physical_connector_{face}_pin_{j}').id] for j in range(4)])
        assert np.linalg.norm(data.site_xpos[src,None,:]-source_geom_positions[None,:,:],axis=-1).min()>.01
        frame=model.site(f'module_{(face+1)%6}/physical_in_frame').id
        local=(data.site_xpos[dst]-data.site_xpos[frame]) @ data.site_xmat[frame].reshape(3,3)-offset
        expected[:,2]=-.001
        np.testing.assert_allclose(local,expected,atol=2e-10)
    assert len(set(map(int,all_ids)))==48


@pytest.mark.parametrize('quarter',range(4))
def test_rotated_registration_reshuffles_but_retains_four_pairs(model,quarter):
    for face in range(6):
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        target=(face+1)%6
        inlet=model.site(f'module_{target}/port_in').id
        axis=data.site_xmat[inlet].reshape(3,3)[:,2].copy()
        qadr=int(model.jnt_qposadr[model.joint(f'module_{target}/module_root').id])
        rotation=np.empty(4);result=np.empty(4)
        mujoco.mju_axisAngle2Quat(rotation,axis,quarter*math.pi/2)
        mujoco.mju_mulQuat(result,rotation,data.qpos[qadr+3:qadr+7].copy())
        data.qpos[qadr+3:qadr+7]=result
        mujoco.mj_forward(model,data)
        a=data.site_xpos[_sites(model,face,'source')];b=data.site_xpos[_sites(model,face,'target')]
        distance=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
        np.testing.assert_allclose(distance.min(axis=1),.002,atol=3e-9)
        np.testing.assert_allclose(distance.min(axis=0),.002,atol=3e-9)
        assert np.all(np.sum(distance<.00201,axis=1)==1)


def test_configurable_magnets():
    model=make_physical_zbot_model(magnet_radius_m=.025,magnet_embed_depth_m=.0005,magnet_phase_rad=0.)
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    src=_sites(model,0,'source');dst=_sites(model,0,'target')
    np.testing.assert_allclose(np.linalg.norm(data.site_xpos[src]-data.site_xpos[dst],axis=1),.001,atol=2e-10)
    mean=data.site_xpos[src].mean(axis=0)
    np.testing.assert_allclose(np.linalg.norm(data.site_xpos[src]-mean,axis=1),.025,atol=2e-10)


@pytest.mark.parametrize('kwargs',[
    {'magnet_radius_m':0.}, {'magnet_radius_m':.05},
    {'magnet_embed_depth_m':-.001}, {'magnet_embed_depth_m':.003},
    {'magnet_phase_rad':float('nan')},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):make_physical_zbot_model(**kwargs)
