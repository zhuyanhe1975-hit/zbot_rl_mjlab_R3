"""Mechanical regression for an *uncalibrated* passive-adhesion experiment."""
import math
import mujoco
import numpy as np
import pytest

from scripts.diagnose_native_magnet_adhesion import make_model, probe, run


@pytest.fixture(scope='module')
def model():
    return make_model()


@pytest.mark.parametrize('quarter', range(4))
def test_four_local_proxy_contacts_and_no_foreign_contact(model, quarter):
    result=probe(model,.0005,quarter*math.pi/2)
    assert result['proxy_contacts']==4
    assert not result['unexpected_proxy_contacts']
    np.testing.assert_allclose(result['source_adhesion_force_N'],[0.,0.,20.],atol=1e-9)
    np.testing.assert_allclose(result['target_adhesion_force_N'],[0.,0.,-20.],atol=1e-9)


def test_finite_gap_is_constant_then_discontinuous(model):
    far=probe(model,.0031)
    middle=probe(model,.002)
    near=probe(model,.0005)
    assert far['proxy_contacts']==0 and middle['proxy_contacts']==near['proxy_contacts']==4
    assert far['source_adhesion_force_N'][2]==pytest.approx(0.)
    assert middle['source_adhesion_force_N'][2]==pytest.approx(20.)
    assert near['source_adhesion_force_N'][2]==pytest.approx(20.)
    assert probe(model,.0005,sideways_m=.006)['proxy_contacts']==0


def test_aligned_approach_remains_finite():
    report=run(duration=.1)
    assert report['finite'] and report['warnings']==0
    assert abs(report['local_gap_mm'])<.01
    assert report['physical_penetration_mm']<.01


@pytest.mark.parametrize('force,connected',[(10.,True),(25.,False)])
def test_pull_off_of_four_contacts(model,force,connected):
    data=mujoco.MjData(model)
    a=model.body('socket').id;b=model.body('pin_plate').id
    q1=int(model.jnt_qposadr[model.body_jntadr[a]])
    q2=int(model.jnt_qposadr[model.body_jntadr[b]])
    data.xfrc_applied[a,2]=-force;data.xfrc_applied[b,2]=force
    mujoco.mj_forward(model,data)
    mujoco.mj_step(model,data,nstep=300)
    gap=float(data.qpos[q2+2]-data.qpos[q1+2])
    if connected:assert gap<.003
    else:assert gap>.003
    assert not np.any(data.warning.number)
