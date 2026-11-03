"""Standalone mjbatch backend: wiring, safety measurements and episode reset."""
import sys
import numpy as np
import pytest
import mujoco
from mjbatch import Batch
from zbot_mjbatch.env import HeadTailBatch


def test_actual_batch_without_mjlab_and_independent_reset():
    env=HeadTailBatch(2,2)
    try:
        assert isinstance(env.batch,Batch)
        assert not any(n=='mjlab' or n.startswith('mjlab.') for n in sys.modules)
        assert env.observation().shape==(2,40)
        assert env.policy_active_observation_index==31
        assert env.model.nsensor==4
        np.testing.assert_allclose(env.model.actuator_gainprm[env.actuators,0],15.)
        np.testing.assert_allclose(env.model.actuator_biasprm[env.actuators,1:3],np.tile([-15.,-1.5],(6,1)))
        assert env.speed_limit==pytest.approx(2*np.pi) and not hasattr(env,'acceleration_limit')
        delta,rot,R=env.geometry();velocity=env.plan(delta,rot,R)
        assert np.max(np.abs(velocity))>.075*env.dt
        for _ in range(3):
            obs,reward,done,info=env.step(np.zeros((2,7)))
            assert not done.any() and np.all(np.isfinite(obs))
            np.testing.assert_allclose(reward,-.004,atol=1e-7)
        state=env.state[1].copy();env.reset([0])
        np.testing.assert_array_equal(env.state[1],state)
        assert env.elapsed[0]==0 and env.elapsed[1]==pytest.approx(.06)
        assert env.eq[0,env.weld[:5]].all() and not env.eq[0,env.weld[5]]
    finally:env.close()


def test_native_weld_load_monitor_matches_direct_mujoco():
    env=HeadTailBatch(1,1)
    try:
        env.monitor.measure(env.state,env.audit);d=mujoco.MjData(env.model)
        mujoco.mj_setState(env.model,d,env.state[0],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(env.model,d)
        eq=env.weld[0];joint=env.model.joint('module_0/module_root').id
        base=int(env.model.jnt_dofadr[joint]);body=int(env.model.jnt_bodyid[joint]);site=env.model.site('module_0/port_out').id
        ids=np.flatnonzero((d.efc_type[:d.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id[:d.nefc]==eq))
        assert len(ids)==6
        g=d.efc_J.reshape(-1,env.model.nv)[ids,base:base+6].T@d.efc_force[ids]
        R=d.xmat[body].reshape(3,3);P=d.site_xmat[site].reshape(3,3);arm=d.site_xpos[site]-d.xpos[body]
        force=P.T@g[:3];moment=P.T@(R@g[3:]-np.cross(arm,g[:3]));peel=max(force[2],0.)+np.linalg.norm(moment[:2])/.05
        assert env.audit[0,0]==pytest.approx(peel,abs=1e-8)
        assert env.audit[0,9]==0
    finally:env.close()


def test_contact_does_not_end_episode_as_collision():
    env=HeadTailBatch(2,2)
    try:
        m=env.model
        head=int(m.joint('module_0/module_root').qposadr[0]);tail=int(m.joint('module_5/module_root').qposadr[0])
        env.qpos[0,tail:tail+7]=env.qpos[0,head:head+7];env.batch.forward()
        _,reward,done,info=env.step(np.zeros((2,7)))
        assert info['reason'][0] not in (2,6)
        assert env.elapsed[1]==pytest.approx(.02) and not info['failed'][1]
    finally:env.close()

def test_optional_mpc_prediction_uses_independent_exact_state():
    from zbot_mjbatch.mpc import PredictiveController
    env=HeadTailBatch(1,1);mpc=PredictiveController(1,1,2,1)
    try:
        action=np.zeros((1,7))
        for _ in range(14):env.step(action)
        before=env.state.copy();selected,info=mpc.select(env,action)
        np.testing.assert_array_equal(env.state,before)
        assert info['safe_fraction']==1 and np.all(np.isfinite(selected))
        env.step(selected)
        np.testing.assert_allclose(env.state[0],mpc.prediction.state[0],rtol=0,atol=1e-9)
    finally:mpc.close();env.close()

def test_mujoco_warning_is_a_failure_and_does_not_end_other_worlds():
    env=HeadTailBatch(2,2)
    try:
        env.warnings[0,int(mujoco.mjtWarning.mjWARN_BADQACC),1]+=1
        _,reward,done,info=env.step(np.zeros((2,7)))
        assert done.tolist()==[True,False]
        assert info['reason'][0]==7 and reward[0]<-19.9
        assert env.elapsed[0]==0 and env.elapsed[1]==pytest.approx(.02)
    finally:env.close()

def test_large_position_request_is_torque_limited_not_an_acceleration_failure():
    env=HeadTailBatch(1,1)
    try:
        env.qtarget[0]=env.qpos[0,env.qids]+2.
        _,_,_,info=env.step(np.zeros((1,7)))
        assert info['reason'][0]!=3
        assert info['peak_torque_Nm'][0]<=20.00001
        assert info['peak_torque_Nm'][0]>14.
    finally:env.close()

def test_play_prefers_validated_best_and_falls_back_to_latest(tmp_path):
    import hashlib
    import json
    from play_mjbatch import select_checkpoint
    for name in ('model_9.pt','model_10.pt','model_100.pt'):(tmp_path/name).touch()
    path,notice=select_checkpoint(tmp_path)
    assert path.name=='model_100.pt' and '尚未经过独立验证' in notice
    report=tmp_path/'deterministic_evaluation.json';report.write_text('{"selected": null}')
    path,notice=select_checkpoint(tmp_path)
    assert path.name=='model_100.pt' and '尚无通过' in notice
    best=tmp_path/'best.pt';best.write_bytes(b'checked')
    report.write_text(json.dumps({'selected':{'sha256':hashlib.sha256(best.read_bytes()).hexdigest()}}))
    assert select_checkpoint(tmp_path)==(best,None)
    best.write_bytes(b'changed');path,notice=select_checkpoint(tmp_path)
    assert path.name=='model_100.pt' and '校验不匹配' in notice
