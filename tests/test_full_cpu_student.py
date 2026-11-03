import numpy as np
import pytest
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import FullCpuStudent, MOVING

@pytest.fixture
def env():
    e=FullCpuStudent()
    try:yield e
    finally:e.close()

def test_observation_and_zero_action(env):
    obs=env.observation()
    assert np.isfinite(obs).all()
    nxt,reward,done,info=env.step(np.zeros(7))
    assert nxt.shape==obs.shape and np.isfinite(nxt).all()
    assert np.isfinite(reward) and not done and info['phase']=='INITIAL_STRAIGHT_HOLD'

def test_policy_cannot_modify_loading_or_holds(env):
    f=env.fsm;env.action[:]=1.
    f.phase='INITIAL_STRAIGHT_HOLD';f.set_pd(outer=True);f.qref[:]=f.data.qpos[f.qadr]
    env._apply();hold=f.data.ctrl[f.actuators].copy()
    env.action[:]=0.;env._apply();np.testing.assert_allclose(hold,f.data.ctrl[f.actuators])
    f.phase='CURL';env.action[:]=1.;env._apply();motion=f.data.ctrl[f.actuators].copy()
    env.action[:]=0.;env._apply();np.testing.assert_allclose(motion-f.data.ctrl[f.actuators],env.scale)
    assert 'LOAD_HEADTAIL' not in MOVING

def test_reference_matches_phase_target_stage_and_time(tmp_path):
    from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES
    path=tmp_path/'teacher.npz'
    np.savez(path,phase_names=np.array(PHASES),phase=np.array([0,0,1]),target=np.array([5,5,5]),stage=np.zeros(3,dtype=int),phase_time=np.array([0.,.2,0.]),q=np.zeros((3,6)),qvel=np.zeros((3,6)),qref=np.zeros((3,6)))
    e=FullCpuStudent(reference=path)
    try:
        e.fsm.data.time=.19
        assert e.reference_sample()==1
        assert np.isfinite(e.observation()).all()
        e.fsm.stage=2
        assert e.reference_sample() is None
    finally:e.close()

def test_student_speed_changes_curve_clock_but_keeps_physical_event_time(env):
    f=env.fsm;assert f.cfg.other_stop_load==80.
    f.phase='CURL';f.phase_start=0.;f.data.time=1.;f.motion_offset=.5
    assert f.phase_time==pytest.approx(1.5)
    f.transition('CAPTURE_HEADTAIL')
    assert f.events[-1]['elapsed_s']==pytest.approx(1.)
    assert f.phase_time==pytest.approx(0.)

def test_non_target_face_uses_80n_break_reset(env):
    f=env.fsm;d=f.data;m=f.model
    f.target=5
    rows=np.flatnonzero((d.efc_type==0)&(d.efc_id==f.old[0]))
    J=d.efc_J.reshape(d.nefc,m.nv)[rows,f.root_dofs[0]:f.root_dofs[0]+6]
    R=d.site_xmat[f.out[0]].reshape(3,3);rootR=d.xmat[f.root_bodies[0]].reshape(3,3)
    arm=d.site_xpos[f.out[0]]-d.xpos[f.root_bodies[0]]
    def set_load(n):
        force=R[:,2]*n
        generalized=np.r_[force,rootR.T@np.cross(arm,force)]
        d.efc_force[rows]=np.linalg.solve(J.T,generalized)
    set_load(70.);f.measure()
    assert f.phase!='FAILED' and f.connected()[0]
    set_load(81.);f.measure()
    assert f.phase=='FAILED' and not f.connected()[0]
