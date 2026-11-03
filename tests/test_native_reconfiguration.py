"""Physical topology and execution smoke checks; not a trajectory safety proof."""
import json
import numpy as np
import pytest
import mujoco
from zbot_mjbatch.native_reconfiguration import NativeReconfiguration, Config, _solve_coupled_response, _solve_retained_load_response

@pytest.fixture(scope='module')
def experiment():
    return NativeReconfiguration()


def test_real_straight_topology_without_weld(experiment):
    x=experiment
    assert x.model.neq==0
    assert [v['contacts'] for v in x.initial_status]==[4,4,4,4,4,0]
    assert all(v['seated'] for v in x.initial_status[:5])
    assert not x.initial_status[5]['attracting']
    assert x.initial_status[5]['position_m']>0.6
    np.testing.assert_allclose(x.model.actuator_gainprm[:,0],15.)
    np.testing.assert_allclose(x.model.actuator_biasprm[:,2],-1.5)
    assert x.model.opt.timestep==.001
    assert np.isfinite(x.closed_q).all()
    assert np.isfinite(x.curl_goal_q).all()
    assert np.linalg.norm(x.curl_goal_q-x.closed_q)>.1


def test_contact_driven_motion_and_report(experiment):
    x=experiment
    for _ in range(1000):
        x.step()
        if x.phase in ('DONE','FAILED'):break
    assert x.audit.samples==x.step_count+1
    if x.audit.first_violation is not None:
        assert x.phase=='FAILED'
        assert x.failure.startswith('safety audit:')
    else:
        assert x.audit.total_violations==0
    assert not x.report()['success']
    assert np.isfinite(x.data.qpos).all()
    assert x.peak_torque<=20.+1e-6
    json.dumps(x.report())


def test_control_period_validation():
    with pytest.raises(ValueError,match='control period'):
        NativeReconfiguration(Config(dt=.003,control_dt=.02))


@pytest.mark.parametrize('phase', ['CURL','CAPTURE','SETTLE','RELEASE','SEPARATE','ROTATE','APPROACH','UNFOLD'])
def test_motion_does_not_fail_just_because_phase_took_long(monkeypatch,phase):
    x=NativeReconfiguration();x.phase=phase;x.target=5 if phase=='CAPTURE' else 0;x.phase_start=-1000.
    x.rotation_anchor=np.zeros(3)
    monkeypatch.setattr(x,'_local_control',lambda *args:None)
    monkeypatch.setattr(x,'_load_control',lambda *args:None)
    monkeypatch.setattr(x,'_approach_control',lambda *args:None)
    x._advance()
    assert x.phase!='FAILED' and not x.failure


def test_target_quarter_proceeds_to_prealignment_without_precision_gate(monkeypatch):
    x=NativeReconfiguration();x.phase='ROTATE';x.target=0;x.rotation_anchor=np.zeros(3)
    statuses=[dict(item) for item in x.interfaces()]
    statuses[0].update(quarter=3,attracting=False,angle_rad=.1)
    monkeypatch.setattr(x,'interfaces',lambda:statuses)
    monkeypatch.setattr(x,'_local_control',lambda *args:None)
    x._advance()
    assert x.phase=='APPROACH' and not x.axial_armed


def test_batch_prediction_does_not_mutate_live_physics(experiment):
    x=experiment
    before=x.data.qpos.copy(); velocity=x.data.qvel.copy();t=x.data.time
    x._local_control(5,np.array([0.,0.,.02]),np.zeros(3))
    np.testing.assert_array_equal(x.data.qpos,before)
    np.testing.assert_array_equal(x.data.qvel,velocity)
    assert x.data.time==t
    assert x.prediction_calls==1
    assert x.prediction_rank is not None
    assert np.isfinite(x.qref).all()


def test_proxy_adhesion_reconstructs_engine_generalized_force():
    x=NativeReconfiguration(); m=x.model; d=x.data
    mujoco.mj_forward(m,d); predicted=np.zeros(m.nv); zero=np.zeros(3)
    for contact in d.contact:
        a=x.proxies.get(int(contact.geom1)); b=x.proxies.get(int(contact.geom2))
        if not (a and b and a[0]==b[0] and a[1]!=b[1]):continue
        force=x.cfg.contact_force_n*contact.frame[:3]
        mujoco.mj_applyFT(m,d,force,zero,contact.pos,int(m.geom_bodyid[contact.geom1]),predicted)
        mujoco.mj_applyFT(m,d,-force,zero,contact.pos,int(m.geom_bodyid[contact.geom2]),predicted)
    np.testing.assert_allclose(predicted,d.qfrc_adhesion,atol=1e-9)


def test_velocity_jacobian_includes_floating_roots_and_moving_frame():
    x=NativeReconfiguration();m=x.model;d=x.audit.scratch;face=5
    src,dst,offset=x.audit.frames[face]
    r0=d.site_xmat[src].reshape(3,3);r1=d.site_xmat[dst].reshape(3,3)
    p0=d.site_xpos[src]+r0@offset;p1=d.site_xpos[dst]+r1@offset
    jp0=np.zeros((3,m.nv));jr0=np.zeros((3,m.nv));jp1=np.zeros((3,m.nv));jr1=np.zeros((3,m.nv))
    mujoco.mj_jac(m,d,jp0,jr0,p0,int(m.site_bodyid[src]))
    mujoco.mj_jac(m,d,jp1,jr1,p1,int(m.site_bodyid[dst]))
    velocity=np.random.default_rng(42).normal(size=m.nv)*.1
    expected=(jp0-jp1-np.cross(jr1.T,p0-p1).T)@velocity
    sample=mujoco.MjData(m);mujoco.mj_copyData(sample,m,d)
    eps=1e-7;mujoco.mj_integratePos(m,sample.qpos,velocity,eps);mujoco.mj_forward(m,sample)
    a=sample.site_xmat[src].reshape(3,3);b=sample.site_xmat[dst].reshape(3,3)
    later=b.T@(sample.site_xpos[src]+a@offset-sample.site_xpos[dst]-b@offset)
    initial=r1.T@(p0-p1)
    np.testing.assert_allclose(r1@(later-initial)/eps,expected,atol=1e-7)


def test_load_jacobian_proposal_does_not_advance_live_state():
    x=NativeReconfiguration();x.phase='RELEASE';x.target=0
    before=x.data.qpos.copy();velocity=x.data.qvel.copy();time=x.data.time
    try:x._load_control(0)
    finally:
        if x._load_pool is not None:x._load_pool.shutdown()
    np.testing.assert_array_equal(x.data.qpos,before)
    np.testing.assert_array_equal(x.data.qvel,velocity)
    assert x.data.time==time and x.load_prediction_calls==1 and x.phase=='RELEASE'


def test_retained_load_optimization_includes_baseline_drift():
    jac=np.zeros((6,6));jac[0,0]=1.;jac[5,0]=1.;jac[5,1]=1.
    predicted=np.zeros(6);current=np.zeros(6);predicted[0]=.1
    delta=_solve_retained_load_response(jac,predicted,current,5,[(-1.,1.)]*6)
    assert delta is not None
    assert predicted[0]+jac[0]@delta<=current[0]+1e-7
    assert jac[5]@delta>0


def test_validated_release_ramp_obeys_target_speed():
    x=NativeReconfiguration();x.phase='RELEASE_HEADTAIL'
    base=x.qref.copy();delta=np.array([.6,-.3,0.,.12,-.06,.24])
    x.release_plan={'base':base,'delta':delta,'elapsed':0.}
    previous=x.qref.copy()
    for _ in range(150):
        x._load_control(5)
        assert np.max(np.abs(x.qref-previous))<=x.cfg.joint_target_speed*x.cfg.control_dt+1e-10
        previous=x.qref.copy()
    np.testing.assert_allclose(x.qref,base+delta,atol=1e-10)
    assert x.release_plan is None


def test_capture_requires_stable_relative_velocity(monkeypatch):
    # Unit test of transition logic only: it does not claim a physical docking.
    x=NativeReconfiguration(); x.phase='CAPTURE'; x.target=5; x.axial_armed=True; x.phase_axial_advance_samples=1
    statuses=[dict(item) for item in x.interfaces()]
    statuses[5].update(seated=True,quarter=1,relative_speed_m_s=.04,
                       relative_angular_speed_rad_s=0.)
    monkeypatch.setattr(x,'interfaces',lambda: statuses)
    for _ in range(3):x._advance()
    assert x.phase=='CAPTURE' and x.confirm==0.
    statuses[5]['relative_speed_m_s']=0.
    for _ in range(3):x._advance()
    assert x.phase=='RELEASE' and x.target==0
    assert x.registration[5]==1


def test_capture_rejects_old_180_degree_registration(monkeypatch):
    x=NativeReconfiguration(); x.phase='CAPTURE'; x.target=5
    statuses=[dict(item) for item in x.interfaces()]
    statuses[5].update(seated=True,quarter=0,relative_speed_m_s=0.,
                       relative_angular_speed_rad_s=0.)
    commands=[]
    monkeypatch.setattr(x,'interfaces',lambda: statuses)
    monkeypatch.setattr(x,'_approach_control',lambda face,info,desired: commands.append(desired))
    for _ in range(4):x._advance()
    assert x.phase=='CAPTURE' and x.confirm==0.
    assert commands==[1]*4


def test_preseated_capture_cannot_skip_final_axial_approach(monkeypatch):
    x=NativeReconfiguration();x.phase='CAPTURE';x.target=5
    statuses=[dict(item) for item in x.interfaces()]
    statuses[5].update(seated=True,quarter=1,relative_speed_m_s=0.,relative_angular_speed_rad_s=0.)
    monkeypatch.setattr(x,'interfaces',lambda:statuses)
    commands=[]
    monkeypatch.setattr(x,'_approach_control',lambda face,info,desired:commands.append(desired))
    for _ in range(4):x._advance()
    assert x.phase=='CAPTURE' and commands==[1]*4


def test_non_target_magnetic_contact_loss_requires_confirmation():
    x=NativeReconfiguration(); x.phase='RELEASE';x.target=0
    active=np.ones(6,dtype=bool);active[1]=False
    for _ in range(49):x._track_non_target_connections(active)
    assert x.phase=='RELEASE'
    x._track_non_target_connections(active)
    assert x.phase=='FAILED' and 'non-target magnetic contact lost' in x.failure


def test_approach_moves_xy_and_normal_together(monkeypatch):
    x=NativeReconfiguration();face=5
    R=x.audit.scratch.site_xmat[x.inlet[face]].reshape(3,3)
    commands=[]
    monkeypatch.setattr(x,'_physical_points',lambda _: (R@np.array([.001,-.002,.04]),np.zeros(3)))
    monkeypatch.setattr(x,'_local_control',lambda face,linear,angular: commands.append((linear,angular)))
    x._approach_control(face,{'seated':False,'angle_rad':0.},0)
    local=R.T@commands[0][0]
    assert local[0]<0 and local[1]>0 and local[2]<0
    monkeypatch.setattr(x,'_physical_points',lambda _: (R@np.array([0.,0.,-.04]),np.zeros(3)))
    x._approach_control(face,{'seated':False,'angle_rad':0.},0)
    assert x.phase!='FAILED' and (R.T@commands[1][0])[2]>0


def test_late_approach_keeps_xy_correction_while_advancing(monkeypatch):
    x=NativeReconfiguration(); face=5; x.axial_armed=True
    R=x.audit.scratch.site_xmat[x.inlet[face]].reshape(3,3)
    commands=[]; local=np.array([.001,0.,.005])
    monkeypatch.setattr(x,'_physical_points',lambda _: (R@local,np.zeros(3)))
    monkeypatch.setattr(x,'_local_control',lambda face,linear,angular: commands.append(R.T@linear))
    x._approach_control(face,{},0)
    assert commands[-1][2]<0 and commands[-1][0]<0
    local[0]=.00001
    x._approach_control(face,{},0)
    assert commands[-1][2]<0


def test_late_axial_deviation_is_recorded_not_a_failure(monkeypatch):
    x=NativeReconfiguration();x.phase='CAPTURE';x.target=5;x.axial_armed=True;x.step_count=1
    samples=iter([(np.array([.001,0.,.005]),.02),
                  (np.array([.001,0.,.0049]),.02)])
    monkeypatch.setattr(x,'_axial_error',lambda *args:next(samples))
    x.step()
    assert x.phase!='FAILED' and x.axial_advance_samples==1
    assert x.axial_peak_lateral_m==pytest.approx(.001)


def test_coupled_prediction_offsets_support_drift_while_moving_target():
    # Both joints move the target, but the first also moves a retained
    # connection. The optimizer can preserve the latter using joint 2.
    response=np.zeros((2,6))
    response[0,:2]=[1.,1.]
    response[1,0]=1.
    dq=_solve_coupled_response(response,np.array([1.,0.]),
                               np.array([0.,.25]),np.array([1.,2.]),damping=1e-12)
    np.testing.assert_allclose(response@dq+np.array([0.,.25]),[1.,0.],atol=1e-9)


def test_later_peel_coordinates_non_target_faces(monkeypatch):
    from zbot_mjbatch import native_reconfiguration
    x=NativeReconfiguration();x.phase='RELEASE';x.stage=1;x.target=1
    captured=[]
    def solve(response,desired,drift,weights,**kwargs):
        captured.append((response.copy(),drift.copy(),weights.copy()))
        return np.zeros(6)
    monkeypatch.setattr(native_reconfiguration,'_solve_coupled_response',solve)
    x._local_control(1,np.zeros(3),np.array([.08,0.,0.]))
    response,drift,weights=captured[0]
    assert response.shape==(36,6)
    np.testing.assert_allclose(weights.reshape(6,6)[1],1.)
    np.testing.assert_allclose(weights.reshape(6,6)[0],.3)
    np.testing.assert_array_equal(drift.reshape(6,6)[1],np.zeros(6))


def test_rotation_restores_measured_released_face_offset(monkeypatch):
    x=NativeReconfiguration();x.phase='ROTATE';x.target=0
    R=x.audit.scratch.site_xmat[x.inlet[0]].reshape(3,3)
    x.rotation_anchor=np.array([0.,0.,.012])
    monkeypatch.setattr(x,'_physical_points',lambda _: (R@np.array([.001,0.,.010]),np.zeros(3)))
    commands=[]
    monkeypatch.setattr(x,'_local_control',lambda face,linear,angular: commands.append((linear,angular)))
    x._advance()
    local=R.T@commands[0][0]
    assert local[0] < 0. and local[2] > 0.


def test_separation_damps_recoil_before_rotation(monkeypatch):
    x=NativeReconfiguration();x.phase='SEPARATE';x.target=0;x.stage=2
    states=[dict(s) for s in x.interfaces()]
    states[0].update(position_m=.02,relative_speed_m_s=.1,relative_angular_speed_rad_s=1.)
    monkeypatch.setattr(x,'interfaces',lambda: states)
    commands=[]
    monkeypatch.setattr(x,'_local_control',lambda face,linear,angular: commands.append((linear,angular)))
    x._advance()
    assert x.phase=='SEPARATE' and x.confirm==0.
    np.testing.assert_allclose(commands[-1][0],np.zeros(3),atol=1e-15)
    states[0].update(relative_speed_m_s=0.,relative_angular_speed_rad_s=0.)
    for _ in range(3):x._advance()
    assert x.phase=='ROTATE' and x.rotation_anchor is not None


def test_fourth_face_aligns_before_rotation(monkeypatch):
    x=NativeReconfiguration();x.phase='SEPARATE';x.target=3;x.stage=2
    states=[dict(s) for s in x.interfaces()]
    states[3].update(attracting=False,relative_speed_m_s=0.,relative_angular_speed_rad_s=0.)
    monkeypatch.setattr(x,'interfaces',lambda: states)
    R=x.audit.scratch.site_xmat[x.inlet[3]].reshape(3,3)
    local=np.array([.009,0.,-.012])
    monkeypatch.setattr(x,'_physical_points',lambda _: (R@local,np.zeros(3)))
    commands=[]
    monkeypatch.setattr(x,'_local_control',lambda face,linear,angular: commands.append(R.T@linear))
    x._advance()
    assert x.phase=='SEPARATE' and commands[-1][0]<0.
    local[:]=[.0001,0.,-.014]
    for _ in range(3):x._advance()
    assert x.phase=='ROTATE'
