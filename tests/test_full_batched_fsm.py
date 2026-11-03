import torch
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import FullBatchedFSM,P


def state(n):
    c=torch.ones(n,6,dtype=torch.bool);r=torch.zeros(n,5,dtype=torch.bool);loads=torch.zeros(n,6)
    g={'gap':torch.zeros(n),'angle':torch.zeros(n),'linear_speed':torch.zeros(n),'angular_speed':torch.zeros(n)}
    return c,r,loads,g,torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool)


def test_async_phases_and_selective_reset():
    f=FullBatchedFSM(4,'cpu');c,r,l,g,curve,straight,collision,torque=state(4)
    f.phase[:]=torch.tensor([P['INITIAL_STRAIGHT_HOLD'],P['LOAD'],P['SETTLE'],P['UNFOLD']]);f.phase_time[:]=torch.tensor([.99,0.,.99,0.])
    f.target[:]=torch.tensor([5,0,1,5]);f.stage[:]=torch.tensor([0,0,1,4]);c[0,5]=False;c[1,0]=False;c[3,5]=False;straight[3]=True
    f.update(c,r,l,g,curve,straight,collision,torque)
    assert f.phase.tolist()==[P['CURL'],P['SEPARATE'],P['LOAD'],P['FINAL_STRAIGHT_HOLD']]
    assert f.target.tolist()==[5,0,3,5]
    f.reset(torch.tensor([1,3]));assert f.phase.tolist()==[P['CURL'],P['INITIAL_STRAIGHT_HOLD'],P['LOAD'],P['INITIAL_STRAIGHT_HOLD']]


def test_other_weld_failure_uses_80N_and_target_is_passive():
    f=FullBatchedFSM(3,'cpu');c,r,l,g,curve,straight,collision,torque=state(3)
    f.phase[:]=P['LOAD'];f.target[:]=0;l[0,1]=79.99;l[1,1]=80.;l[2,0]=90.
    f.update(c,r,l,g,curve,straight,collision,torque)
    assert f.phase.tolist()==[P['LOAD'],P['FAILED'],P['LOAD']]
    assert c[2,0] # Controller did not force the target weld open.


def test_complete_phase_sequence_requires_passive_breaks_and_captures():
    f=FullBatchedFSM(1,'cpu');c,r,l,g,curve,straight,collision,torque=state(1);c[0,5]=False
    def step():return f.update(c,r,l,g,curve,straight,collision,torque)
    f.phase_time[:]=.99;step();assert f.phase.item()==P['CURL']
    step();assert f.phase.item()==P['CAPTURE_HEADTAIL']
    step();event=step();assert event['capture_headtail'].item();c[0,5]=True
    f.phase_time[:]=.99;step();assert f.phase.item()==P['LOAD']
    for target in (0,1,3,4):
        assert f.target.item()==target
        c[0,target]=False;step();assert f.phase.item()==P['SEPARATE']
        g['gap'][:]=.051;step();assert f.phase.item()==P['ROTATE']
        step();assert f.phase.item()==P['APPROACH']
        g['gap'][:]=0.;step();event=step();assert event['capture_internal'].item()
        c[0,target]=True;r[0,target]=True;f.phase_time[:]=.99;step()
    assert f.phase.item()==P['LOAD_HEADTAIL'] and f.target.item()==5
    c[0,5]=False;step();assert f.phase.item()==P['UNFOLD']
    straight[:]=True;step();assert f.phase.item()==P['FINAL_STRAIGHT_HOLD']
    f.phase_time[:]=.99;step();assert f.phase.item()==P['DONE']


def test_every_live_phase_has_independent_simulation_time_watchdog():
    names=['INITIAL_STRAIGHT_HOLD','CURL','CAPTURE_HEADTAIL','SETTLE_HEADTAIL','LOAD',
           'SEPARATE','ROTATE','APPROACH','SETTLE','LOAD_HEADTAIL','UNFOLD','FINAL_STRAIGHT_HOLD']
    f=FullBatchedFSM(len(names),'cpu');c,r,l,g,curve,straight,collision,torque=state(len(names))
    f.phase[:]=torch.tensor([P[name] for name in names]);f.phase_time.zero_();f.phase_elapsed[:]=f.phase_timeouts[f.phase]-.01
    f.target[:]=torch.tensor([5,5,5,5,0,0,0,0,0,5,5,5])
    for row,name in enumerate(names):
        if name in ('INITIAL_STRAIGHT_HOLD','CURL','CAPTURE_HEADTAIL','UNFOLD','FINAL_STRAIGHT_HOLD'):c[row,5]=False
        if name in ('SEPARATE','ROTATE','APPROACH'):c[row,0]=False
    g['gap'].fill_(1.);curve.zero_()
    result=f.update(c,r,l,g,curve,straight,collision,torque,dt=torch.zeros(len(names)))
    assert result['failed'].all() and (f.failure_code==5).all()
    assert (f.phase_elapsed==0).all()
    f.reset(torch.tensor([1,7]));assert f.phase[1]==P['INITIAL_STRAIGHT_HOLD'] and f.phase[7]==P['INITIAL_STRAIGHT_HOLD']
    assert f.phase[0]==P['FAILED'] and f.failure_code[0]==5


def test_phase_completion_at_watchdog_boundary_takes_precedence():
    f=FullBatchedFSM(1,'cpu');c,r,l,g,curve,straight,collision,torque=state(1)
    c[0,5]=False;f.phase_elapsed[:]=f.phase_timeouts[P['INITIAL_STRAIGHT_HOLD']]-.01;f.phase_time[:]=.99
    result=f.update(c,r,l,g,curve,straight,collision,torque)
    assert not result['failed'].item() and f.phase.item()==P['CURL']


def test_optional_substep_other_margin_is_a_failure(monkeypatch):
    from types import SimpleNamespace
    import zbot_rl_mjlab.tasks.reconfiguration.full_batched_env as module
    flags=torch.tensor([False, True])
    state=SimpleNamespace(fsm=SimpleNamespace(phase=torch.tensor([0, 0])),
        wrong_break=torch.zeros(2,dtype=torch.bool), invalid_state=torch.zeros(2,dtype=torch.bool),
        contact_violation=torch.zeros(2,dtype=torch.bool), safety_margin_violation=flags)
    action=SimpleNamespace(torque_failure=torch.zeros(2,dtype=torch.bool))
    env=SimpleNamespace(cfg=SimpleNamespace(full_enforce_other_margin=True),
                        action_manager=SimpleNamespace(get_term=lambda name: action))
    monkeypatch.setattr(module,'state',lambda env: state)
    assert module.failed(env).tolist()==[False, True]
    env.cfg.full_enforce_other_margin=False
    assert module.failed(env).tolist()==[False, False]
