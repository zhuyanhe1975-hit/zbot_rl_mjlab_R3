"""Native CPU initialization, rigid chain invariants and rupture handoff."""
import mujoco
import numpy as np
import pytest
import torch
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu import FullCpuReconfiguration,headtail_projected_increments


@pytest.fixture
def fsm():
    instance=FullCpuReconfiguration()
    try:yield instance
    finally:instance.close()


def test_straight_cpu_initialization_and_hold(fsm):
    assert fsm.model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert fsm.model.opt.timestep==.001
    assert fsm.connected().tolist()==[True,True,True,True,True,False]
    np.testing.assert_array_equal(fsm.data.qpos[fsm.qadr],np.zeros(6))
    assert not fsm.data.eq_active[fsm.new].any()
    snapshot=fsm.snapshot()
    assert snapshot['root_collinearity_error_m']<.0001
    assert len(snapshot['loads_N'])==len(snapshot['peak_loads_N'])==6
    assert snapshot['interface_gap_m'][5]>.5
    for _ in range(1000):fsm.step()
    assert fsm.phase!='FAILED' and fsm.connected()[:5].all()
    assert fsm.peak_other<64
    assert fsm.data.time==pytest.approx(1.,abs=1e-10)


def test_passive_rupture_removes_loading_on_same_substep(fsm):
    fsm.target=0;fsm.phase='LOAD';fsm.loading_mode=True;fsm.loading_torque[:]=1.
    m=fsm.model;d=fsm.data
    rows=np.flatnonzero((d.efc_type==0)&(d.efc_id==fsm.old[0]))
    J=d.efc_J.reshape(d.nefc,m.nv)[rows,fsm.root_dofs[0]:fsm.root_dofs[0]+6]
    force=d.site_xmat[fsm.out[0]].reshape(3,3)[:,2]*85.
    arm=d.site_xpos[fsm.out[0]]-d.xpos[fsm.root_bodies[0]]
    rootR=d.xmat[fsm.root_bodies[0]].reshape(3,3)
    generalized=np.r_[force,rootR.T@np.cross(arm,force)]
    d.efc_force[rows]=np.linalg.solve(J.T,generalized)
    fsm.measure()
    assert not fsm.data.eq_active[fsm.old[0]]
    assert fsm.phase=='LOAD'  # CPU phase transition still waits for its control tick.
    assert not fsm.loading_mode  # Force output does not wait for that tick.
    np.testing.assert_allclose(fsm.qref,d.qpos[fsm.qadr])
    np.testing.assert_array_equal(m.actuator_gainprm[fsm.actuators,0],np.full(6,400.))


def test_final_auxiliary_loading_preserves_native_position_hold(fsm):
    fsm.start_load('LOAD_HEADTAIL',5)
    assert fsm.loading_hold and fsm.loading_mode
    np.testing.assert_allclose(fsm.loading_qhold,fsm.data.qpos[fsm.qadr])
    np.testing.assert_array_equal(fsm.model.actuator_biasprm[fsm.actuators,1],np.full(6,-400.))
    np.testing.assert_allclose(fsm.data.ctrl[fsm.actuators],fsm.loading_qhold)


def test_connection_hold_is_one_second_including_soft_weld_ramp(fsm):
    fsm.phase='SETTLE_HEADTAIL';fsm.phase_start=0.;fsm.data.eq_active[fsm.old[5]]=True
    fsm.data.time=.999;fsm.advance()
    assert fsm.phase=='SETTLE_HEADTAIL'
    # Accumulated floating-point time can be infinitesimally below 1.0s.
    fsm.data.time=1.-1e-10;fsm.advance()
    assert fsm.phase=='LOAD' and fsm.target==0
    assert fsm.model.eq_solref[fsm.old[5],0]==pytest.approx(.005)
    assert fsm.cfg.unfold_target_step==pytest.approx(.008)


def test_headtail_projection_does_not_treat_compression_as_tension():
    # A small torque change leaves the face compressed and its bending load
    # unchanged; it must not be selected as an increase in opening load.
    A=torch.zeros(1,6,6,6);w=torch.zeros(1,6,6);torque=torch.zeros(1,6)
    w[0,5,2]=-10.;w[0,5,3]=.5;A[0,5,2,0]=100.
    delta,usable,predicted=headtail_projected_increments(A,w,torque,torch.tensor([5]),.05)
    torch.testing.assert_close(delta,torch.zeros_like(delta))
    assert not usable.item()
    assert predicted[0,5].item()==pytest.approx(10.)


def test_headtail_mating_pair_enabled_during_rotation_and_unfold(fsm):
    m=fsm.model;d=fsm.data;ga,gb=fsm.pair_geoms[5]
    # Put the two halves into clear geometric overlap in this isolated model.
    adr=fsm.roots[5]
    d.qpos[adr:adr+3]+=d.geom_xpos[ga]-d.geom_xpos[gb]
    d.qpos[adr+1]+=.02
    mujoco.mj_forward(m,d)
    def contacts():return [c for c in d.contact if set(c.geom)=={ga,gb}]
    assert not contacts()
    fsm.target=5;fsm.phase='ROTATE';fsm.set_contact_phase();mujoco.mj_forward(m,d)
    assert contacts()

    fsm.phase='UNFOLD';fsm.set_contact_phase();mujoco.mj_forward(m,d)
    assert contacts()
    fsm.phase='FINAL_STRAIGHT_HOLD';fsm.set_contact_phase();mujoco.mj_forward(m,d)
    assert contacts()
