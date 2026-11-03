"""Asynchronous worlds, selective reset, GPU Jacobian and load constraints."""
import copy
import numpy as np
import pytest
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
from zbot_rl_mjlab.tasks.reconfiguration.live_loading import response
from zbot_rl_mjlab.tasks.reconfiguration.batched_loading import projected_increments,peel
from zbot_rl_mjlab.tasks.reconfiguration.state_machine import Phase

TASK='Mjlab-Zbot-6dof-Reconfigure-Mixed-270-270-180-90-90-PPO'


@pytest.fixture
def env():
    if not torch.cuda.is_available():pytest.skip('GPU physical integration test')
    cfg=load_env_cfg(TASK,play=True);cfg.scene.num_envs=4
    with torch.inference_mode():
        e=ManagerBasedRlEnv(cfg,device='cuda:0');e.reset()
        try:yield e
        finally:e.close()


def test_projected_loading_makes_safe_partial_progress():
    A=torch.zeros(2,6,6,6);w=torch.zeros(2,6,6);torque=torch.zeros(2,6)
    A[:,0,2,0]=10.;A[:,1,2,0]=30.;w[:,0,2]=10.;w[:,1,2]=55.
    delta,feasible,predicted=projected_increments(A,w,torque,torch.zeros(2,dtype=torch.long),.05)
    assert (predicted[:,1:]<=56.+1e-4).all()
    assert (delta.abs()<=.200001).all() and ((torque+delta).abs()<=14.).all()
    assert (predicted[:,0]>10.001).all() and feasible.all()
    torch.testing.assert_close(predicted,peel(w+torch.einsum('bfwa,ba->bfw',A,delta),.05))


def test_independent_phases_recapture_advance_and_selective_reset(env,monkeypatch):
    s=state(env);a=env.action_manager.get_term('joint_torque');c=a.distance_controller;dev=env.device
    s.index.copy_(torch.tensor([0,1,2,3],device=dev))
    for world,stage in enumerate((0,1,2,3)):
        for connector in (0,1,3,4)[:stage]:
            env.sim.data.eq_active[world,s.eq_ids[connector]]=False
            env.sim.data.eq_active[world,s.variant_ids[connector]]=True
    target=s.target_indices()
    env.sim.data.eq_active[s.worlds,s.eq_ids[target]]=False
    env.sim.data.eq_active[3,s.variant_ids[4]]=True
    c.phase.copy_(torch.tensor([0,2,3,4],device=dev));c.phase_time.copy_(torch.tensor([0.,.2,.035,2.2],device=dev));c.capture_time[2]=.03
    q0=c.goal_q.clone();q1=torch.zeros_like(q0);q1[:,0]=1.
    distance=torch.tensor([.001,.055,0.,0.],device=dev);zero=torch.zeros(4,3,device=dev)
    monkeypatch.setattr(s,'geometry',lambda:dict(distance=distance,q0=q0,q1=q1,dv=zero,dw=zero,delta=zero))
    c.update(s,torch.ones(4,device=dev,dtype=torch.bool))
    assert c.phase.tolist()==[Phase.SEPARATE,Phase.APPROACH,Phase.SETTLE,Phase.DONE]
    assert c.index.tolist()==[0,1,2,4]
    assert env.sim.data.eq_active[2,s.variant_ids[3]]
    assert env.cfg.target_connector==0  # immutable recipe, not a global selector
    before_phase=c.phase.clone();before_eq=env.sim.data.eq_active.clone()
    a.loading.torque.copy_(torch.arange(24,device=dev).reshape(4,6))
    before_torque=a.loading.torque.clone();before_pose=env.sim.data.qpos.clone()
    env.reset(env_ids=torch.tensor([1,3],device=dev))
    assert c.index.tolist()==[0,0,2,0]
    assert c.phase[1]==0 and c.phase[3]==0
    torch.testing.assert_close(c.phase[[0,2]],before_phase[[0,2]])
    torch.testing.assert_close(env.sim.data.eq_active[[0,2]],before_eq[[0,2]])
    torch.testing.assert_close(env.sim.data.qpos[[0,2]],before_pose[[0,2]])
    torch.testing.assert_close(a.loading.torque[[0,2]],before_torque[[0,2]])
    assert a.loading.torque[[1,3]].count_nonzero()==0
    assert env.sim.data.eq_active[[1,3]][:,s.eq_ids].all()
    assert not env.sim.data.eq_active[[1,3]][:,s.variant_ids].any()


def test_gpu_weld_operator_agrees_with_cpu_formula(env):
    s=state(env);a=env.action_manager.get_term('joint_torque');ids=torch.arange(4,device=env.device)
    old=env.sim.data.efc.state.clone()
    # Compare the same weld-only operator; native CPU/Warp contact sets can
    # differ at mesh boundaries, so do not compare different active sets.
    env.sim.data.efc.state.masked_fill_(env.sim.data.efc.type!=0,0)
    try:A,_,present=a.loading.response(s,ids)
    finally:env.sim.data.efc.state.copy_(old)
    m=copy.copy(env.sim.mj_model);m.opt.disableflags|=mujoco.mjtDisableBit.mjDSBL_CONTACT;d=mujoco.MjData(m)
    d.qpos[:]=env.sim.data.qpos[0].cpu().numpy();d.qvel[:]=env.sim.data.qvel[0].cpu().numpy();d.eq_active[:]=env.sim.data.eq_active[0].cpu().numpy()
    cpu,_=response(m,d,s.constraint_ids()[0].cpu().numpy())
    assert present.all()
    relative=np.linalg.norm(cpu-A[0].cpu().numpy())/np.linalg.norm(cpu)
    assert relative<1e-4



def test_closed_face_gpu_filter_is_per_world(env):
    import warp as wp
    from zbot_rl_mjlab.tasks.reconfiguration.batched_collision import _filter_closed_faces, _MARKER
    s=state(env);s.index.copy_(torch.tensor([0,1,2,3],device=env.device))
    target=s.target_indices()
    env.sim.data.eq_active[s.worlds[:2],s.eq_ids[target[:2]]]=False
    lookup,old,new=getattr(env.sim.wp_model.opt,_MARKER)
    m=env.sim.mj_model;pairs=s.pair_ids.cpu().numpy()
    first=torch.as_tensor(m.pair_geom1[pairs].copy(),device=env.device,dtype=torch.int32)
    second=torch.as_tensor(m.pair_geom2[pairs].copy(),device=env.device,dtype=torch.int32)
    geoms=torch.stack((first[target],second[target]),-1).contiguous()
    worlds=torch.arange(4,device=env.device,dtype=torch.int32)
    kinds=torch.ones(4,device=env.device,dtype=torch.int32)
    count=torch.tensor([4],device=env.device,dtype=torch.int32)
    wp.launch(_filter_closed_faces,dim=4,inputs=[wp.from_torch(count),wp.from_torch(worlds),
              wp.from_torch(geoms,dtype=wp.vec2i),wp.from_torch(kinds),env.sim.wp_data.eq_active,
              lookup,old,new],device=env.sim.wp_device)
    assert kinds.tolist()==[1,1,0,0]
