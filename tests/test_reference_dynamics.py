"""Axial geometry, native reference control and smooth timing regressions."""
from types import SimpleNamespace as NS
import math
import mujoco
import numpy as np
import torch
import warp as wp
from zbot_rl_mjlab.tasks.reconfiguration.reference_trajectory import TimedCurve,AxialPlanner
from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import inverse
from zbot_rl_mjlab.tasks.reconfiguration.batched_collision import _filter_phase_faces


def test_continuous_time_scaling_respects_speed_acceleration_and_endpoint():
    points=np.array([[0.,0.],[.1,.03],[.4,-.2],[.45,-.21],[.9,.2]])
    curve=TimedCurve(points,speed=math.pi/4,acceleration=math.pi,dt=.005)
    assert np.max(abs(curve.v))<=math.pi/4+1e-9
    assert np.max(abs(curve.a))<=math.pi+1e-9
    np.testing.assert_allclose(curve.q[-1],points[-1],atol=1e-10)
    np.testing.assert_allclose(curve.v[-1],0.,atol=1e-10)
    np.testing.assert_allclose(curve.a[-1],0.,atol=1e-10)
    assert curve.duration>1.


def test_first_interface_separates_axially_and_has_checked_rotation():
    planner=AxialPlanner(.051);k=planner.kin;m=k.model
    modules=[]
    for i,adr in enumerate(k.rootadr):
        modules.append(NS(data=NS(joint_pos=torch.tensor([[k.q[i]]]),
            root_link_pos_w=torch.tensor(m.qpos0[adr:adr+3])[None],
            root_link_quat_w=torch.tensor(m.qpos0[adr+3:adr+7])[None])))
    fake=NS(modules=modules,eq_ids=torch.tensor(k.old_eq),variant_ids=torch.tensor(k.new_eq[:5]),
            env=NS(sim=NS(data=NS(eq_active=torch.tensor(k.active)[None]))))
    anchor=k.roots[1].copy();q0=k.q.copy();_,source,_=k.fk(q0,0,anchor);start=inverse(anchor)@source
    result=planner.plan(fake,0)
    assert set(result.points)=={'SEPARATE','ROTATE','APPROACH'}
    for q in result.points['SEPARATE']:
        _,source,_=k.fk(q,0,anchor);pose=inverse(anchor)@source
        np.testing.assert_allclose(pose[:2,3],start[:2,3],atol=2e-5)
        np.testing.assert_allclose(pose[:3,:3],start[:3,:3],atol=2e-4)
    _,source,_=k.fk(result.points['SEPARATE'][-1],0,anchor)
    assert ((inverse(anchor)@source)[2,3]-start[2,3])<-.0509
    assert all(row['max_nonmating_penetration_m']<=1e-5 for row in result.diagnostics)
    assert result.diagnostics[1]['min_rotating_pair_clearance_m']>=.001


def test_reference_cfg_keeps_load_limits_and_uses_native_pd():
    cfg=reference_env_cfg()
    assert not cfg.auto_reset and cfg.scene.num_envs==1 and cfg.rewards=={}
    assert cfg.sim.mujoco.timestep==.001 and cfg.decimation==20
    assert cfg.break_force==80 and cfg.minimum_other_margin==.2
    assert cfg.continuous_torque_limit==14 and cfg.torque_limit==20
    assert cfg.separation_stiffness==400 and cfg.separation_damping==4
    assert cfg.reference_speed_deg_s==45 and cfg.reference_acceleration_deg_s2==180
    assert cfg.reference_clearance_m==.001 and cfg.separation_distance==.05


def test_phase_contact_filter_exempts_retract_and_approach_preserving_sensors():
    wp.init()
    dev='cuda:0' if torch.cuda.is_available() else 'cpu'
    count=torch.tensor([4],device=dev,dtype=torch.int32);world=torch.arange(4,device=dev,dtype=torch.int32)
    geoms=torch.tensor([[0,1]]*4,device=dev,dtype=torch.int32);kinds=torch.full((4,),3,device=dev,dtype=torch.int32)
    active=torch.zeros(4,11,device=dev,dtype=torch.bool);active[3,0]=True
    enabled=torch.zeros(4,5,device=dev,dtype=torch.bool);enabled[0,0]=True;enabled[3,0]=True
    lookup=torch.full((2,2),-1,device=dev,dtype=torch.int32);lookup[0,1]=0
    old=torch.arange(5,device=dev,dtype=torch.int32);new=old+6
    wp.launch(_filter_phase_faces,dim=4,inputs=[wp.from_torch(count),wp.from_torch(world),wp.from_torch(geoms,dtype=wp.vec2i),
        wp.from_torch(kinds),wp.from_torch(active),wp.from_torch(lookup),wp.from_torch(old),wp.from_torch(new),wp.from_torch(enabled)],device=dev)
    assert kinds.tolist()==[3,2,2,2]


def test_phase_switches_keep_native_gains_fixed(monkeypatch):
    from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import ReferenceAction,ReferencePhase
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.env_cfg import TorqueAction
    monkeypatch.setattr(TorqueAction,'apply_actions',lambda self:None)
    a=object.__new__(ReferenceAction)
    s=NS(waiting=torch.zeros(1,dtype=torch.bool),target_indices=lambda:torch.tensor([0]))
    a._env=NS(cfg=NS(separation_stiffness=400.,separation_damping=4.),_disconnect_state=s)
    a.motion_stiffness=400.;a.motion_damping=4.
    a.pd_active=torch.ones(1,dtype=torch.bool);a.phase_contacts=torch.zeros(1,6,dtype=torch.bool)
    a.distance_controller=NS(phase=torch.tensor([ReferencePhase.SEPARATE]))
    for phase in (ReferencePhase.SEPARATE,ReferencePhase.ROTATE,ReferencePhase.APPROACH,ReferencePhase.SETTLE):
        a.distance_controller.phase.fill_(phase);a.apply_actions()
        assert a._env.cfg.separation_stiffness==400. and a._env.cfg.separation_damping==4.
        assert bool(a.phase_contacts[0,0])==(phase==ReferencePhase.ROTATE)
