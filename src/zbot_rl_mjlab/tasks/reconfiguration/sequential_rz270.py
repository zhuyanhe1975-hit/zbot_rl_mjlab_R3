"""Closed 180-degree assembly -> five physical re-clockings -> open RZ270 snake."""
import math
from dataclasses import dataclass,fields
from enum import IntEnum
import mujoco
import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.lab_api.math import quat_mul,quat_conjugate,quat_apply_inverse,quat_apply
from .state_machine import StateMachineEnvCfg,ReconfigurationController,Phase,rotvec,state_machine_env_cfg
from .disconnect.env_cfg import TorqueAction,TorqueActionCfg
from .disconnect.state import ConnectorState,state,failure
from .disconnect.separation import DistanceJacobianController
from .disconnect.model import add_connectors
from .live_loading import LiveLoadingController

EXPERIMENT_ID='Zbot-Closed180-To-Snake-RZ270-FSM-v1'

class SequencePhase(IntEnum):
    LOAD=0;SEPARATE=1;ROTATE=2;APPROACH=3;SETTLE=4;DONE=5;FAILED=6
    UNFOLD=7;SNAKE_HOLD=8

@dataclass
class SequentialRz270EnvCfg(StateMachineEnvCfg):
    sequential_rz270: bool=True
    connection_order: tuple[int,...]=(0,1,2,3,4)
    experiment_id: str=EXPERIMENT_ID
    unfold_speed: float=1.
    unfold_gain: float=3.
    snake_angle_tolerance: float=.02
    snake_velocity_tolerance: float=.05
    unfold_timeout_s: float=20.


def add_sequential_connectors(spec,include_headtail=False):
    add_connectors(spec)
    for i in range(6 if include_headtail else 5):
        original=spec.site(f'module_{i}/port_out')
        q=np.empty(4)
        mujoco.mju_mulQuat(q,original.quat,np.array([math.cos(math.pi/4),0.,0.,math.sin(math.pi/4)]))
        R=np.empty(9);mujoco.mju_quat2Mat(R,original.quat);R=R.reshape(3,3)
        turn=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        inlet=spec.site(f'module_{(i+1)%6}/port_in').pos
        # RZ270 rotates about the CAD mounting origin, not the slightly offset
        # diagnostic port site. Preserve the reference model's translation.
        position=original.pos+R@(turn@inlet-inlet)
        original.parent.add_site(name=f'module_{i}/port_out_rz270',pos=position,quat=q,size=(.004,)*3)
        spec.add_equality(name=f'connector_{i}_rz270',type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_SITE,name1=f'module_{i}/port_out_rz270',name2=f'module_{(i+1)%6}/port_in',
            active=False,data=(*([0.]*10),1.),solref=(.03,1.),solimp=(.999,.9999,.001,.5,2.))


class SequentialConnectorState(ConnectorState):
    def __init__(self,env):
        super().__init__(env)
        self.variant_ids=torch.tensor([env.sim.mj_model.equality(f'connector_{i}_rz270').id for i in range(5)],device=env.device)

    @property
    def new_eq(self):
        i=self.cfg.target_connector
        return int(self.variant_ids[i]) if i<5 else int(self.eq_ids[5])

    def constraint_ids(self):
        ids=super().constraint_ids().clone()
        on=self.env.sim.data.eq_active[:,self.variant_ids].bool()
        ids[:,:5]=torch.where(on,self.variant_ids[None],ids[:,:5])
        return ids

    def reset(self,ids):
        ids=slice(None) if ids is None else ids
        super().reset(ids)
        if isinstance(ids,slice):self.env.sim.data.eq_active[:,self.variant_ids]=False
        else:self.env.sim.data.eq_active[ids[:,None],self.variant_ids[None]]=False

    def update_separation(self,next_active):
        self.max_target_gap.copy_(torch.maximum(self.max_target_gap,self.geometry()['distance']))
        self.safety_margin_violation |= (1.-self.other_margin())>(1.-self.cfg.minimum_other_margin)


class SequentialController(ReconfigurationController):
    def __init__(self,env,modules):
        super().__init__(env,modules)
        self.phase_durations=torch.zeros(env.num_envs,len(SequencePhase),device=env.device)
        self.goal_q=torch.tensor([math.cos(math.pi/4),0.,0.,-math.sin(math.pi/4)],device=env.device).expand(env.num_envs,4)
        self.index=0;self.history=[];self.stable_time=0.

    def reset(self,ids):
        DistanceJacobianController.reset(self,ids)
        self.index=0;self.history=[];self.stable_time=0.
        self.select(self.env.cfg.connection_order[0])
        self.phase[ids]=Phase.LOAD;self.phase_time[ids]=0.;self.phase_durations[ids]=0.
        self.capture_time[ids]=0.;self.gap_goal[ids]=.055;self.failure_code[ids]=0
        s=state(self.env)
        for eq in s.variant_ids:self.env.sim.model.eq_solref[ids,int(eq),0]=.03

    def select(self,target):
        self.env.cfg.target_connector=target
        s=state(self.env);s.other_ids=[i for i in range(6) if i!=target]

    def advance(self,s):
        a=self.env.action_manager.get_term('joint_torque')
        self.history.append({'connector':self.env.cfg.target_connector,'durations_s':self.phase_durations[0].tolist(),
                             'joint_angle_rad':torch.cat([m.data.joint_pos for m in self.modules],-1)[0].tolist()})
        self.index+=1;self.select(self.env.cfg.connection_order[self.index] if self.index<5 else 5)
        self.phase.fill_(Phase.LOAD);self.phase_time.zero_();self.phase_durations.zero_()
        self.capture_time.zero_();self.gap_goal.fill_(.055);self.command.zero_();self.predicted_speed.zero_()
        self.angle_error.zero_();self.position_error.zero_();s.max_target_gap.zero_()
        a.pd_active.zero_();a.position_target.copy_(torch.cat([m.data.joint_pos for m in self.modules],-1))
        a.jacobian_torque.zero_();a.jacobian_gain.fill_(1.);a.loading.reset()
        a.jacobian_qref.copy_(a.position_target[0])

    def pose_command(self,s):
        full=super().pose_command(s)
        if self.phase.item()!=Phase.ROTATE:return full
        # Rotation needs clearance + orientation, not a fully fixed 3D point.
        # The two remaining joint-space freedoms let the grounded chain reshape.
        cfg=self.env.cfg;g=s.geometry();i=cfg.target_connector;other=(i+1)%6
        qb=s.modules[other].data.site_quat_w[:,s.in_sites[other]]
        relative=quat_mul(quat_conjugate(qb),g['q0'])
        angular=quat_apply(qb,rotvec(quat_mul(self.goal_q,quat_conjugate(relative))))
        omega=cfg.pose_orientation_gain*angular+1.5*g['dw']
        omega=omega/(omega.norm(dim=-1,keepdim=True)/cfg.rotation_speed).clamp_min(1.)
        row,distance=self.jacobian(s)
        rate=(g['delta']*g['dv']).sum(-1)/distance.clamp_min(1e-6)
        radial=(cfg.pose_position_gain*(.055-distance)-3.*rate).clamp(-cfg.pose_linear_speed,cfg.pose_linear_speed)
        anchors=torch.stack([m.data.body_link_pos_w[:,b] for m,b in zip(self.modules,self.tip_ids)],1)
        rotations=torch.stack([m.data.body_link_quat_w[:,b] for m,b in zip(self.modules,self.tip_ids)],1)
        z=torch.zeros_like(anchors);z[...,2]=1.;axes=quat_apply(rotations,z)
        J=torch.cat((row[:,None],.1*axes.transpose(1,2)),1)
        task=torch.cat((radial[:,None],.1*omega),1)
        H=J@J.transpose(1,2)+.005**2*torch.eye(4,device=self.env.device)[None]
        return (J.transpose(1,2)@torch.linalg.solve(H,task[:,:,None])).squeeze(-1)

    def update(self,s,enabled):
        if self.index<5:
            result=super().update(s,enabled)
            if self.phase.item()==Phase.DONE:self.advance(s)
            return result
        # The five internal edges are now RZ270; open the temporary head-tail brace.
        cfg=self.env.cfg;dt=self.env.physics_dt;self.phase_time.add_(dt)
        if self.phase.item()==Phase.LOAD and not s.connected()[0,5]:
            self.transition(torch.ones(1,dtype=torch.bool,device=self.env.device),SequencePhase.UNFOLD)
        q=torch.cat([m.data.joint_pos for m in self.modules],-1)
        dq=torch.cat([m.data.joint_vel for m in self.modules],-1)
        pose_ok=(q.abs().amax()<cfg.snake_angle_tolerance)&(dq.abs().amax()<cfg.snake_velocity_tolerance)
        if self.phase.item() in (SequencePhase.UNFOLD,SequencePhase.SNAKE_HOLD):
            desired=(-cfg.unfold_gain*q).clamp(-cfg.unfold_speed,cfg.unfold_speed)
            step=2.*dt
            self.command.add_((desired-self.command).clamp(-step,step))
            if pose_ok:
                if self.phase.item()==SequencePhase.UNFOLD:
                    self.transition(torch.ones(1,dtype=torch.bool,device=self.env.device),SequencePhase.SNAKE_HOLD)
                self.stable_time+=dt
            else:self.stable_time=0.
            if self.stable_time>=cfg.settle_s:
                self.transition(torch.ones(1,dtype=torch.bool,device=self.env.device),Phase.DONE)
        else:self.command.zero_()
        bad=failure(self.env).any() or not s.connected()[0,:5].all()
        timeout=self.phase_time.item()>cfg.unfold_timeout_s
        if (bad or timeout) and self.phase.item()!=Phase.DONE:
            self.failure_code.fill_(1 if bad else 2)
            self.transition(torch.ones(1,dtype=torch.bool,device=self.env.device),Phase.FAILED)
            self.command.zero_()
        self.command.masked_fill_(~enabled[:,None],0.)
        s.waiting.copy_(self.phase==Phase.FAILED)
        return self.command


@dataclass(kw_only=True)
class SequentialActionCfg(TorqueActionCfg):
    def build(self,env):return SequentialAction(self,env)


class SequentialAction(TorqueAction):
    def __init__(self,cfg,env):
        enabled=env.cfg.jacobian_feedforward;env.cfg.jacobian_feedforward=False
        try:super().__init__(cfg,env)
        finally:env.cfg.jacobian_feedforward=enabled
        env.sim.expand_model_fields(('eq_solref',))
        env._disconnect_state=SequentialConnectorState(env)
        self.distance_controller=SequentialController(env,self.modules)
        self.loading=LiveLoadingController(env)
        self.jacobian_report={'method':'live_measured_constraint_J_M_R','experiment':EXPERIMENT_ID}

    def update_jacobian_gain(self,s):pass

    def process_actions(self,actions):
        c=self.distance_controller;s=state(self._env)
        if c.phase.item()==Phase.LOAD and s.connected().all():
            torque=self.loading.update(s)
            self.jacobian_torque.copy_(torch.as_tensor(torque,device=self._env.device,dtype=self._raw.dtype))
            self.jacobian_gain.fill_(1.)
            self.jacobian_qref.copy_(torch.cat([m.data.joint_pos for m in self.modules],-1)[0])
        super().process_actions(torch.zeros_like(self._raw))

    def apply_actions(self):
        state(self._env).waiting.copy_(self.distance_controller.phase==Phase.FAILED)
        super().apply_actions()


def sequence_index(env):return torch.as_tensor(env.action_manager.get_term('joint_torque').distance_controller.index,device=env.device).expand(env.num_envs)


def sequential_rz270_env_cfg(num_envs=1,order=(0,1,2,3,4),acceleration=8.):
    if num_envs!=1:raise ValueError('This independent sequential experiment currently runs one physical robot')
    base=state_machine_env_cfg(num_envs)
    cfg=SequentialRz270EnvCfg(**{f.name:getattr(base,f.name) for f in fields(base)})
    cfg.scene.spec_fn=add_sequential_connectors;cfg.connection_order=tuple(order);cfg.target_connector=order[0]
    cfg.separation_joint_acceleration=acceleration
    cfg.actions={'joint_torque':SequentialActionCfg(entity_name='module_0')}
    cfg.episode_length_s=300.;cfg.terminations={}
    cfg.metrics['sequence_index']=MetricsTermCfg(func=sequence_index)
    return cfg
