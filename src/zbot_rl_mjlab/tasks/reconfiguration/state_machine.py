"""Independent deterministic reconfiguration; no runner, policy or learning."""
import math
from dataclasses import dataclass,fields
from enum import IntEnum

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply,quat_apply_inverse,quat_mul,quat_conjugate
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from .disconnect.env_cfg import DisconnectTorqueEnvCfg,TorqueAction,TorqueActionCfg,disconnect_env_cfg
from .disconnect.state import ConnectorState,state,failure
from .disconnect.separation import DistanceJacobianController
from .disconnect.model import add_connectors


class Phase(IntEnum):
    LOAD=0
    SEPARATE=1
    ROTATE=2
    APPROACH=3
    SETTLE=4
    DONE=5
    FAILED=6


@dataclass
class StateMachineEnvCfg(DisconnectTorqueEnvCfg):
    reconfiguration_state_machine: bool=True
    separation_target_error_limit: float=.5
    rotation_angle: float=math.pi/2
    rotation_speed: float=6.
    pose_linear_speed: float=.15
    approach_speed: float=.15
    capture_distance: float=.002
    capture_angle: float=2*math.pi/180
    capture_speed: float=.03
    capture_angular_speed: float=.3
    capture_hold_s: float=.03
    settle_s: float=2.
    weld_ramp_s: float=.2
    phase_timeout_s: float=40.
    motion_duration_s: float=1.
    pose_position_gain: float=60.
    pose_orientation_gain: float=15.


def add_reconfiguration_connectors(spec):
    add_connectors(spec)
    original=spec.site('module_5/port_out')
    body=original.parent
    q=np.empty(4);mujoco.mju_mulQuat(q,original.quat,np.array([math.cos(-math.pi/4),0.,0.,math.sin(-math.pi/4)]))
    body.add_site(name='module_5/port_out_rot90',pos=original.pos,quat=q,size=(.004,)*3)
    spec.add_equality(name='connector_5_rot90',type=mujoco.mjtEq.mjEQ_WELD,
                      objtype=mujoco.mjtObj.mjOBJ_SITE,name1='module_5/port_out_rot90',name2='module_0/port_in',
                      active=False,data=(*([0.]*10),1.),solref=(.03,1.),solimp=(.999,.9999,.001,.5,2.))


class ReconfigurationConnectorState(ConnectorState):
    def __init__(self,env):
        super().__init__(env)
        self.new_eq=int(env.sim.mj_model.equality('connector_5_rot90').id)

    def constraint_ids(self):
        ids=super().constraint_ids().clone()
        ids[:,5]=torch.where(self.env.sim.data.eq_active[:,self.new_eq].bool(),self.new_eq,self.eq_ids[5])
        return ids

    def reset(self,ids):
        ids=slice(None) if ids is None else ids
        super().reset(ids)
        self.env.sim.data.eq_active[ids,self.new_eq]=False

    def update_separation(self,next_active):
        # This standalone sequence continues after 5cm, instead of resetting.
        self.max_target_gap.copy_(torch.maximum(self.max_target_gap,self.geometry()['distance']))
        self.safety_margin_violation |= (1.-self.other_margin())>(1.-self.cfg.minimum_other_margin)


def rotvec(q):
    q=torch.where(q[:,:1]<0,-q,q)
    v=q[:,1:];length=v.norm(dim=-1,keepdim=True)
    return v*(2*torch.atan2(length,q[:,:1].clamp_min(0.)))/length.clamp_min(1e-8)


class ReconfigurationController(DistanceJacobianController):
    def __init__(self,env,modules):
        super().__init__(env,modules)
        n=env.num_envs;d=env.device
        self.phase=torch.zeros(n,dtype=torch.long,device=d)
        self.phase_time=torch.zeros(n,device=d)
        self.phase_durations=torch.zeros(n,len(Phase),device=d)
        self.capture_time=torch.zeros(n,device=d)
        self.gap_goal=torch.full((n,),.055,device=d)
        self.angle_error=torch.zeros(n,device=d)
        self.position_error=torch.zeros(n,device=d)
        self.failure_code=torch.zeros(n,dtype=torch.long,device=d)
        self.goal_q=torch.tensor([math.cos(math.pi/4),0.,0.,math.sin(math.pi/4)],device=d).expand(n,4)

    def reset(self,ids):
        super().reset(ids)
        self.env.sim.model.eq_solref[ids,state(self.env).new_eq,0]=.03
        self.phase[ids]=Phase.LOAD;self.phase_time[ids]=0.;self.phase_durations[ids]=0.;self.capture_time[ids]=0.
        self.gap_goal[ids]=.055;self.angle_error[ids]=0.;self.position_error[ids]=0.;self.failure_code[ids]=0

    def transition(self,mask,phase):
        mask=mask & (self.phase!=phase)
        if phase!=Phase.FAILED:
            timed=(self.phase==Phase.SEPARATE)|(self.phase==Phase.ROTATE)|(self.phase==Phase.APPROACH)
            mask &= ~timed | (self.phase_time<=self.env.cfg.motion_duration_s+1e-6)
        previous=self.phase_durations.gather(1,self.phase[:,None]).squeeze(-1)
        self.phase_durations.scatter_(1,self.phase[:,None],torch.where(mask,self.phase_time,previous)[:,None])
        self.phase.copy_(torch.where(mask,phase,self.phase))
        self.phase_time.masked_fill_(mask,0.)

    def pose_command(self,s):
        # Open-chain source pose relative to the anchored destination inlet.
        g=s.geometry();target=s.cfg.target_connector;other=(target+1)%6
        dst=s.modules[other];qb=dst.data.site_quat_w[:,s.in_sites[other]]
        qa=g['q0'];current=quat_mul(quat_conjugate(qb),qa)
        angular=quat_apply(qb,rotvec(quat_mul(self.goal_q,quat_conjugate(current))))
        self.angle_error.copy_(angular.norm(dim=-1))
        z=angular.new_tensor([0.,0.,1.]).expand_as(angular)
        desired=-quat_apply(qb,z)*self.gap_goal[:,None]
        linear=desired+g['delta']
        self.position_error.copy_(linear.norm(dim=-1))
        v=self.env.cfg.pose_position_gain*linear+3.*g['dv'];v=v/(v.norm(dim=-1,keepdim=True)/self.env.cfg.pose_linear_speed).clamp_min(1.)
        omega=self.env.cfg.pose_orientation_gain*angular+1.5*g['dw'];omega=omega/(omega.norm(dim=-1,keepdim=True)/self.env.cfg.rotation_speed).clamp_min(1.)
        anchors=torch.stack([m.data.body_link_pos_w[:,b] for m,b in zip(self.modules,self.tip_ids)],1)
        rotations=torch.stack([m.data.body_link_quat_w[:,b] for m,b in zip(self.modules,self.tip_ids)],1)
        axes=quat_apply(rotations,z[:,None].expand_as(anchors))
        source=s.modules[target].data.site_pos_w[:,s.out_sites[target]]
        Jp=torch.cross(axes,source[:,None]-anchors,dim=-1).transpose(1,2)
        J=torch.cat((Jp,.1*axes.transpose(1,2)),1)
        task=torch.cat((v,.1*omega),1)
        H=J@J.transpose(1,2)+.005**2*torch.eye(6,device=self.env.device)[None]
        desired_velocity=(J.transpose(1,2)@torch.linalg.solve(H,task[:,:,None])).squeeze(-1)
        return desired_velocity

    def update(self,s,enabled):
        cfg=self.env.cfg;dt=self.env.physics_dt
        self.phase_time.add_(dt)
        broken=~s.connected()[:,cfg.target_connector]
        self.transition((self.phase==Phase.LOAD)&broken,Phase.SEPARATE)
        self.transition((self.phase==Phase.SEPARATE)&(s.geometry()['distance']>.05),Phase.ROTATE)
        pose=self.pose_command(s)
        self.transition((self.phase==Phase.ROTATE)&(self.angle_error<math.radians(1))
                        &(s.geometry()['distance']>=.05)&(s.geometry()['dw'].norm(dim=-1)<.5),Phase.APPROACH)
        approach=self.phase==Phase.APPROACH
        descent=approach & (self.angle_error<math.radians(5))
        self.gap_goal.copy_(torch.where(descent,(self.gap_goal-cfg.approach_speed*dt).clamp_min(0.),self.gap_goal))
        g=s.geometry()
        eligible=(approach & (g['distance']<cfg.capture_distance)&(self.angle_error<cfg.capture_angle)
                  &(g['dv'].norm(dim=-1)<cfg.capture_speed)&(g['dw'].norm(dim=-1)<cfg.capture_angular_speed))
        self.capture_time.copy_(torch.where(eligible,self.capture_time+dt,0.))
        capture=eligible & (self.capture_time>=cfg.capture_hold_s)
        self.env.sim.data.eq_active[:,s.new_eq] |= capture
        self.env.action_manager.get_term('joint_torque').position_target.copy_(torch.where(
            capture[:,None],torch.cat([m.data.joint_pos for m in self.modules],-1),
            self.env.action_manager.get_term('joint_torque').position_target))
        self.transition(capture,Phase.SETTLE)
        settling=self.phase==Phase.SETTLE
        blend=(self.phase_time/cfg.weld_ramp_s).clamp(0.,1.)
        self.env.sim.model.eq_solref[:,s.new_eq,0]=torch.where(settling,.03+(.005-.03)*blend,torch.where(self.phase==Phase.DONE,.005,.03))
        self.transition(settling&(self.phase_time>=cfg.settle_s+cfg.weld_ramp_s)&s.connected()[:,cfg.target_connector],Phase.DONE)
        bad=failure(self.env) | ((self.phase==Phase.SETTLE)&~s.connected()[:,cfg.target_connector])
        timeout=(self.phase_time>cfg.phase_timeout_s)&(self.phase<Phase.DONE)
        deadline=((self.phase==Phase.SEPARATE)|(self.phase==Phase.ROTATE)|(self.phase==Phase.APPROACH)) & (self.phase_time>=cfg.motion_duration_s)
        self.failure_code.copy_(torch.where(bad,1,torch.where(deadline,3,torch.where(timeout,2,self.failure_code))))
        self.transition((bad|timeout|deadline)&(self.phase!=Phase.DONE),Phase.FAILED)
        previous=self.command.clone()
        separating=(self.phase==Phase.SEPARATE)&enabled
        radial=super().update(s,separating).clone()
        moving=((self.phase==Phase.ROTATE)|(self.phase==Phase.APPROACH))&enabled
        pose=pose/(pose.abs().amax(-1,keepdim=True)/cfg.separation_max_target_speed).clamp_min(1.)
        step=cfg.separation_joint_acceleration*dt
        pose=previous+(pose-previous).clamp(-step,step)
        desired=torch.where(moving[:,None],pose,radial)
        self.command.copy_(desired)
        self.command.masked_fill_(~(moving|separating)[:,None],0.)
        s.waiting.copy_(self.phase==Phase.FAILED)
        return self.command


@dataclass(kw_only=True)
class StateMachineActionCfg(TorqueActionCfg):
    def build(self,env):return StateMachineAction(self,env)


class StateMachineAction(TorqueAction):
    def __init__(self,cfg,env):
        if env.cfg.target_connector!=5 or not math.isclose(env.cfg.rotation_angle,math.pi/2):
            raise ValueError('Standalone state machine currently supports target 5 and +90 degrees')
        for name in ('motion_duration_s','rotation_speed','pose_linear_speed','approach_speed',
                     'capture_distance','capture_angle','capture_speed','capture_angular_speed',
                     'capture_hold_s','settle_s','weld_ramp_s','phase_timeout_s','pose_position_gain','pose_orientation_gain'):
            if not math.isfinite(getattr(env.cfg,name)) or getattr(env.cfg,name)<=0:
                raise ValueError(f'{name} must be positive and finite')
        super().__init__(cfg,env)
        if not self.native_pd:raise ValueError('State machine requires MJLab BuiltinPositionActuator')
        env.sim.expand_model_fields(('eq_solref',))
        env._disconnect_state=ReconfigurationConnectorState(env)
        self.distance_controller=ReconfigurationController(env,self.modules)

    def process_actions(self,actions):
        # External action is deliberately ignored: standalone deterministic FSM.
        super().process_actions(torch.zeros_like(self._raw))

    def apply_actions(self):
        c=self.distance_controller;s=state(self._env)
        # Failures retain damping; completed reconnection holds the PD target.
        s.waiting.copy_(c.phase==Phase.FAILED)
        super().apply_actions()


def phase_metric(env):return env.action_manager.get_term('joint_torque').distance_controller.phase.float()
def angle_metric(env):return env.action_manager.get_term('joint_torque').distance_controller.angle_error

def fsm_done(env):return env.action_manager.get_term('joint_torque').distance_controller.phase==Phase.DONE

def fsm_failed(env):return env.action_manager.get_term('joint_torque').distance_controller.phase==Phase.FAILED


def state_machine_env_cfg(num_envs=1):
    base=disconnect_env_cfg(play=True,native_pd=True)
    cfg=StateMachineEnvCfg(**{f.name:getattr(base,f.name) for f in fields(base)})
    cfg.scene.num_envs=num_envs;cfg.scene.spec_fn=add_reconfiguration_connectors
    cfg.jacobian_feedforward=True;cfg.separation_velocity_pd=True;cfg.separation_distance_jacobian=True
    cfg.separation_target_error_limit=.5
    cfg.separation_cartesian_speed=.15;cfg.separation_distance_gain=20.
    cfg.separation_max_target_speed=6.;cfg.separation_joint_acceleration=30.
    cfg.auto_reset=False;cfg.episode_length_s=120.
    cfg.actions={'joint_torque':StateMachineActionCfg(entity_name='module_0')}
    cfg.rewards={}
    cfg.terminations={'failure':TerminationTermCfg(func=fsm_failed),'done':TerminationTermCfg(func=fsm_done)}
    cfg.metrics['success']=MetricsTermCfg(func=fsm_done,reduce='max')
    cfg.metrics['fsm_phase']=MetricsTermCfg(func=phase_metric)
    cfg.metrics['fsm_orientation_error_rad']=MetricsTermCfg(func=angle_metric)
    return cfg
