"""Experimental full-FSM GPU action; not registered for production training."""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_mul,quat_conjugate,quat_apply
from .state_machine import rotvec
from .disconnect.model import TIP_NAMES
from .disconnect.env_cfg import TorqueAction,TorqueActionCfg
from .disconnect.state import state
from .batched_loading import BatchedLoading
from .batched_teacher_reference import BatchedTeacherReference
from .full_batched_state import FullBatchedState
from .full_batched_fsm import P

REFERENCE=Path('outputs/reconfiguration/full_cpu_student/teacher_reference.npz')

@dataclass(kw_only=True)
class FullBatchedActionCfg(TorqueActionCfg):
    def build(self,env):return FullBatchedAction(self,env)

class FullBatchedAction(TorqueAction):
    def __init__(self,cfg,env):
        ff=env.cfg.jacobian_feedforward;sep=env.cfg.separation_distance_jacobian
        env.cfg.jacobian_feedforward=False;env.cfg.separation_distance_jacobian=False
        try:super().__init__(cfg,env)
        finally:env.cfg.jacobian_feedforward=ff;env.cfg.separation_distance_jacobian=sep
        env.sim.expand_model_fields(('eq_solref',))
        env._disconnect_state=FullBatchedState(env)
        model=env.sim.mj_model
        self.joint_qpos_ids=torch.tensor([int(model.joint(f'module_{i}/joint{i+1}').qposadr[0]) for i in range(6)],device=env.device)
        self.joint_dof_ids=torch.tensor([int(model.jnt_dofadr[model.joint(f'module_{i}/joint{i+1}').id]) for i in range(6)],device=env.device)
        self.tip_body_ids=torch.tensor([int(model.body(f'module_{i}/{TIP_NAMES[i]}').id) for i in range(6)],device=env.device)
        self.policy=torch.zeros(env.num_envs,7,device=env.device)
        self.position_target=torch.cat([m.data.joint_pos for m in self.modules],-1).clone()
        self.loading=BatchedLoading(env,async_linalg=True);reference=Path(getattr(env.cfg,"full_teacher_reference",REFERENCE))
        self.online_planner=None
        if getattr(env.cfg,'full_online_headtail',False):
            from .online_headtail_planner import OnlineHeadtailPlanner
            self.online_planner=OnlineHeadtailPlanner(env);self.reference=None
        else:self.reference=BatchedTeacherReference(reference,env.device)
        if self.reference is not None and (getattr(env.cfg,'full_teacher_load_replay',False) or getattr(env.cfg,'full_teacher_headtail_replay',False)) and not (self.reference.has_control and self.reference.has_connected):
            raise ValueError('teacher loading replay requires recorded control and connection samples')
        self.nominal=torch.zeros_like(self.position_target);self.nominal_vel=torch.zeros_like(self.nominal)
        self.loading_qhold=torch.zeros_like(self.position_target)
        self.teacher_headtail_ctrl=torch.zeros_like(self.position_target)
        self.teacher_velocity=torch.zeros_like(self.position_target)
        self.motion_velocity=torch.zeros_like(self.position_target)
        self.curl_residual_velocity=torch.zeros_like(self.position_target)
        self.curl_residual_offset=torch.zeros_like(self.position_target)
        self.pose_potential_before=torch.zeros(env.num_envs,device=env.device)
        self.pose_progress_active=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.policy_torque=torch.zeros_like(self.position_target)
        self.last_phase=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        self.progress_before=torch.zeros(env.num_envs,device=env.device)
        self.actuation_graph=None;self.actuation_ready=False
        self.tip_ids=[m.find_bodies(TIP_NAMES[i])[0][0] for i,m in enumerate(self.modules)]
        self.phase_ends=torch.zeros(len(P)*30,device=env.device)
        if self.reference is not None:
            with np.load(reference,allow_pickle=False) as d:
                key=d['phase']*30+d['target']*5+d['stage'];ends=np.zeros(len(P)*30,dtype=np.float32)
                np.maximum.at(ends,key,d['phase_time']);self.phase_ends.copy_(torch.as_tensor(ends,device=env.device))
    @property
    def action_dim(self):return 7
    @property
    def raw_action(self):return self.policy
    def joint_state(self):
        data=self._env.sim.data
        return data.qpos[:,self.joint_qpos_ids],data.qvel[:,self.joint_dof_ids]
    @property
    def torque(self):
        return self._env.sim.data.qfrc_actuator[:,self.joint_dof_ids]
    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        # This action does not use the base distance controller.
        super().reset(env_ids)
        self.policy[ids]=0.;self.loading.reset(ids);self.nominal[ids]=0.;self.nominal_vel[ids]=0.;self.loading_qhold[ids]=0.;self.teacher_headtail_ctrl[ids]=0.;self.teacher_velocity[ids]=0.;self.motion_velocity[ids]=0.;self.curl_residual_velocity[ids]=0.;self.curl_residual_offset[ids]=0.;self.policy_torque[ids]=0.;self.progress_before[ids]=0.
        self.pose_potential_before[ids]=0.;self.pose_progress_active[ids]=False
    def approach_correction(self,s,g,goal):
        data=self._env.sim.data
        positions=data.xpos[:,self.tip_body_ids]
        rotations=data.xquat[:,self.tip_body_ids]
        z=torch.zeros_like(positions);z[...,2]=1.;axes=quat_apply(rotations,z)
        source=data.site_xpos[s.worlds,s.capture_port_ids[s.target_indices()]]
        linear=torch.cross(axes,source[:,None,:]-positions,dim=-1).transpose(1,2)
        angular=axes.transpose(1,2)
        relative=quat_mul(quat_conjugate(g["q1"]),g["q0"])
        orientation=quat_apply(g["q1"],rotvec(quat_mul(goal,quat_conjugate(relative))))
        J=torch.cat((linear,angular*.05),1);desired=torch.cat((g["delta"],orientation*.05),-1)
        eye=torch.eye(6,device=self._env.device,dtype=J.dtype)[None]
        solution,info=torch.linalg.solve_ex(J@J.transpose(1,2)+.01**2*eye,desired[...,None],check_errors=False)
        delta=(J.transpose(1,2)@solution).squeeze(-1)
        valid=(info==0)&torch.isfinite(delta).all(-1)
        return torch.where(valid[:,None],delta.clamp(-.08,.08),0.)
    def process_actions(self,actions):
        if self.online_planner is not None:
            return self.process_online_actions(actions)
        e=self._env;s=state(e);f=s.fsm;s.begin_step();self.peak_overload_cost.zero_()
        if getattr(e.cfg,'full_policy_curl_only',False):
            actions=torch.where((f.phase==P['CURL'])[:,None],actions,torch.zeros_like(actions))
        self.policy.lerp_(torch.nan_to_num(actions).clamp(-1.,1.),.1)
        old=f.phase.clone();self.last_phase.copy_(old)
        if getattr(e.cfg,'full_pose_progress_reward',False):
            pose=s.capture_geometry()
            self.pose_potential_before.copy_(torch.exp(-pose['gap']/.15-pose['angle']/.75))
            self.pose_progress_active.copy_(((old==P['CURL'])|(old==P['CAPTURE_HEADTAIL']))&(f.target==5))
        self.progress_before.copy_(f.stage.float()*4.+f.phase.float()*.01)
        g=s.geometry();target=s.target_indices();sign=torch.where(target<2,-1.,1.)
        goal=torch.stack((torch.full_like(sign,math.sqrt(.5)),torch.zeros_like(sign),torch.zeros_like(sign),sign*math.sqrt(.5)),-1)
        relative=quat_mul(quat_conjugate(g['q1']),g['q0']);err=quat_mul(goal,quat_conjugate(relative))
        angle=2.*torch.atan2(err[:,1:].norm(dim=-1),err[:,0].abs().clamp_min(1e-8))
        angle=torch.where(target==5,g['angle'],angle)
        q,v=self.joint_state()
        index=(f.phase*30+target*5+f.stage).clamp(0,len(self.phase_ends)-1)
        done=f.phase_time>=self.phase_ends[index]-.01
        angle_tolerance=float(getattr(e.cfg,'full_final_angle_tolerance',.02))
        straight=s.terminal_ready(q,v,angle_tolerance)
        if getattr(e.cfg,'full_terminal_recheck',False):
            retry=(old==P['DONE'])&~straight
            f.phase.masked_fill_(retry,P['FINAL_STRAIGHT_HOLD'])
            f.phase_time.masked_fill_(retry,0.);f.phase_elapsed.masked_fill_(retry,0.)
            old=torch.where(retry,P['FINAL_STRAIGHT_HOLD'],old)
        geometry={'gap':g['distance'],'angle':angle,'linear_speed':g['dv'].norm(dim=-1),'angular_speed':g['dw'].norm(dim=-1)}
        capture=(f.phase==P['APPROACH'])|(f.phase==P['CAPTURE_HEADTAIL'])
        actual_capture=s.capture_geometry()
        geometry={key:torch.where(capture,actual_capture[key],value) for key,value in geometry.items()}
        move=(old==P['CURL'])|(old==P['SEPARATE'])|(old==P['ROTATE'])|(old==P['APPROACH'])
        reference_rate=float(getattr(e.cfg,'full_reference_rate',1.))
        # A slowed unfold must advance its reference clock at the same rate
        # as its velocity, or the reference expires before the motion ends.
        if reference_rate!=1.:move|=(old==P['UNFOLD'])
        if getattr(e.cfg,'full_policy_curl_only',False):
            fraction=float(getattr(e.cfg,'full_curl_clock_fraction',0.))
            clock_action=2.*fraction*self.policy[:,6]*(old==P['CURL'])
        else:clock_action=self.policy[:,6]
        elapsed=e.step_dt*torch.where(move,reference_rate*(1.+.5*clock_action),1.)
        # Slowest allowed phase-clock rate is 0.5x; independently bound
        # actual simulation time so policy clock changes cannot hide a stall.
        curve_timeout=move&((f.phase_time>self.phase_ends[index]+4.)|(f.phase_elapsed>(2./reference_rate)*self.phase_ends[index]+4.))
        events=f.update(s.connected(),e.sim.data.eq_active[:,s.variant_ids].bool(),s.utilization*e.cfg.break_force,geometry,done,straight,s.contact_violation,self.torque_failure,dt=elapsed,curve_timeout=curve_timeout)
        e.sim.data.eq_active[s.worlds,s.eq_ids[5]]|=events['capture_headtail']
        eq=s.variant_ids[f.target.clamp_max(4)]
        e.sim.data.eq_active[s.worlds,eq]|=events['capture_internal']
        settling=(f.phase==P['SETTLE'])|(f.phase==P['SETTLE_HEADTAIL'])
        blend=(f.phase_time/.2).clamp(0.,1.);eq=torch.where(f.target==5,s.eq_ids[5],s.variant_ids[f.target.clamp_max(4)])
        prev=e.sim.model.eq_solref[s.worlds,eq,0]
        e.sim.model.eq_solref[s.worlds,eq,0]=torch.where(settling,.03-.025*blend,prev)
        s.set_contact_phase()
        ref=self.reference.sample(f.phase,f.target,f.stage,f.phase_time)
        moving=(f.phase==P['CURL'])|(f.phase==P['CAPTURE_HEADTAIL'])|(f.phase==P['SEPARATE'])|(f.phase==P['ROTATE'])|(f.phase==P['APPROACH'])|(f.phase==P['UNFOLD'])
        phase_key=(f.phase*30+f.target*5+f.stage).clamp(0,len(self.phase_ends)-1)
        ref_active=ref['valid']&moving&(f.phase_time<self.phase_ends[phase_key])
        reference_velocity=ref['qref_velocity'] if getattr(e.cfg,'full_teacher_target_velocity',False) else ref['qvel']
        position_tracking=getattr(e.cfg,'full_teacher_separation_position',False)
        if position_tracking:
            position_phase=(f.phase==P['SEPARATE'])
            reference_velocity=torch.where(position_phase[:,None],ref['qref_slope'],reference_velocity)
        self.teacher_velocity.copy_(torch.where(ref_active[:,None],reference_velocity,0.))
        if getattr(e.cfg,'full_curl_symmetric',False):
            signs=self.teacher_velocity.new_tensor([-1.,1.,-1.,1.,-1.,1.])
            symmetric=(self.teacher_velocity*signs).mean(-1,keepdim=True)*signs
            self.teacher_velocity.copy_(torch.where((f.phase==P['CURL'])[:,None],symmetric,self.teacher_velocity))
        velocity_scale=float(getattr(e.cfg,'full_velocity_scale',.5))
        residual=self.policy[:,:6]
        if getattr(e.cfg,'full_policy_curl_only',False):residual=residual*(f.phase==P['CURL'])[:,None]
        if getattr(e.cfg,'full_policy_curl_only',False):self.curl_residual_velocity.copy_(velocity_scale*residual)
        else:self.curl_residual_velocity.zero_()
        speed_clock=clock_action*(f.phase==P['CURL']) if getattr(e.cfg,'full_policy_curl_only',False) else clock_action
        speed=reference_rate*(1.+.5*speed_clock[:,None])*self.teacher_velocity+velocity_scale*residual
        approaching=(f.phase==P['APPROACH'])|(f.phase==P['CAPTURE_HEADTAIL'])
        identity=torch.stack((torch.ones_like(sign),torch.zeros_like(sign),torch.zeros_like(sign),torch.zeros_like(sign)),-1)
        # Fixed-size GPU batch avoids the previous per-step host synchronization.
        correction=self.approach_correction(s,s.capture_geometry(),identity)
        gain=torch.where(ref_active,.02,2.)[:,None]
        speed=speed+approaching[:,None]*(gain*correction).clamp(-.25,.25)
        speed_limit=7.2 if getattr(e.cfg,'full_refined_reference',False) else 3.
        self.motion_velocity.copy_(torch.where(moving[:,None],speed.clamp(-speed_limit,speed_limit),0.))
        changed=old!=f.phase
        if getattr(e.cfg,'full_policy_curl_only',False):
            self.curl_residual_offset.masked_fill_(changed[:,None],0.)
        self.position_target.copy_(torch.where(changed[:,None],q,self.position_target))
        if position_tracking:
            track=position_phase&ref['valid']
            self.position_target.copy_(torch.where(track[:,None],ref['qref_interp'],self.position_target))
        self.nominal.copy_(self.position_target);self.nominal_vel.copy_(self.teacher_velocity)
        loading=((f.phase==P['LOAD'])|(f.phase==P['LOAD_HEADTAIL']))&s.connected().all(-1)
        if getattr(e.cfg,'full_stop_after_initial_closure',False):
            loading &= ~((f.phase==P['LOAD'])&(f.stage==0)&(f.target==0))
        # Solve only genuine load stages; no policy can command a release.
        entering=(old!=f.phase)&loading
        self.loading.reset(entering)
        self.loading_qhold.copy_(torch.where(entering[:,None],q,self.loading_qhold))
        internal=loading&(f.target<5)&getattr(e.cfg,'full_teacher_load_replay',False)
        headtail=loading&(f.target==5)&getattr(e.cfg,'full_teacher_headtail_replay',False)
        if getattr(e.cfg,'full_teacher_load_replay',False):
            replay=ref['load_ctrl_extended'] if getattr(e.cfg,'full_teacher_load_extend',False) else ref['load_ctrl']
            self.loading.torque.copy_(torch.where((internal&ref['valid'])[:,None],replay,self.loading.torque))
        if getattr(e.cfg,'full_teacher_headtail_replay',False):
            headtail_ctrl=ref['load_ctrl_extended'] if getattr(e.cfg,'full_teacher_load_extend',False) else ref['load_ctrl']
            self.teacher_headtail_ctrl.copy_(torch.where((headtail&ref['valid'])[:,None],headtail_ctrl,self.teacher_headtail_ctrl))
        self.loading.update(s,loading&~internal&~headtail)

    def process_online_actions(self,actions):
        e=self._env;s=state(e);f=s.fsm;s.begin_step();self.peak_overload_cost.zero_()
        actions=torch.where((f.phase==P['CURL'])[:,None],actions,torch.zeros_like(actions))
        self.policy.lerp_(torch.nan_to_num(actions).clamp(-1.,1.),.1)
        old=f.phase.clone();self.last_phase.copy_(old)
        g=s.capture_geometry();q,v=self.joint_state()
        geometry={key:g[key] for key in ('gap','angle','linear_speed','angular_speed')}
        curve_done=(g['gap']<.05)&(g['angle']<.8)
        straight=s.terminal_ready(q,v,float(getattr(e.cfg,'full_final_angle_tolerance',.02)))
        events=f.update(s.connected(),e.sim.data.eq_active[:,s.variant_ids].bool(),s.utilization*e.cfg.break_force,
                        geometry,curve_done,straight,s.contact_violation,self.torque_failure,dt=e.step_dt)
        e.sim.data.eq_active[s.worlds,s.eq_ids[5]]|=events['capture_headtail']
        settling=f.phase==P['SETTLE_HEADTAIL'];blend=(f.phase_time/.2).clamp(0.,1.)
        eq=s.eq_ids[5];prev=e.sim.model.eq_solref[:,eq,0]
        e.sim.model.eq_solref[:,eq,0]=torch.where(settling,.03-.025*blend,prev)
        s.set_contact_phase();changed=old!=f.phase
        self.position_target.copy_(torch.where(changed[:,None],q,self.position_target))
        self.curl_residual_offset.masked_fill_(changed[:,None],0.)
        self.curl_residual_velocity.copy_(float(e.cfg.full_velocity_scale)*self.policy[:,:6]*(f.phase==P['CURL'])[:,None])
        self.teacher_velocity.zero_();self.nominal.copy_(self.position_target);self.nominal_vel.zero_()
    def apply_actions(self):
        if not getattr(self._env.cfg,'full_cuda_graphs',False):
            return self._apply_actions_eager()
        if not self.actuation_ready:
            self._apply_actions_eager();self.actuation_ready=True
        elif self.actuation_graph is None:
            self.actuation_graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.actuation_graph):
                self._apply_actions_eager()
            self.actuation_graph.replay()
        else:
            self.actuation_graph.replay()
    def _apply_actions_eager(self):
        e=self._env;s=state(e);f=s.fsm
        q,v=self.joint_state()
        active=(f.phase!=P['FAILED'])&~self.torque_failure
        if getattr(e.cfg,'full_headtail_penetration_guard',False):active &= ~s.headtail_penetration
        if not getattr(e.cfg,'full_terminal_recheck',False):active &= f.phase!=P['DONE']
        if getattr(e.cfg,'full_enforce_other_margin',False):
            active &= ~s.safety_margin_violation
        loading=((f.phase==P['LOAD'])|(f.phase==P['LOAD_HEADTAIL']))&s.connected()[s.worlds,f.target]&active
        initial_closure=(f.phase==P['LOAD'])&(f.stage==0)&(f.target==0)
        if getattr(e.cfg,'full_stop_after_initial_closure',False):loading &= ~initial_closure
        motion=(f.phase==P['CURL'])|(f.phase==P['CAPTURE_HEADTAIL'])|(f.phase==P['SEPARATE'])|(f.phase==P['ROTATE'])|(f.phase==P['APPROACH'])|(f.phase==P['UNFOLD'])
        handoff=((f.phase==P['LOAD'])|(f.phase==P['LOAD_HEADTAIL']))&~s.connected()[s.worlds,f.target]
        outer=handoff|(f.phase==P['INITIAL_STRAIGHT_HOLD'])|(f.phase==P['CURL'])|(f.phase==P['CAPTURE_HEADTAIL'])|(f.phase==P['SETTLE_HEADTAIL'])|(f.phase==P['UNFOLD'])|(f.phase==P['FINAL_STRAIGHT_HOLD'])
        if getattr(e.cfg,'full_stop_after_initial_closure',False):outer|=initial_closure
        if getattr(e.cfg,'full_terminal_recheck',False):outer|=f.phase==P['DONE']
        kp=torch.where(outer,float(getattr(e.cfg,"full_outer_stiffness",80.)),float(getattr(e.cfg,"full_inner_stiffness",400.)))*active;kd=torch.where(outer,float(getattr(e.cfg,"full_outer_damping",4.)),float(getattr(e.cfg,"full_inner_damping",4.)))*active
        # Integrate a velocity target at every physics substep.  No position
        # trajectory samples are sent to the native position actuator.
        velocity=self.motion_velocity
        if getattr(e.cfg,'full_policy_curl_only',False):
            limit=float(getattr(e.cfg,'full_curl_offset_limit',.02))
            next_offset=(self.curl_residual_offset+self.curl_residual_velocity*e.physics_dt).clamp(-limit,limit)
            applied=(next_offset-self.curl_residual_offset)/e.physics_dt
            self.curl_residual_offset.copy_(next_offset)
            velocity=self.motion_velocity-self.curl_residual_velocity+applied
        if self.online_planner is not None:
            residual=self.curl_residual_velocity*(f.phase==P['CURL'])[:,None]
            clock=self.policy[:,6]*(f.phase==P['CURL'])
            requested=self.online_planner.calculate(s,q,residual,clock,self.motion_velocity)
            self.motion_velocity.copy_(torch.where(motion[:,None],requested,0.))
            velocity=self.motion_velocity
        integrated=(self.position_target+velocity*e.physics_dt).clamp(min=q-.25,max=q+.25)
        self.position_target.copy_(torch.where(motion[:,None],integrated,self.position_target))
        target=self.position_target+(kd/kp.clamp_min(1.))[:,None]*velocity
        target=torch.where(handoff[:,None],q,target)
        hold_loading=loading&(f.phase==P['LOAD_HEADTAIL'])
        torque_loading=loading&~hold_loading
        residual_scale=0. if getattr(e.cfg,'full_policy_curl_only',False) else float(getattr(e.cfg,'full_torque_scale',2.))
        self.policy_torque.copy_(torch.where(loading[:,None],residual_scale*self.policy[:,:6],0.))
        load_torque=self.loading.torque+self.policy_torque
        target=torch.where(hold_loading[:,None],self.loading_qhold+load_torque/kp.clamp_min(1.)[:,None],target)
        if getattr(e.cfg,'full_teacher_headtail_replay',False):
            target=torch.where(hold_loading[:,None],self.teacher_headtail_ctrl+self.policy_torque/kp.clamp_min(1.)[:,None],target)
        demand=kp[:,None]*(target-q)-kd[:,None]*v
        demand=torch.where(torque_loading[:,None],load_torque-1.5*v,demand)
        self.requested_torque.copy_(demand);self.check_torque_demand()
        stopped=self.torque_failure|~active
        gain_parameters=e.sim.model.actuator_gainprm
        bias_parameters=e.sim.model.actuator_biasprm
        if getattr(e.cfg,'full_cuda_graphs',False):
            gain_parameters=gain_parameters._tensor;bias_parameters=bias_parameters._tensor
        for i,m in enumerate(self.modules):
            aid=self.native_ids[i];gain=torch.where(torque_loading,1.,torch.where(stopped,0.,kp))
            gain_parameters[:,aid,0]=gain
            bias_parameters[:,aid,1]=torch.where(torque_loading|stopped,0.,-kp)
            bias_parameters[:,aid,2]=torch.where(torque_loading, -1.5,torch.where(stopped,0.,-kd))
            ctrl=torch.where(torque_loading,load_torque[:,i],torch.where(stopped,0.,target[:,i]))
            m.set_joint_position_target(ctrl[:,None]);m.set_joint_velocity_target(torch.zeros_like(ctrl[:,None]));m.set_joint_effort_target(torch.zeros_like(ctrl[:,None]))
