"""Deterministic six-module reconstruction guided by the seven-module study."""
import math
from collections import deque
from types import SimpleNamespace
from dataclasses import dataclass,fields
from enum import IntEnum
import mujoco
import numpy as np
import torch
from scipy.spatial import ConvexHull,QhullError
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.lab_api.math import quat_mul,quat_conjugate,quat_from_matrix

from .disconnect.env_cfg import TorqueAction,TorqueActionCfg
from .disconnect.state import state,failure
from .mixed_closed_dynamic import mixed_dynamic_env_cfg
from .sequential_rz270 import SequentialRz270EnvCfg
from .batched_state import BatchedState,add_batched_connectors
from .batched_controller import BatchedController
from .batched_loading import BatchedLoading
from .batched_collision import install_model_contact_filter
from .reference_trajectory import AxialPlanner,TimedCurve,clearance_lower_bound
from .state_machine import Phase,rotvec

EXPERIMENT_ID='Zbot-Mixed-Closed-Reference-Dynamics-v3'


class ReferencePhase(IntEnum):
    LOAD=0;SEPARATE=1;ROTATE=2;APPROACH=3;SETTLE=4;DONE=5;FAILED=6;STABILIZE=7


@dataclass
class ReferenceEnvCfg(SequentialRz270EnvCfg):
    reference_dynamic:bool=True
    reference_cuda_graphs:bool=True
    reference_speed_deg_s:float=45.
    reference_acceleration_deg_s2:float=180.
    reference_settle_timeout_s:float=15.
    reference_stable_window_s:float=1.
    reference_motion_wait_s:float=4.
    reference_clearance_m:float=.001


def add_reference_connectors(spec):
    add_batched_connectors(spec)
    spec.add_pair(name='open_face_5',geomname1='module_5/b6_geom',geomname2='module_0/a1_geom',
                  condim=3,margin=0.,gap=0.,friction=(.6,.6,.1,.1,.1))


class ReferenceState(BatchedState):
    def __init__(self,env):
        super().__init__(env)
        self.cpu_data=mujoco.MjData(env.sim.mj_model)
        self.max_loads_N=torch.zeros(1,6,device=env.device)
        self.last_support={};self.hull_cache={}
        self.substep_trace=deque(maxlen=math.ceil(3./env.physics_dt))
        self.port_ids=torch.tensor([[env.sim.mj_model.site(f'module_{i}/port_{side}').id
            for side in ('out','in')] for i in range(6)],device=env.device)
        self.port_body=torch.as_tensor(env.sim.mj_model.site_bodyid.copy(),device=env.device)[self.port_ids]
        self.port_roots=torch.tensor([m.data.indexing.root_body_id for m in self.modules],device=env.device)
        self.measurement_graph=None;self.measurement_ready=False
        self.port_sides=torch.tensor([0,1],device=env.device)

    def reset(self,ids):
        super().reset(ids);self.max_loads_N[slice(None) if ids is None else ids]=0.;self.last_support={}
        self.substep_trace.clear()

    def update(self):
        if self.last_substep==self.env._sim_step_counter:return
        if self.cfg.reference_cuda_graphs and self.measurement_ready:
            if self.measurement_graph is None:
                # All operations below read fixed sim buffers and are GPU-only.
                self.last_substep=None
                self.measurement_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.measurement_graph):
                    BatchedState.update(self)
                    self.max_loads_N.copy_(torch.maximum(self.max_loads_N,self.utilization*self.cfg.break_force))
            self.measurement_graph.replay();self.last_substep=self.env._sim_step_counter
        else:
            super().update()
            self.max_loads_N.copy_(torch.maximum(self.max_loads_N,self.utilization*self.cfg.break_force))
            self.measurement_ready=True
        # Retain the final three seconds including the weld-deactivation sample.
        a=self.env.action_manager.get_term('joint_torque');c=a.distance_controller
        if c.phase_host!=ReferencePhase.FAILED:
            values=torch.cat((self.utilization[0]*self.cfg.break_force,
                self.connected()[0].float(),self.force[0].flatten(),self.moment[0].flatten(),
                torch.cat([m.data.joint_vel for m in self.modules],-1)[0],
                self.env.sim.data.qvel[0],a.torque[0],a.requested_torque[0]))
            self.substep_trace.append((self.env._sim_step_counter,int(c.phase_host),values.clone()))

    def outgoing_port_frames(self):
        d=self.env.sim.data;ids=self.port_ids[:,0]
        return d.site_xpos[:,ids],quat_from_matrix(d.site_xmat[:,ids])

    def geometry(self,index=None):
        # Gather just the two selected ports, rather than converting every site
        # of all six modules eight times per query.
        d=self.env.sim.data;t=self.target_indices() if index is None else torch.full_like(self.index,index)
        other=(t+1)%6;mods=torch.stack((t,other),-1);sides=self.port_sides
        ids=self.port_ids[mods,sides];bodies=self.port_body[mods,sides]
        pos=d.site_xpos[self.worlds[:,None],ids];quat=quat_from_matrix(d.site_xmat[self.worlds[:,None],ids])
        cvel=d.cvel[self.worlds[:,None],bodies]
        com=d.subtree_com[self.worlds[:,None],self.port_roots[mods]]
        velocity=cvel[...,3:6]-torch.cross(cvel[...,:3],com-pos,dim=-1)
        p0,p1=pos[:,0],pos[:,1];q0,q1=quat[:,0],quat[:,1]
        relative=quat_mul(quat_conjugate(q0),q1);relative=torch.where(relative[:,:1]<0,-relative,relative)
        delta=p1-p0
        return {'delta':delta,'distance':delta.norm(dim=-1),'q0':q0,'q1':q1,'relative_quat':relative,
                'angle':2*torch.atan2(relative[:,1:].norm(dim=-1),relative[:,0].abs()),
                'dv':velocity[:,1]-velocity[:,0],'dw':cvel[:,1,:3]-cvel[:,0,:3]}

    def clearance(self):
        m=self.env.sim.mj_model;d=self.cpu_data;i=int(self.target_indices()[0])
        d.qpos[:]=self.env.sim.data.qpos[0].cpu().numpy();mujoco.mj_kinematics(m,d)
        a=m.geom(f'module_{i}/b{i+1}_geom').id;b=m.geom(f'module_{(i+1)%6}/a{(i+1)%6+1}_geom').id
        distance=float(mujoco.mj_geomDistance(m,d,a,b,.5,None))
        return max(distance,clearance_lower_bound(m,d,a,b,self.hull_cache)) if distance<.001 else distance

    def support(self):
        d=self.env.sim.data;c=d.contact
        valid=(torch.arange(c.dist.numel(),device=self.env.device)<d.nacon[0])&(c.worldid==0)&((c.type.int()&1)!=0)
        geom=c.geom.long().clamp(0,len(self.geom_module)-1);modules=self.geom_module[geom]
        ground=valid&((modules[:,0]>=0)^(modules[:,1]>=0))
        addresses=c.efc_address.long();available=(addresses>=0)&(addresses<d.nefc[0])
        force=torch.where(available,d.efc.force[0,addresses.clamp(0,d.efc.force.shape[1]-1)],0.)
        normal=force.clamp_min(0.).sum(-1)
        carrying=ground&(normal>.05)
        positions=c.pos[carrying,:2].cpu().numpy()
        module=torch.where(modules[:,0]>=0,modules[:,0],modules[:,1])[carrying]
        com=d.subtree_com[0,0].cpu().numpy()
        margin=None
        if len(positions)>=3:
            try:
                hull=ConvexHull(positions);margin=float(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2]).max())
            except QhullError:pass
        result={'ground_contacts':len(positions),'support_modules':len(module.unique()),
                'normal_load_N':float(torch.where(ground,normal,0.).sum()),'support_margin_m':margin}
        self.last_support=result
        return result


class ReferenceController(BatchedController):
    def __init__(self,env,modules):
        super().__init__(env,modules)
        self.phase_durations=torch.zeros(1,len(ReferencePhase),device=env.device)
        self.planner=AxialPlanner(env.cfg.separation_distance+.001)
        self.curves={};self.curve=None;self.stable_time=0.;self.history=[];self.events=[]
        self.failure_message='';self.plan_diagnostics=[];self.reference_done=False
        self.phase_host=ReferencePhase.STABILIZE;self.phase_start_step=env._sim_step_counter
        self.reference_q=torch.cat([m.data.joint_pos for m in modules],-1).clone()
        self.reference_velocity=torch.zeros_like(self.reference_q)

    def reset(self,ids):
        super().reset(ids)
        self.phase[ids]=ReferencePhase.STABILIZE
        self.curves={};self.curve=None;self.stable_time=0.;self.history=[];self.events=[]
        self.failure_message='';self.plan_diagnostics=[];self.reference_done=False
        self.phase_host=ReferencePhase.STABILIZE;self.phase_start_step=self.env._sim_step_counter
        self.reference_q.copy_(torch.cat([m.data.default_joint_pos for m in self.modules],-1));self.reference_velocity.zero_()

    def transition_to(self,phase):
        old=int(self.phase[0]);duration=float(self.phase_time[0])
        self.phase_host=ReferencePhase(phase);self.phase_start_step=self.env._sim_step_counter
        self.phase_durations[0,old]=duration;self.phase.fill_(phase);self.phase_time.zero_();self.capture_time.zero_()
        if phase in (ReferencePhase.SEPARATE,ReferencePhase.ROTATE,ReferencePhase.APPROACH):
            self.curve=self.curves[ReferencePhase(phase).name]
        self.events.append({'time_s':self.env._sim_step_counter*self.env.physics_dt,'connection':int(state(self.env).target_indices()[0])+1,
                            'from':ReferencePhase(old).name,'to':ReferencePhase(phase).name,'elapsed_s':duration})

    def stop(self,code,message):
        self.failure_code.fill_(code);self.failure_message=message
        if self.phase.item()!=ReferencePhase.FAILED:self.transition_to(ReferencePhase.FAILED)
        state(self.env).waiting.fill_(True);self.command.zero_();self.reference_velocity.zero_()

    def prepare(self,s):
        cut=int(s.target_indices()[0]);plan=self.planner.plan(s,cut)
        curves={}
        for name,points in plan.points.items():
            curve=TimedCurve(points,math.radians(self.env.cfg.reference_speed_deg_s),
                             math.radians(self.env.cfg.reference_acceleration_deg_s2),self.env.physics_dt)
            # Screen spline samples every five physics steps (not a continuous proof).
            k=self.planner.kin;anchor=k.roots[(cut+1)%6].copy();saved_q=k.q.copy();saved_roots=k.roots.copy()
            for q in curve.q[::5]:
                k.q=q;k.roots,_,_=k.fk(q,cut,anchor)
                penetration,pair,clearance=self.planner.screen(cut,name)
                if penetration>1e-5 or (name=='ROTATE' and clearance<self.env.cfg.reference_clearance_m):
                    raise RuntimeError(f'spline collision/clearance: {name}, {pair}, penetration={penetration}, clearance={clearance}')
            k.q=saved_q;k.roots=saved_roots
            curves[name]={'q':torch.tensor(curve.q,device=self.env.device,dtype=torch.float32),
                          'v':torch.tensor(curve.v,device=self.env.device,dtype=torch.float32),'duration':curve.duration}
            plan.diagnostics.append({'phase':name,'duration_s':curve.duration,'peak_speed_deg_s':math.degrees(getattr(curve,'peak_speed',0.)),
                                     'peak_acceleration_deg_s2':math.degrees(getattr(curve,'peak_acceleration',0.))})
        self.curves=curves;self.plan_diagnostics.append({'connection':cut+1,'segments':plan.diagnostics})

    def update(self,s,enabled,control_dt=None):
        e=self.env;cfg=e.cfg;a=e.action_manager.get_term('joint_torque');dt=e.physics_dt if control_dt is None else control_dt
        if control_dt is None:self.phase_time.add_(dt)
        self.command.zero_();phase=ReferencePhase(self.phase.item())
        active=e.sim.data.eq_active
        invariant=bool(active[0,s.eq_ids[2]]) and bool(active[0,s.eq_ids[5]]) and not bool(active[0,s.variant_ids[2]])
        if phase in (ReferencePhase.DONE,ReferencePhase.FAILED):return self.command
        if bool(failure(e).any()) or bool(s.contact_violation.any()) or not invariant:
            reasons=[]
            if bool(s.safety_margin_violation.any()):
                reasons.append(f'other-interface load protection: peak={float(s.max_other_load[0]):.3f}N')
            if bool(s.wrong_break.any()):reasons.append('non-target weld deactivated')
            if bool(a.torque_failure.any()):reasons.append('motor torque protection')
            if bool(s.contact_violation.any()):reasons.append('contact penetration protection')
            if not invariant:reasons.append('fixed connection invariant violated')
            self.stop(4 if s.contact_violation.any() else 1,'; '.join(reasons) or 'invalid physical state');return self.command
        q=torch.cat([m.data.joint_pos for m in self.modules],-1)
        g=s.geometry();angular=rotvec(quat_mul(self.goal_q,quat_conjugate(quat_mul(quat_conjugate(g['q1']),g['q0']))))
        self.angle_error.copy_(angular.norm(dim=-1))
        if phase==ReferencePhase.STABILIZE:
            if not s.connected().all():self.stop(1,'connection broke before loading');return self.command
            if e._sim_step_counter%cfg.decimation==0:
                support=s.support()
                linear=max(float(m.data.root_link_lin_vel_w.norm()) for m in self.modules)
                angular_speed=max(float(m.data.root_link_ang_vel_w.norm()) for m in self.modules)
                hinge=max(float(m.data.joint_vel.abs().max()) for m in self.modules)
                weight=e.sim.mj_model.body_mass.sum()*9.81
                stable=(linear<.015 and angular_speed<.1 and hinge<.08 and support['ground_contacts']>=3 and support['support_modules']>=2
                        and support['support_margin_m'] is not None and support['support_margin_m']>=.001 and .8*weight<support['normal_load_N']<1.2*weight)
                self.stable_time=self.stable_time+e.step_dt if stable else 0.
                if self.stable_time>=cfg.reference_stable_window_s:
                    self.transition_to(ReferencePhase.LOAD);a.pd_active.zero_();a.loading.reset(slice(None));a.jacobian_torque.zero_()
            if float(self.phase_time[0])>cfg.reference_settle_timeout_s:self.stop(2,'initial ground stabilization timeout')
            return self.command
        if phase==ReferencePhase.LOAD:
            if not bool(s.connected()[0,s.target_indices()[0]]):
                try:self.prepare(s)
                except Exception as error:self.stop(8,f'planning failed: {error}');return self.command
                self.transition_to(ReferencePhase.SEPARATE)
            elif float(self.phase_time[0])>cfg.phase_timeout_s:self.stop(2,'safe Jacobian loading timeout')
        phase=ReferencePhase(self.phase.item())
        if phase in (ReferencePhase.SEPARATE,ReferencePhase.ROTATE,ReferencePhase.APPROACH):
            t=float(self.phase_time[0]);curve=self.curve;index=min(max(0,round(t/e.physics_dt)-1),len(curve['q'])-1)
            self.reference_q.copy_(curve['q'][index][None]);self.reference_done=t>=curve['duration']
            self.reference_velocity.copy_(torch.zeros_like(self.reference_velocity) if self.reference_done else curve['v'][index][None])
            actual_gap=float(g['distance'][0]);angle=float(self.angle_error[0]);clearance=s.clearance() if self.reference_done and phase!=ReferencePhase.APPROACH else None
            if self.reference_done:
                if phase==ReferencePhase.SEPARATE and actual_gap>=cfg.separation_distance and clearance>=cfg.reference_clearance_m:
                    self.transition_to(ReferencePhase.ROTATE)
                elif phase==ReferencePhase.ROTATE and clearance>=cfg.reference_clearance_m:
                    self.transition_to(ReferencePhase.APPROACH)
                elif phase==ReferencePhase.APPROACH:
                    eligible=actual_gap<cfg.capture_distance and angle<cfg.capture_angle and float(g['dv'].norm())<cfg.capture_speed and float(g['dw'].norm())<cfg.capture_angular_speed
                    self.capture_time.add_(dt) if eligible else self.capture_time.zero_()
                    if self.capture_time[0]>=cfg.capture_hold_s:
                        e.sim.data.eq_active[0,s.new_eq[0]]=True;self.transition_to(ReferencePhase.SETTLE)
                if t>curve['duration']+cfg.reference_motion_wait_s:
                    self.stop(7 if phase==ReferencePhase.SEPARATE else 3,f'{phase.name} tracking/clearance/capture timeout: gap={actual_gap}, angle={angle}, clearance={clearance}')
        elif phase==ReferencePhase.SETTLE:
            eq=int(s.new_eq[0]);t=float(self.phase_time[0])
            e.sim.model.eq_solref[0,eq,0]=.03+(.005-.03)*min(t/cfg.weld_ramp_s,1.)
            if not bool(s.connected()[0,s.target_indices()[0]]):self.stop(1,'rebuilt connection broke');return self.command
            if t>=cfg.settle_s+cfg.weld_ramp_s:
                self.phase_durations[0,ReferencePhase.SETTLE]=self.phase_time[0]
                self.history.append({'connection':int(s.target_indices()[0])+1,'durations_s':self.phase_durations[0].tolist(),
                                     'joint_angle_rad':q[0].tolist(),'peak_other_load_N':s.max_other_load.item()})
                self.index.add_(1)
                if int(self.index[0])==4:self.transition_to(ReferencePhase.DONE)
                else:
                    self.transition_to(ReferencePhase.LOAD);a.pd_active.zero_();a.loading.reset(slice(None));a.jacobian_torque.zero_()
                    a.jacobian_qref.copy_(q);self.phase_durations.zero_()
        # Position reference plus velocity feedforward enters the native MuJoCo
        # position actuator; no explicit torque PD loop is applied here.
        hold=self.phase.item() not in (ReferencePhase.LOAD,ReferencePhase.FAILED)
        if hold:a.position_target.copy_(self.reference_q+cfg.separation_damping/cfg.separation_stiffness*self.reference_velocity)
        s.waiting.copy_(self.phase==ReferencePhase.FAILED)
        return self.command


@dataclass(kw_only=True)
class ReferenceActionCfg(TorqueActionCfg):
    def build(self,env):
        if env.num_envs!=1:raise ValueError('Reference dynamics FSM is a single-robot diagnostic')
        return ReferenceAction(self,env)


class ReferenceAction(TorqueAction):
    def __init__(self,cfg,env):
        feedforward=env.cfg.jacobian_feedforward;distance=env.cfg.separation_distance_jacobian
        env.cfg.jacobian_feedforward=False;env.cfg.separation_distance_jacobian=False
        try:super().__init__(cfg,env)
        finally:env.cfg.jacobian_feedforward=feedforward;env.cfg.separation_distance_jacobian=distance
        self.motion_stiffness=env.cfg.separation_stiffness;self.motion_damping=env.cfg.separation_damping
        self.phase_contacts=torch.zeros(1,6,device=env.device,dtype=torch.bool)
        install_model_contact_filter(env,self.phase_contacts,include_headtail=True);env.sim.expand_model_fields(('eq_solref',))
        env._disconnect_state=ReferenceState(env)
        self.jacobian_torque=torch.zeros_like(self._raw);self.jacobian_qref=self.position_target.clone()
        self.loading=BatchedLoading(env);self.jacobian_report=self.loading.report
        self.distance_controller=ReferenceController(env,self.modules)
        self.actuation_graph=None;self.actuation_ready=False
        self.capture_native_parameters=env.cfg.reference_cuda_graphs

    def update_jacobian_gain(self,s):pass

    def jacobian_command(self,positions=None,active=None):
        s=state(self._env);valid=(self.distance_controller.phase==ReferencePhase.LOAD)&s.connected().all(-1)&~s.waiting&~s.safety_margin_violation&~s.wrong_break&~self.torque_failure
        return self.jacobian_torque*valid[:,None]

    def process_actions(self,actions):
        c=self.distance_controller;s=state(self._env)
        loading=(c.phase==ReferencePhase.LOAD)&s.connected().all(-1)&~s.safety_margin_violation&~self.torque_failure
        self.loading.update(s,loading);self.jacobian_torque.copy_(self.loading.torque)
        self.jacobian_qref.copy_(torch.cat([m.data.joint_pos for m in self.modules],-1))
        super().process_actions(torch.zeros_like(actions))

    def apply_actions(self):
        c=self.distance_controller;s=state(self._env)
        s.waiting.copy_(c.phase==ReferencePhase.FAILED)
        # Match the reference execution strategy: fixed native gains throughout
        # the motion, no extra end-of-phase gain ramp or position correction jump.
        self._env.cfg.separation_stiffness=self.motion_stiffness
        self._env.cfg.separation_damping=self.motion_damping
        self.pd_active|=c.phase==ReferencePhase.STABILIZE
        if not getattr(self._env.cfg,'reference_cuda_graphs',False):
            super().apply_actions()
        else:
            # Keep the CPU FSM outside the graph; native actuator writes and
            # demand protection are replayed on every 1ms physical step.
            step=self._env._sim_step_counter
            elapsed=(step-c.phase_start_step)*self._env.physics_dt
            c.phase_time.fill_(elapsed)
            if step%self._env.cfg.decimation==0:
                stopped=s.waiting|s.safety_margin_violation|s.wrong_break|self.torque_failure
                entered=(~s.connected()[s.worlds,s.target_indices()])&~self.pd_active&~stopped
                position=torch.cat([m.data.joint_pos for m in self.modules],-1)
                self.position_target.copy_(torch.where(entered[:,None],position,self.position_target));self.pd_active|=entered
                c.update(s,self.pd_active&~stopped,control_dt=self._env.step_dt)
            if c.phase_host in (ReferencePhase.SEPARATE,ReferencePhase.ROTATE,ReferencePhase.APPROACH):
                elapsed=(step-c.phase_start_step)*self._env.physics_dt;curve=c.curve
                idx=min(max(0,step-c.phase_start_step-1),len(curve['q'])-1)
                c.reference_q.copy_(curve['q'][idx][None])
                if elapsed>=curve['duration']:c.reference_velocity.zero_()
                else:c.reference_velocity.copy_(curve['v'][idx][None])
                self.position_target.copy_(c.reference_q+self.motion_damping/self.motion_stiffness*c.reference_velocity)
            proxy=SimpleNamespace(phase=c.phase,update=lambda state,enabled:c.command)
            self.distance_controller=proxy
            try:
                if not self.actuation_ready:
                    TorqueAction.apply_actions(self);self.actuation_ready=True
                elif self.actuation_graph is None:
                    self.actuation_graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.actuation_graph):TorqueAction.apply_actions(self)
                    self.actuation_graph.replay()
                else:self.actuation_graph.replay()
            finally:self.distance_controller=c
        self.phase_contacts.zero_()
        self.phase_contacts[0,s.target_indices()[0]]=(c.phase[0]==ReferencePhase.ROTATE)

    def reset(self,env_ids=None):
        super().reset(env_ids);self.loading.reset(slice(None));self.jacobian_torque.zero_();self.phase_contacts.zero_()


def reference_env_cfg():
    base=mixed_dynamic_env_cfg();cfg=ReferenceEnvCfg(**{f.name:getattr(base,f.name) for f in fields(base)})
    cfg.experiment_id=EXPERIMENT_ID
    cfg.scene.spec_fn=add_reference_connectors;cfg.actions={'joint_torque':ReferenceActionCfg(entity_name='module_0')}
    cfg.separation_stiffness=400.;cfg.separation_damping=4.;
    # Resolve stiff weld/contact transients at 1 ms while retaining 20 ms control.
    cfg.sim.mujoco.timestep=.001;cfg.decimation=20
    cfg.episode_length_s=300.;cfg.auto_reset=False;cfg.terminations={};cfg.rewards={}
    return cfg
