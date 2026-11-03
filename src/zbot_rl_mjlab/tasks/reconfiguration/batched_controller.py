"""Independent asynchronous mixed-reconfiguration FSM on GPU tensors."""
import math
import torch
import numpy as np
from mjlab.utils.lab_api.math import quat_mul,quat_conjugate
from .state_machine import Phase,rotvec
from .disconnect.state import failure,state
from .kinematic_guide import planned_reference


class BatchedGuide:
    def __init__(self,env):
        refs,self.report=planned_reference(env.cfg.motion_duration_s)
        self.samples=torch.tensor(np.stack([[refs[(i,p)] for p in (1,2,3)] for i in (0,1,3,4)]),device=env.device,dtype=torch.float32)
        self.duration=env.cfg.motion_duration_s

    def reference(self,c,position):
        active=(c.phase>=Phase.SEPARATE)&(c.phase<=Phase.APPROACH)
        phase=(c.phase-1).clamp(0,2);stage=c.index.clamp_max(3)
        time=(c.phase_time+.10).clamp(0.,self.duration)
        t=time/self.duration*20;idx=t.long().clamp(0,19);f=(t-idx)[:,None]
        left=self.samples[stage,phase,idx];right=self.samples[stage,phase,idx+1]
        q=left+f*(right-left);v=(right-left)*(20/self.duration)
        q=torch.where(active[:,None],q,position)
        v=torch.where((active&(time<self.duration))[:,None],v,0.)
        return q,v


class BatchedController:
    def __init__(self,env,modules):
        self.env=env;self.modules=modules;n=env.num_envs;dev=env.device
        self.index=state(env).index
        self.phase=torch.zeros(n,device=dev,dtype=torch.long)
        self.phase_time=torch.zeros(n,device=dev)
        self.phase_durations=torch.zeros(n,len(Phase),device=dev)
        self.capture_time=torch.zeros(n,device=dev)
        self.command=torch.zeros(n,6,device=dev)
        self.predicted_speed=torch.zeros(n,device=dev)
        self.angle_error=torch.zeros(n,device=dev)
        self.position_error=torch.zeros(n,device=dev)
        self.failure_code=torch.zeros(n,device=dev,dtype=torch.long)
        self.gap_goal=torch.full((n,),.055,device=dev)

    @property
    def goal_q(self):
        sign=torch.where(state(self.env).target_indices()<2,-1.,1.)
        zero=torch.zeros_like(sign)
        return torch.stack((torch.full_like(sign,math.cos(math.pi/4)),zero,zero,sign*math.sin(math.pi/4)),-1)

    def reset(self,ids):
        self.index[ids]=0;self.phase[ids]=Phase.LOAD
        for v in (self.phase_time,self.phase_durations,self.capture_time,self.command,self.predicted_speed,self.angle_error,self.position_error,self.failure_code):v[ids]=0
        self.gap_goal[ids]=.055
        s=state(self.env)
        if isinstance(ids,slice):self.env.sim.model.eq_solref[:,s.variant_ids,0]=.03
        else:self.env.sim.model.eq_solref[ids[:,None],s.variant_ids[None],0]=.03

    def transition(self,mask,phase):
        mask=mask&(self.phase!=phase)
        if phase!=Phase.FAILED:
            timed=(self.phase>=Phase.SEPARATE)&(self.phase<=Phase.APPROACH)
            mask&=~timed|(self.phase_time<=self.env.cfg.motion_duration_s+1e-6)
        old=self.phase_durations.gather(1,self.phase[:,None]).squeeze(1)
        self.phase_durations.scatter_(1,self.phase[:,None],torch.where(mask,self.phase_time,old)[:,None])
        self.phase.copy_(torch.where(mask,phase,self.phase));self.phase_time.masked_fill_(mask,0.)

    def update(self,s,enabled):
        e=self.env;cfg=e.cfg;dt=e.physics_dt;a=e.action_manager.get_term('joint_torque')
        self.phase_time.add_(dt)
        connected=s.connected();target=s.target_indices();worlds=s.worlds
        invariant=e.sim.data.eq_active[:,s.eq_ids[2]].bool()&e.sim.data.eq_active[:,s.eq_ids[5]].bool()&~e.sim.data.eq_active[:,s.variant_ids[2]].bool()
        bad=failure(e)|s.contact_violation|~invariant
        self.failure_code.copy_(torch.where(s.contact_violation,4,torch.where(~invariant,5,torch.where(bad,1,self.failure_code))))
        self.transition(bad,Phase.FAILED)
        self.transition((self.phase==Phase.LOAD)&~connected[worlds,target],Phase.SEPARATE)
        g=s.geometry()
        relative=quat_mul(quat_conjugate(g['q1']),g['q0'])
        angular=rotvec(quat_mul(self.goal_q,quat_conjugate(relative)))
        self.angle_error.copy_(angular.norm(dim=-1))
        self.transition((self.phase==Phase.SEPARATE)&(g['distance']>.05),Phase.ROTATE)
        self.transition((self.phase==Phase.ROTATE)&(self.angle_error<math.radians(1))&(g['distance']>=.05)&(g['dw'].norm(dim=-1)<.5),Phase.APPROACH)
        approaching=self.phase==Phase.APPROACH
        self.gap_goal.copy_(torch.where(approaching&(self.angle_error<math.radians(5)),(self.gap_goal-cfg.approach_speed*dt).clamp_min(0.),self.gap_goal))
        eligible=approaching&(g['distance']<cfg.capture_distance)&(self.angle_error<cfg.capture_angle)&(g['dv'].norm(dim=-1)<cfg.capture_speed)&(g['dw'].norm(dim=-1)<cfg.capture_angular_speed)
        self.capture_time.copy_(torch.where(eligible,self.capture_time+dt,0.))
        capture=eligible&(self.capture_time>=cfg.capture_hold_s)
        eq=s.new_eq
        e.sim.data.eq_active[worlds,eq]|=capture
        position=torch.cat([m.data.joint_pos for m in self.modules],-1)
        a.position_target.copy_(torch.where(capture[:,None],position,a.position_target))
        self.transition(capture,Phase.SETTLE)
        settling=self.phase==Phase.SETTLE
        blend=(self.phase_time/cfg.weld_ramp_s).clamp(0.,1.)
        current=e.sim.model.eq_solref[worlds,eq,0]
        e.sim.model.eq_solref[worlds,eq,0]=torch.where(settling,.03+(.005-.03)*blend,current)
        lost=settling&~s.connected()[worlds,target]
        self.failure_code.copy_(torch.where(lost,1,self.failure_code));self.transition(lost,Phase.FAILED)
        advance=settling&(self.phase_time>=cfg.settle_s+cfg.weld_ramp_s)&s.connected()[worlds,target]&~bad&~lost
        final=advance&(self.index==3)
        self.transition(advance,Phase.DONE)
        self.index.add_(advance.long())
        following=advance&~final
        self.phase.copy_(torch.where(following,Phase.LOAD,self.phase))
        self.phase_time.masked_fill_(following,0.);self.phase_durations.masked_fill_(following[:,None],0.)
        self.capture_time.masked_fill_(following,0.);self.gap_goal.masked_fill_(following,.055)
        self.command.masked_fill_(following[:,None],0.);s.max_target_gap.masked_fill_(following,0.)
        a.pd_active&=~following
        a.position_target.copy_(torch.where(following[:,None],position,a.position_target))
        a.jacobian_torque.masked_fill_(following[:,None],0.)
        a.jacobian_qref.copy_(torch.where(following[:,None],position,a.jacobian_qref))
        a.loading.reset(following)
        deadline=(self.phase>=Phase.SEPARATE)&(self.phase<=Phase.APPROACH)&(self.phase_time>=cfg.motion_duration_s)
        timeout=(self.phase_time>cfg.phase_timeout_s)&(self.phase<Phase.DONE)
        self.failure_code.copy_(torch.where(deadline,3,torch.where(timeout,2,self.failure_code)))
        self.transition(deadline|timeout,Phase.FAILED)
        reference,velocity=a.guide.reference(self,position)
        a.guide_position.copy_(reference)
        error=torch.atan2(torch.sin(reference-position),torch.cos(reference-position))
        guided=(velocity+4.*error).clamp(-cfg.separation_max_target_speed,cfg.separation_max_target_speed)
        a.nominal_velocity.copy_(guided)
        moving=(self.phase>=Phase.SEPARATE)&(self.phase<=Phase.APPROACH)&enabled
        desired=(guided+a.motion_action*cfg.motion_residual_speed).clamp(-cfg.separation_max_target_speed,cfg.separation_max_target_speed)
        step=cfg.separation_joint_acceleration*dt
        self.command.add_((desired-self.command).clamp(-step,step))
        self.command.masked_fill_(~moving[:,None],0.)
        s.waiting.copy_(self.phase==Phase.FAILED)
        return self.command
