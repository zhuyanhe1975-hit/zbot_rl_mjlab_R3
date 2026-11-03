"""Vectorized, asynchronous phase logic for the full six-module reconstruction.

Pure tensor state machine: geometry, load, collision, actuator and weld state
are supplied by the physics integration layer. No state or policy commands can
force a load-triggered connector release; `capture_*` requests only closure.
"""
import math
import torch
from .full_cpu_student import PHASES

P={name:i for i,name in enumerate(PHASES)}
ORDER=(0,1,3,4)


class FullBatchedFSM:
    def __init__(self,num_envs,device,dt=.02,refined=False):
        self.num_envs=num_envs;self.device=device;self.dt=float(dt);self.refined=bool(refined)
        self.phase=torch.full((num_envs,),P['INITIAL_STRAIGHT_HOLD'],device=device,dtype=torch.long)
        self.phase_time=torch.zeros(num_envs,device=device)
        # Independent simulation-time watchdog, not affected by policy clock scaling.
        self.phase_elapsed=torch.zeros(num_envs,device=device)
        self.phase_timeouts=torch.tensor([4.,30.,4.,2.,40.,30.,30.,30.,2.,40.,120.,3.,float("inf"),float("inf")],device=device)
        self.stage=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.target=torch.full((num_envs,),5,device=device,dtype=torch.long)
        self.capture_time=torch.zeros(num_envs,device=device)
        self.failure_code=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.active_load_phase=torch.zeros(num_envs,device=device,dtype=torch.bool)
        self.order=torch.tensor(ORDER,device=device,dtype=torch.long)
        self.expected_registration=torch.tensor([True,True,False,True,True],device=device)
        self.worlds=torch.arange(num_envs,device=device)

    def reset(self,ids):
        self.phase[ids]=P['INITIAL_STRAIGHT_HOLD'];self.phase_time[ids]=0.;self.phase_elapsed[ids]=0.;self.stage[ids]=0
        self.target[ids]=5;self.capture_time[ids]=0.;self.failure_code[ids]=0;self.active_load_phase[ids]=False

    def update(self,connected,registrations,loads,geometry,curve_done,straight,collision,torque_failure,dt=None,curve_timeout=None):
        """Advance independently; caller applies returned closure requests.

        `geometry` is the target-port gap, angle error and relative linear and
        angular speeds. `curve_done` is a per-world teacher trajectory completion
        predicate; `straight` includes terminal angle, speed and gap checks.
        Physics must supply the *current measured* connection flags and loads.
        """
        n=self.num_envs
        if connected.shape!=(n,6) or registrations.shape!=(n,5) or loads.shape!=(n,6):
            raise ValueError('full-FSM world tensor shape mismatch')
        old=self.phase.clone();self.phase_time.add_(self.dt if dt is None else dt);self.phase_elapsed.add_(self.dt)
        active=(old!=P['DONE'])&(old!=P['FAILED'])
        phase_load=(old==P['LOAD'])|(old==P['LOAD_HEADTAIL'])
        self.active_load_phase.copy_(phase_load&active)
        target=self.target;actual_target=connected[self.worlds,target]
        other=torch.arange(6,device=self.device)[None,:]!=target[:,None]
        expected=connected.clone()
        # The head-tail connection is intentionally absent before initial
        # capture and after its passive release, never a lost internal weld.
        head_open=(old==P['INITIAL_STRAIGHT_HOLD'])|(old==P['CURL'])|(old==P['CAPTURE_HEADTAIL'])|(old==P['UNFOLD'])|(old==P['FINAL_STRAIGHT_HOLD'])
        if self.refined:head_open|=(old==P['SETTLE_HEADTAIL'])&(self.stage==4)
        expected[:,5]|=head_open
        # The currently selected internal weld is intentionally absent during
        # its separated/rotation/approach phases.
        open_target=phase_load|(old==P['SEPARATE'])|(old==P['ROTATE'])|(old==P['APPROACH'])
        expected[self.worlds,target]|=open_target
        missing=~expected.all(-1)
        other_over=((loads>=80.)&other&connected).any(-1)
        invalid=(~torch.isfinite(loads).all(-1))
        bad=active&(collision|torque_failure|missing|other_over|invalid)
        self.failure_code.copy_(torch.where(bad,torch.where(collision,3,torch.where(torque_failure,4,torch.where(missing,2,1))),self.failure_code))
        new=old.clone();new[bad]=P['FAILED']
        ok=active&~bad
        def set_phase(mask,name):new[mask&ok]=P[name]
        gap,angle,linear,angular=(geometry[k] for k in ('gap','angle','linear_speed','angular_speed'))
        eligible=(gap<(.001 if self.refined else .002))&(angle<math.radians(2.))&(linear<(.1 if self.refined else .03))&(angular<(.6 if self.refined else .3))
        captures=(old==P['CAPTURE_HEADTAIL'])|(old==P['APPROACH'])
        self.capture_time.copy_(torch.where(captures&eligible,self.capture_time+(self.dt if dt is None else dt),0.))
        close_head=ok&(old==P['CAPTURE_HEADTAIL'])&(self.capture_time>=.03)
        close_internal=ok&(old==P['APPROACH'])&(self.capture_time>=.03)
        settle=.25 if self.refined else 1.
        set_phase((old==P['INITIAL_STRAIGHT_HOLD'])&(self.phase_time>=settle),'CURL')
        set_phase((old==P['CURL'])&curve_done,'CAPTURE_HEADTAIL')
        set_phase(close_head,'SETTLE_HEADTAIL')
        first_load=ok&(old==P['SETTLE_HEADTAIL'])&(self.phase_time>=settle)&connected[:,5]
        set_phase(first_load,'LOAD');self.stage[first_load]=0;self.target[first_load]=0
        set_phase((old==P['LOAD'])&~actual_target,'SEPARATE')
        set_phase((old==P['SEPARATE'])&curve_done&(gap>=(.02 if self.refined else .05)), 'APPROACH' if self.refined else 'ROTATE')
        set_phase((old==P['ROTATE'])&curve_done,'APPROACH')
        set_phase(close_internal,'SETTLE')
        advance=ok&(old==P['SETTLE'])&(self.phase_time>=settle)&actual_target
        next_stage=self.stage+1
        internal_next=advance&(next_stage<4)
        final_next=advance&(next_stage>=4)
        self.stage.copy_(torch.where(advance,next_stage,self.stage))
        self.target.copy_(torch.where(internal_next,self.order[next_stage.clamp(0,3)],torch.where(final_next,5,self.target)))
        set_phase(internal_next,'LOAD');set_phase(final_next,'LOAD_HEADTAIL')
        set_phase((old==P['LOAD_HEADTAIL'])&~actual_target,'SETTLE_HEADTAIL' if self.refined else 'UNFOLD')
        if self.refined:set_phase((old==P['SETTLE_HEADTAIL'])&(self.stage==4)&~actual_target&(self.phase_time>=.5),'UNFOLD')
        set_phase((old==P['UNFOLD'])&straight,'FINAL_STRAIGHT_HOLD')
        finished=(old==P['FINAL_STRAIGHT_HOLD'])&(self.phase_time>=settle)&straight&connected[:,:5].all(-1)&~connected[:,5]
        set_phase(finished&(registrations==self.expected_registration).all(-1),'DONE')
        # A phase that completes at the boundary wins over its watchdog;
        # timeout applies only if the world is still stuck in that phase.
        timeout=ok&(new==old)&(self.phase_elapsed>=self.phase_timeouts[old])
        if curve_timeout is not None:timeout|=ok&(new==old)&curve_timeout
        new[timeout]=P['FAILED'];self.failure_code[timeout]=5
        changed=new!=old;self.phase.copy_(new);self.phase_time.masked_fill_(changed,0.);self.phase_elapsed.masked_fill_(changed,0.);self.capture_time.masked_fill_(changed,0.)
        return {'capture_headtail':close_head,'capture_internal':close_internal,'failed':new==P['FAILED'],'done':new==P['DONE']}
