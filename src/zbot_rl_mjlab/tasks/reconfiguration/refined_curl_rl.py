"""First-stage RL on top of the verified refined CPU FSM.

Only a bounded joint-velocity residual during CURL is learned.  All other
phases remain in the verified FSM.  This intentionally uses one CPU world.
"""
import numpy as np


class RefinedCurlEnv:
    def __init__(self,residual_speed=.02,residual_offset=.02):
        if not (0 < residual_speed <= .1 and 0 < residual_offset <= .05):
            raise ValueError('invalid residual limits')
        self.residual_speed=float(residual_speed);self.residual_offset=float(residual_offset)
        self.f=None;self.action=np.zeros(6);self.offset=np.zeros(6);self.peak_other=0.

    def reset(self):
        # Lazy import avoids constructing the MuJoCo viewer or touching other experiments.
        import scripts.view_headtail_mjbatch_refined as source
        if self.f is not None:self.f.close()
        self.f=source.create_refined('forward');self.action[:]=0.;self.offset[:]=0.;self.peak_other=0.
        f=self.f;old_apply=f.apply
        def apply():
            old_apply()
            if f.phase!='CURL' or f.failure:return
            self.offset[:]=np.clip(self.offset+self.action*self.residual_speed*f.cfg.dt,-self.residual_offset,self.residual_offset)
            f.data.ctrl[f.actuators]+=self.offset+(f.kd/f.kp)*self.action*self.residual_speed
            q=f.data.qpos[f.qadr];dq=f.data.qvel[f.dofs]
            demand=f.kp*(f.data.ctrl[f.actuators]-q)-f.kd*dq
            if not np.isfinite(demand).all() or np.max(np.abs(demand))>f.cfg.peak_torque:
                f.stop('RL residual torque protection')
        f.apply=apply
        return self.observation()

    def observation(self):
        f=self.f;q=f.data.qpos[f.qadr];dq=f.data.qvel[f.dofs]
        phase=np.array([f.phase=='CURL',f.phase=='CAPTURE_HEADTAIL',f.phase=='SETTLE_HEADTAIL'],dtype=float)
        return np.concatenate((q/np.pi,dq/8.,f.loads[:5]/100.,f.qref/np.pi,self.offset/.05,phase)).astype(np.float32)

    def step(self,action):
        f=self.f;self.action[:]=np.clip(np.asarray(action,dtype=float),-1.,1.) if f.phase=='CURL' else 0.
        was_curl=f.phase=='CURL';peak_before=self.peak_other
        for _ in range(round(f.cfg.control_dt/f.cfg.dt)):
            f.step()
            active=f.connected().copy()
            if f.phase in ('LOAD','LOAD_HEADTAIL'):active[f.target]=False
            if active.any():self.peak_other=max(self.peak_other,float(f.loads[active].max()))
            if f.phase in ('DONE','FAILED','LOAD'):break
        success=f.phase=='LOAD' and f.connected().all()
        failed=f.phase=='FAILED';done=bool(success or failed)
        # This is a local shaping objective for this one phase, not a full-task success metric.
        reward=-.002*float(was_curl)-.02*max(self.peak_other-peak_before,0.)/80.
        if success:reward+=5.
        if failed:reward-=5.
        return self.observation(),float(reward),done,{'success':bool(success),'failure':f.failure,
               'phase':f.phase,'elapsed_s':float(f.data.time),'peak_other_N':self.peak_other,'policy_active':was_curl}

    def close(self):
        if self.f is not None:self.f.close();self.f=None
