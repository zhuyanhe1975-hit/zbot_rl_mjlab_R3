"""Full CPU FSM with a bounded policy position residual during motion."""
import numpy as np
from .full_cpu import FullCpuReconfiguration, FullCpuConfig

PHASES=('INITIAL_STRAIGHT_HOLD','CURL','CAPTURE_HEADTAIL','SETTLE_HEADTAIL','LOAD',
        'SEPARATE','ROTATE','APPROACH','SETTLE','LOAD_HEADTAIL','UNFOLD','FINAL_STRAIGHT_HOLD','DONE','FAILED')
MOVING=('CURL','SEPARATE','ROTATE','APPROACH','UNFOLD')
CURVE_PHASES=('CURL','SEPARATE','ROTATE','APPROACH')

class TimedStudentFSM(FullCpuReconfiguration):
    def __init__(self):
        self.motion_offset=0.;self.motion_speed=1.
        super().__init__(FullCpuConfig(other_stop_load=80.))
    @property
    def phase_time(self):
        return self.data.time-self.phase_start+(self.motion_offset if self.phase in CURVE_PHASES else 0.)
    def transition(self,phase):
        actual=self.data.time-self.phase_start
        super().transition(phase)
        self.events[-1]['elapsed_s']=float(actual)
        self.motion_offset=0.
    def step(self):
        if self.phase in CURVE_PHASES:self.motion_offset+=self.cfg.dt*(self.motion_speed-1.)
        super().step()

class FullCpuStudent:
    def __init__(self, residual_scale=.01, reference=None):
        self.fsm=TimedStudentFSM();self.scale=float(residual_scale);self.action=np.zeros(7)
        self.reference=None if reference is None else dict(np.load(reference,allow_pickle=False))
        if self.reference is not None and tuple(self.reference['phase_names']) != PHASES:
            raise ValueError("teacher phase vocabulary mismatch")
        self._original_apply=self.fsm.apply
        self.fsm.apply=self._apply
    def _apply(self):
        f=self.fsm;self._original_apply()
        if f.phase not in MOVING or f.loading_mode:return
        speed_comp=(f.kd/f.kp)*(f.motion_speed-1.)*f.vref if f.phase in CURVE_PHASES else 0.
        target=f.data.ctrl[f.actuators]+self.scale*self.action[:6]+speed_comp
        demand=f.kp*(target-f.data.qpos[f.qadr])-f.kd*f.data.qvel[f.dofs]
        if not np.isfinite(demand).all() or abs(demand).max()>f.cfg.peak_torque:
            f.stop('student requested torque protection');return
        f.data.ctrl[f.actuators]=target
    def reference_sample(self):
        f=self.fsm;r=self.reference
        if r is None:return None
        candidates=np.flatnonzero((r["phase"]==PHASES.index(f.phase)) & (r["target"]==f.target) & (r["stage"]==f.stage))
        if not len(candidates):return None
        return int(candidates[np.argmin(abs(r["phase_time"][candidates]-f.phase_time))])
    def observation(self):
        f=self.fsm;phase=np.zeros(len(PHASES));phase[PHASES.index(f.phase)]=1
        target=np.zeros(6);target[f.target]=1
        q=f.data.qpos[f.qadr];v=f.data.qvel[f.dofs]
        gap,angle,lin,ang=f.geometry(f.target)
        idx=self.reference_sample();r=self.reference
        ref=np.zeros(18) if idx is None else np.r_[(r["q"][idx]-q)/3.2,(r["qvel"][idx]-v)/5.,(r["qref"][idx]-q)/3.2]
        return np.concatenate((np.sin(q),np.cos(q),v/5.,(f.qref-q)/3.2,
            f.loads/80.,f.connected().astype(float),f.data.eq_active[f.new].astype(float),
            phase,target,[min(f.phase_time/10.,10.),f.stage/4.,gap/.5,angle/np.pi,lin/2.,ang/5.],
            f.data.qfrc_actuator[f.dofs]/20.,self.action,ref)).astype(np.float32)
    def step(self, action):
        f=self.fsm;command=np.clip(np.asarray(action,dtype=float),-1.,1.)
        self.action += .1*(command-self.action)
        f.motion_speed=1.+.5*self.action[6]
        before_phase=f.phase;before_stage=f.stage;start=f.data.time;peak=0.
        for _ in range(round(f.cfg.control_dt/f.cfg.dt)):
            f.step()
            peak=max(peak,float(np.max(np.delete(f.loads,f.target))))
            if f.phase in ('DONE','FAILED'):break
        dt=f.data.time-start
        penalty_safety=max(0.,(peak-60.)/20.)**2
        progress=(f.stage-before_stage)*2. + (0.5 if f.phase!=before_phase and f.phase not in ('FAILED','DONE') else 0.)
        idx=self.reference_sample()
        tracking_error=0. if idx is None else float(np.mean((self.reference["q"][idx]-f.data.qpos[f.qadr])**2))
        reward=progress-dt*(.08+.5*penalty_safety+.2*tracking_error)
        if f.phase=='DONE':reward+=10.
        if f.phase=='FAILED':reward-=10.
        return self.observation(),float(reward),f.phase in ('DONE','FAILED'),{'phase':f.phase,'peak_other_N':peak,'episode_peak_other_N':f.peak_other,'sim_time':f.data.time,'failure':f.failure,'penalty_safety_margin':penalty_safety,'teacher_tracking_mse_rad2':tracking_error}
    def close(self):self.fsm.close()
