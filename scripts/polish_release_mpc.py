"""1 ms receding-horizon release correction using mjbatch native rollouts."""
import math
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from scripts.optimize_headtail_mjbatch import STATE, lib_open


class ReleaseMPC:
    def __init__(self, f, library, candidates=1024, horizon_ms=20, threads=4, seed=2026, action_span_rad=.4, replan_on_clear=False, distance_margin_m=None):
        self.f=f
        self.lib=lib_open(Path(library))
        if distance_margin_m is not None and not hasattr(self.lib,'zbatch_eval_distance'):
            raise ValueError('native rollout library does not support minimum-distance prediction')
        self.distance_margin_m=distance_margin_m
        self.candidates=int(candidates)
        self.horizon=int(horizon_ms)
        self.threads=int(threads)
        self.rng=np.random.default_rng(seed)
        self.action_span_rad=float(action_span_rad)
        self.replan_on_clear=bool(replan_on_clear)
        self.replanned_key=None
        self.temp=tempfile.TemporaryDirectory(prefix='zbot-release-mpc-')
        self.pool=None
        self.face=None
        self.key=None
        self.stats={'calls':0,'safe_candidates_total':0,'no_safe_calls':0,
                    'horizon_ms':self.horizon,'candidates':self.candidates,'update_period_s':f.cfg.dt,
                    'serial_state_difference':None,'action_span_rad':self.action_span_rad,'replans':0,
                    'last_status_counts':{},'last_before_distance_margin':None,'last_best_distance_m':None,
                    'distance_margin_m':distance_margin_m,'min_selected_distance_m':None}

    def reset_pool(self):
        f=self.f
        if self.pool:
            self.lib.zbatch_free(self.pool)
        self.face=int(f.target)
        self.key=(self.face, float(f.phase_start))
        path=Path(self.temp.name)/f'face_{self.face}_{f.step_count}.mjb'
        mujoco.mj_saveModel(f.model,str(path),None)
        self.pool=self.lib.zbatch_create(os.fsencode(path),self.threads)
        if not self.pool:
            raise RuntimeError('cannot create release MPC native batch')
        self.model=mujoco.MjModel.from_binary_path(str(path))
        self.data=mujoco.MjData(self.model)
        eq=int(f.old[self.face]);self.src=int(f.model.eq_obj1id[eq]);self.dst=int(f.model.eq_obj2id[eq])
        self.qadr=f.qadr.copy()

    def select(self, nominal):
        f=self.f
        phase=f.phase
        key=(int(f.target),float(f.phase_start))
        if phase == 'SEPARATE' and self.replan_on_clear and not f.cfg.overlap_rotation and self.replanned_key != key:
            gap, _, linear_speed, angular_speed = f.geometry()
            ga,gb=map(int,f.pair_geoms[f.target])
            clearance=float(mujoco.mj_geomDistance(f.model,f.data,ga,gb,.5,None))
            if (gap >= f.cfg.separation_gap and clearance >= .001
                    and linear_speed < .1 and angular_speed < .6):
                old_gap=f.planner.gap
                try:
                    # Rebuild the rotation and approach from the actual open
                    # state; no additional fictitious separation is inserted.
                    f.planner.gap=0.
                    plan=f.planner.plan(f.shim(),f.target)
                    f.curves['ROTATE']=f.curve_from(plan.points['ROTATE'])
                    f.curves['APPROACH']=f.curve_from(plan.points['APPROACH'])
                finally:
                    f.planner.gap=old_gap
                f.curve.duration=f.phase_time
                self.replanned_key=key
                self.stats['replans']+=1
                return nominal
        if self.key != (int(f.target), float(f.phase_start)):
            self.reset_pool()
        n=self.candidates;h=self.horizon
        state=np.empty(mujoco.mj_stateSize(f.model,STATE));mujoco.mj_getState(f.model,f.data,state,STATE)
        q=f.data.qpos[f.qadr].copy(); v=f.data.qvel[f.dofs].copy()
        alternatives=np.empty((n,6))
        alternatives[0]=nominal
        if n>1: alternatives[1]=q
        if n>2: alternatives[2:]=q+self.rng.uniform(-self.action_span_rad,self.action_span_rad,(n-2,6))
        low=q+(f.kd*v-f.cfg.peak_torque)/f.kp
        high=q+(f.kd*v+f.cfg.peak_torque)/f.kp
        alternatives=np.clip(alternatives,low+1e-5,high-1e-5)
        controls=np.ascontiguousarray(np.repeat(alternatives[:,None,:],h,axis=1))
        lengths=np.full(n,h,dtype=np.int32)
        result=np.zeros((n,13));ends=np.zeros((n,len(state)));metrics=np.zeros((n,4))
        distances=np.full(n,.05)
        if self.distance_margin_m is None:
            self.lib.zbatch_eval(self.pool,state,n,h,lengths,controls,result,ends,80.,metrics,0)
        else:
            self.lib.zbatch_eval_distance(self.pool,state,n,h,lengths,controls,result,ends,80.,metrics,0,distances)
        safe=(result[:,0]==0)
        status,count=np.unique(result[:,0].astype(int),return_counts=True)
        self.stats['last_status_counts']={str(int(k)):int(v) for k,v in zip(status,count)}
        self.stats['last_before_distance_margin']=int(np.count_nonzero(safe))
        self.stats['last_best_distance_m']=(float(np.max(distances[safe])) if np.any(safe) and self.distance_margin_m is not None else None)
        if self.distance_margin_m is not None:
            safe &= distances >= self.distance_margin_m
        # Native short rollouts start their local counters at zero. If an
        # already-running overload could reach its limit within the horizon,
        # conservatively reject any candidate whose peak torque exceeds the
        # continuous rating, even if its peak is on another joint.
        overload_limit=math.ceil(f.cfg.overload_seconds/f.cfg.dt)
        if np.any(f.overload + h >= overload_limit):
            safe &= result[:,2] <= f.cfg.continuous_torque
        ids=np.flatnonzero(safe)
        self.stats['calls']+=1;self.stats['safe_candidates_total']+=len(ids)
        if not len(ids):
            self.stats['no_safe_calls']+=1
            return None
        best=None;best_score=float('inf')
        for i in ids:
            mujoco.mj_setState(self.model,self.data,ends[i],STATE)
            mujoco.mj_forward(self.model,self.data)
            rot=self.data.site_xmat[self.dst].reshape(3,3)
            pos=rot.T@(self.data.site_xpos[self.src]-self.data.site_xpos[self.dst])
            # The old outlet should move towards negative local z while
            # avoiding transverse displacement and unnecessary actuation.
            if phase != 'SEPARATE':
                if f.curve is not None and f.phase_time < f.curve.duration:
                    index=min(int((f.phase_time+self.horizon*f.cfg.dt)/f.cfg.dt),len(f.curve.q)-1)
                    target=f.curve.q[index]
                else:
                    target=f.qref
                score=float(np.linalg.norm(self.data.qpos[self.qadr]-target)
                            +.002*np.linalg.norm(self.data.qvel[f.dofs])
                            +.002*np.linalg.norm(alternatives[i]-nominal))
            elif self.replan_on_clear:
                v0=np.zeros(6);v1=np.zeros(6)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_SITE,self.src,v0,0)
                mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_SITE,self.dst,v1,0)
                relative_velocity=v1-v0
                score=float(abs(pos[2]+f.cfg.separation_gap+.002)+.5*np.linalg.norm(pos[:2])
                            +.01*np.linalg.norm(relative_velocity[3:])+.003*np.linalg.norm(relative_velocity[:3])
                            +.0003*np.linalg.norm(alternatives[i]-nominal))
            else:
                score=float(pos[2]+.5*np.linalg.norm(pos[:2])+.0003*np.linalg.norm(alternatives[i]-nominal))
            if self.distance_margin_m is not None:
                score += .00001 / (distances[i] + .001)
            if score<best_score:
                best_score=score;best=int(i)
        if self.stats['serial_state_difference'] is None:
            mujoco.mj_setState(self.model,self.data,state,STATE)
            mujoco.mj_forward(self.model,self.data)
            self.data.ctrl[f.actuators]=alternatives[best]
            mujoco.mj_step(self.model,self.data,nstep=h)
            serial=np.empty_like(state)
            mujoco.mj_getState(self.model,self.data,serial,STATE)
            diff=float(np.max(np.abs(serial-ends[best])))
            self.stats['serial_state_difference']=diff
            if diff>1e-8:
                raise RuntimeError(f'release MPC batch/serial state mismatch: {diff}')
        if self.distance_margin_m is not None:
            previous=self.stats['min_selected_distance_m']
            self.stats['min_selected_distance_m']=float(distances[best]) if previous is None else min(previous,float(distances[best]))
        return alternatives[best].copy()

    def close(self):
        if self.pool:
            self.lib.zbatch_free(self.pool);self.pool=None
        self.temp.cleanup()


def enable_release_mpc(f, library, candidates=1024, horizon_ms=20, threads=4, seed=2026, action_span_rad=.4, replan_on_clear=False, modes=None, distance_margin_m=None):
    if candidates < 2 or horizon_ms < 1 or threads < 1 or not np.isfinite(action_span_rad) or action_span_rad <= 0 or (distance_margin_m is not None and (not np.isfinite(distance_margin_m) or distance_margin_m < 0 or distance_margin_m >= .05)):
        raise ValueError('invalid release MPC parameters')
    planner=ReleaseMPC(f,library,candidates,horizon_ms,threads,seed,action_span_rad,replan_on_clear,distance_margin_m)
    phases=set(modes or ['SEPARATE'])
    if not phases <= {'SEPARATE','ROTATE','APPROACH'}:
        raise ValueError('unsupported release MPC phase')
    old_advance=f.advance
    def advance():
        if (replan_on_clear and f.phase == 'SEPARATE' and not f.cfg.overlap_rotation
                and planner.replanned_key != (int(f.target),float(f.phase_start))):
            return
        old_advance()
    f.advance=advance
    old_apply=f.apply
    def apply():
        old_apply()
        if f.phase not in phases or f.loading_mode:
            return
        # Holding the measured configuration is the neutral proposal; the
        # old open-loop separation curve may initially move into the face.
        nominal=(f.data.qpos[f.qadr].copy() if f.phase=='SEPARATE'
                 else f.data.ctrl[f.actuators].copy())
        try:
            selected=planner.select(nominal)
        except Exception as exc:
            f.stop(f'release MPC error: {type(exc).__name__}: {exc}')
            return
        if selected is None:
            f.stop('release MPC: no safe predicted candidate')
        else:
            f.data.ctrl[f.actuators]=selected
    f.apply=apply
    old_close=f.close
    def close():
        try:planner.close()
        finally:old_close()
    f.close=close
    f.release_mpc_stats=planner.stats
