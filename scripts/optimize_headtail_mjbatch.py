"""Batch-search reverse/forward curl trajectories using mjbatch's C++ thread pool.

The sampled open-loop candidates are screened at 1 ms; the normal FSM must
subsequently validate the entire chosen trajectory, including docking and loads.
"""
import argparse
import ctypes
import json
import math
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import mujoco
import numpy as np

import scripts.view_headtail_bidirectional as bidirectional
from zbot_rl_mjlab.tasks.reconfiguration.reference_trajectory import TimedCurve

OUT=Path('outputs/reconfiguration/mjbatch_refinement')
LIB=OUT/'libcurl_rollout.so'
STATE=mujoco.mjtState.mjSTATE_INTEGRATION


def ready(direction,continuous,state_path=None,kp=None,kd=None,control_dt=None,acceleration_limit=True):
    f=bidirectional.create(direction)
    if control_dt is not None:
        f.cfg.control_dt=control_dt
    f.cfg.acceleration_limit_enabled=acceleration_limit
    if kp is not None:
        f.cfg.stiffness=f.cfg.outer_stiffness=kp
        f.cfg.damping=f.cfg.outer_damping=kd
        f.set_pd()
        mujoco.mj_forward(f.model,f.data)
    if continuous:
        with np.load(state_path or bidirectional.START) as snapshot:
            mujoco.mj_setState(f.model,f.data,snapshot['state'],STATE)
        f.phase_start=float(f.data.time);f.qref=f.data.qpos[f.qadr].copy()
        f.next_log=f.phase_start;f.next_progress=f.phase_start
        f.set_contact_phase();f.set_pd();mujoco.mj_forward(f.model,f.data)
    while f.phase=='INITIAL_STRAIGHT_HOLD':
        if f.step_count%round(f.cfg.control_dt/f.cfg.dt)==0 and f.phase_time>=f.cfg.initial_hold_seconds:
            f.advance();break
        f.step()
    if f.phase!='CURL':raise RuntimeError(f'Cannot reach curl start: {f.phase}: {f.failure}')
    return f


def controls_for(f, speed_multiplier, acceleration_multiplier):
    cfg=f.cfg
    speed=math.radians(cfg.outer_speed_deg_s*speed_multiplier)
    acceleration=math.radians(cfg.outer_acceleration_deg_s2*speed_multiplier**2*acceleration_multiplier) if getattr(cfg,'acceleration_limit_enabled',True) else None
    curve=TimedCurve(np.stack((f.data.qpos[f.qadr].copy(),f.closed_q)),speed,acceleration,cfg.dt)
    steps=math.ceil(curve.duration/cfg.control_dt-1e-10)*round(cfg.control_dt/cfg.dt)
    values=np.empty((steps,6));time_s=float(f.data.time)
    for i in range(steps):
        phase_time=time_s-f.phase_start
        index=min(max(0,int(phase_time/cfg.dt)),len(curve.q)-1)
        v=curve.v[index] if phase_time<curve.duration else np.zeros(6)
        values[i]=curve.q[index]+f.kd/f.kp*v
        time_s+=cfg.dt
    return values,curve


def lib_open(path=LIB):
    lib=ctypes.CDLL(str(path.resolve()))
    lib.zbatch_create.argtypes=[ctypes.c_char_p,ctypes.c_int];lib.zbatch_create.restype=ctypes.c_void_p
    lib.zbatch_free.argtypes=[ctypes.c_void_p]
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    lib.zbatch_eval.argtypes=[ctypes.c_void_p,dp,ctypes.c_int,ctypes.c_int,ip,dp,dp,dp,ctypes.c_double,dp,ctypes.c_int]
    lib.zbatch_eval.restype=ctypes.c_int
    if hasattr(lib,'zbatch_eval_distance'):
        lib.zbatch_eval_distance.argtypes=lib.zbatch_eval.argtypes+[dp]
        lib.zbatch_eval_distance.restype=ctypes.c_int
    return lib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direction',choices=['forward','reverse'],default='reverse')
    p.add_argument('--continuous',action='store_true',help='reverse starts at forward terminal time and state')
    p.add_argument('--state',type=Path,help='integration-state snapshot for continuous reverse start')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--count',type=int,default=33)
    p.add_argument('--low',type=float,default=1.9)
    p.add_argument('--high',type=float,default=2.7)
    p.add_argument('--accel-factor',type=float,default=1.)
    p.add_argument('--load-limit',type=float,default=80.,help='maximum non-target load in N')
    p.add_argument('--offset-rad',type=float,default=0.,help='sample smooth per-joint position offsets')
    p.add_argument('--goal-file',type=Path,help='JSON list of candidate closed joint configurations')
    p.add_argument('--goal-index',type=int,default=0)
    p.add_argument('--output-dir',type=Path,default=OUT,help='directory for model and search report')
    p.add_argument('--kp',type=float,help='position servo gain for all motion phases')
    p.add_argument('--kd',type=float,default=1.5,help='position servo damping when --kp is set')
    p.add_argument('--seed-candidate',type=Path,help='additional candidate with curl_speed_multiplier and offset_rad')
    p.add_argument('--native-library',type=Path,default=LIB,help='native rollout library')
    p.add_argument('--control-dt',type=float,help='FSM update period in seconds')
    p.add_argument('--no-acceleration-limit',action='store_true')
    args=p.parse_args()
    if args.threads<1 or args.count<2 or not (0<args.low<args.high) or args.accel_factor<=0 or not (0<args.load_limit<=100) or args.offset_rad<0 or (args.kp is not None and (args.kp<=0 or args.kd<0)):p.error('invalid search settings')
    if args.state and not args.continuous:p.error('--state requires --continuous')
    if args.control_dt is not None and (not math.isfinite(args.control_dt) or args.control_dt < .001 or abs(args.control_dt/.001-round(args.control_dt/.001)) > 1e-9):p.error('control-dt must be a positive integer number of 1ms steps')
    args.output_dir.mkdir(parents=True,exist_ok=True);f=ready(args.direction,args.continuous,args.state,args.kp,args.kd,args.control_dt,not args.no_acceleration_limit)
    try:
        if args.goal_file:
            f.closed_q=np.asarray(json.loads(args.goal_file.read_text())[args.goal_index],dtype=float)
            if f.closed_q.shape!=(6,):raise ValueError('closed goal must contain six joint angles')
        factors=np.linspace(args.low,args.high,args.count)
        # Append the exact baseline for a direct numerical check.
        factors=np.r_[factors,2.6]
        seed=None
        if args.seed_candidate:
            seed=json.loads(args.seed_candidate.read_text())
            seed_offset=np.asarray(seed['offset_rad'],dtype=float)
            seed_speed=float(seed['curl_speed_multiplier'])
            if seed_offset.shape!=(6,) or not np.isfinite(seed_offset).all() or not math.isfinite(seed_speed) or seed_speed<=0:raise ValueError('invalid seed candidate')
            factors=np.r_[seed_speed,factors]
        seq=[controls_for(f,float(x),args.accel_factor) for x in factors]
        rng=np.random.default_rng(20261029)
        offsets=rng.uniform(-args.offset_rad,args.offset_rad,(len(seq),6))
        offsets[-1]=0.
        if seed is not None:offsets[0]=seed_offset
        for i,(control,_) in enumerate(seq):
            envelope=np.sin(np.linspace(0.,np.pi,len(control)))**2
            control+=envelope[:,None]*offsets[i]
        lengths=np.asarray([len(c) for c,_ in seq],dtype=np.int32);n=len(seq);maxsteps=int(lengths.max())
        controls=np.zeros((n,maxsteps,6),dtype=np.float64)
        for i,(control,_) in enumerate(seq):controls[i,:len(control)]=control
        state=np.empty(mujoco.mj_stateSize(f.model,STATE),dtype=np.float64);mujoco.mj_getState(f.model,f.data,state,STATE)
        model_file=args.output_dir/f'{args.direction}_curl_model.mjb';mujoco.mj_saveModel(f.model,str(model_file),None)
        ns=len(state);results=np.zeros((n,13));endstates=np.zeros((n,ns));metrics=np.zeros((n,4));lib=lib_open(args.native_library)
        ptr=lib.zbatch_create(os.fsencode(model_file),args.threads)
        if not ptr:raise RuntimeError('Unable to create dense-Jacobian native batch')
        start=time.perf_counter()
        try:lib.zbatch_eval(ptr,state,n,maxsteps,lengths,controls,results,endstates,args.load_limit,metrics,0)
        finally:lib.zbatch_free(ptr)
        elapsed=time.perf_counter()-start
        # Independently run an unfailed candidate with exactly the same controls.
        valid_indices=np.flatnonzero(results[:,0]==0)
        difference=None
        if valid_indices.size:
            check=int(valid_indices[-1]);reference=mujoco.MjData(f.model)
            mujoco.mj_setState(f.model,reference,state,STATE)
            for control in seq[check][0]:
                reference.ctrl[f.actuators]=control;mujoco.mj_step(f.model,reference)
            actual=np.empty(ns);mujoco.mj_getState(f.model,reference,actual,STATE)
            difference=float(np.max(np.abs(actual-endstates[check])))
            if difference>1e-8:raise RuntimeError(f'native rollout differs from serial MuJoCo: {difference}')
        candidates=[]
        for i,(factor,(control,curve)) in enumerate(zip(factors,seq)):
            row=results[i]
            candidates.append({'curl_speed_multiplier':float(factor),'acceleration_multiplier':args.accel_factor,
                'predicted_duration_s':float(curve.duration),'sampled_time_s':len(control)*f.cfg.dt,
                'status':int(row[0]),'max_load_N':float(row[1]),'max_torque_Nm':float(row[2]),
                'max_module_penetration_m':float(row[3]),'max_ground_penetration_m':float(row[4]),
                'failure_step':int(row[5]),'failure_face':int(row[6]),'per_face_peak_N':row[7:13].tolist(),
                'offset_rad':offsets[i].tolist(),'peak_root_angular_speed_rad_s':float(metrics[i,0]),
                'peak_com_height_m':float(metrics[i,1]),'peak_base_roll_rad':float(metrics[i,2]),'min_ground_contacts':int(metrics[i,3])})
        report={'direction':args.direction,'continuous_start':args.continuous,'state_source':str(args.state) if args.state else None,'closed_goal_rad':f.closed_q.tolist(),'load_limit_N':args.load_limit,'method':'mjbatch upstream ThreadPool with MuJoCo 3.11 C++ rollouts',
                'native_library':str(args.native_library),
                'control_dt_s':f.cfg.control_dt,
                'acceleration_limit_enabled':f.cfg.acceleration_limit_enabled,
                'scope':'curl stage only; no docking or full FSM validation','threads':args.threads,'count':n,
                'elapsed_s':elapsed,'candidate_steps':int(lengths.sum()),'max_serial_state_difference':difference,'candidates':candidates}
        path=args.output_dir/f'{args.direction}_curl_accel_{args.accel_factor:g}.json';path.write_text(json.dumps(report,indent=2)+'\n')
        valid=[x for x in candidates if x['status']==0];valid.sort(key=lambda x:x['predicted_duration_s'])
        print('BATCH',path,'elapsed_s',round(elapsed,3),'valid',len(valid),'baseline max state difference',difference,flush=True)
        print('FASTEST',json.dumps(valid[:5],ensure_ascii=False),flush=True)
    finally:f.close()


if __name__=='__main__':main()
