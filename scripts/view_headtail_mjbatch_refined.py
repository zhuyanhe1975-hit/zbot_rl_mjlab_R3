"""View the mjbatch-refined forward/reverse motions with an 80 N margin guard.

1/2 continue from the current compatible state; Enter resets the current mode.
The forward unfold controls were selected offline by native parallel rollouts.
"""
from pathlib import Path
import json,math
import mujoco
import numpy as np
import scripts.view_headtail_bidirectional as viewer
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import rz

REFERENCE=Path('outputs/reconfiguration/mjbatch_refinement/unfold_best_1024.npz')
REVERSE_REFERENCE=Path('outputs/reconfiguration/mjbatch_refinement/reverse_stability/selected.json')
REVERSE_UNFOLD=Path('outputs/reconfiguration/mjbatch_refinement/reverse_fast/unfold_best_1024.npz')
NON_TARGET_LOAD_LIMIT=80.
USE_RECORDED_UNFOLD=True
original_create=viewer.create
original_reverse_from_current=viewer.reverse_from_current
original_forward_from_current=viewer.forward_from_current


def create_refined(direction):
    f=original_create(direction)
    try:
        f.cfg.other_stop_load=NON_TARGET_LOAD_LIMIT
        if direction=='forward':
            f.cfg.curl_speed_multiplier=12.
            f.cfg.closest_unfold_ramp_s=4.
            with np.load(REFERENCE) as data:controls=data['control'].copy()
            old_apply=f.apply
            if USE_RECORDED_UNFOLD:f.closest_unfold_update=lambda:None
            def apply():
                if f.phase=='UNFOLD' and USE_RECORDED_UNFOLD:
                    i=min(max(0,int(f.phase_time/f.cfg.dt)),len(controls)-1)
                    f.qref=controls[i].copy();f.vref[:]=0.
                old_apply()
            f.apply=apply
        else:
            selected=json.loads(REVERSE_REFERENCE.read_text())
            f.closed_q=np.asarray(selected['goal'],dtype=float)
            f.cfg.curl_speed_multiplier=selected['candidate']['curl_speed_multiplier']
            f.cfg.speed_deg_s*=.7;f.cfg.acceleration_deg_s2*=.49
            f.cfg.closest_unfold_gain=6.;f.cfg.closest_unfold_max_speed=4.8;f.cfg.closest_unfold_ramp_s=4.
            # Use the 270-degree temporary brace; it is passive and opens at
            # the end of the reverse sequence just as the former 90-degree one.
            m=f.model;original=m.site('module_5/port_out').id;site=m.site('module_5/search_port').id
            delta=math.pi/2;turn=rz(delta)[:3,:3];inlet=m.site_pos[m.site('module_0/port_in').id]
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.site_quat[original]);rotation=rotation.reshape(3,3)
            m.site_pos[site]=m.site_pos[original]+rotation@(turn@inlet-inlet)
            quat=np.empty(4);mujoco.mju_mulQuat(quat,m.site_quat[original],np.array([math.cos(delta/2),0.,0.,math.sin(delta/2)]));m.site_quat[site]=quat
            old_sync=f.planner.sync
            def sync(state,cut):
                anchor=old_sync(state,cut);f.planner.kin.registration[5]=delta;return anchor
            f.planner.sync=sync
            with np.load(REVERSE_UNFOLD) as data:unfold_controls=data['control'].copy()
            f.reverse_fast_unfold=False
            old_unfold=f.closest_unfold_update
            def unfold_update():
                if not f.reverse_fast_unfold or not USE_RECORDED_UNFOLD:old_unfold()
            f.closest_unfold_update=unfold_update
            old_apply=f.apply;offset=np.asarray(selected['candidate']['offset_rad']);first_step=None
            def apply_reverse():
                nonlocal first_step
                if f.phase=='UNFOLD' and f.reverse_fast_unfold and USE_RECORDED_UNFOLD:
                    index=min(max(0,int(f.phase_time/f.cfg.dt)),len(unfold_controls)-1)
                    f.qref=unfold_controls[index].copy();f.vref[:]=0.
                old_apply()
                if f.phase=='CURL':
                    if first_step is None:first_step=f.step_count
                    n=math.ceil(f.curve.duration/f.cfg.control_dt-1e-10)*round(f.cfg.control_dt/f.cfg.dt)
                    i=min(max(f.step_count-first_step,0),n-1)
                    f.data.ctrl[f.actuators]+=math.sin(math.pi*i/(n-1))**2*offset
                    demand=f.kp*(f.data.ctrl[f.actuators]-f.data.qpos[f.qadr])-f.kd*f.data.qvel[f.dofs]
                    if not np.isfinite(demand).all() or np.max(np.abs(demand))>f.cfg.peak_torque:
                        f.stop('requested curl torque protection')
            f.apply=apply_reverse
            old_advance=f.advance
            def advance_reverse():
                if f.phase=='UNFOLD' and np.max(np.abs(f.data.qpos[f.qadr]))<.15:
                    f.qref=f.data.qpos[f.qadr].copy();f.vref[:]=0.
                    f.transition('FINAL_STRAIGHT_HOLD');return
                if f.phase=='FINAL_STRAIGHT_HOLD':
                    if not f.connected()[:5].all() or f.connected()[5]:f.stop('final chain topology incorrect')
                    elif f.phase_time>=f.cfg.settle_seconds-.5*f.cfg.dt:
                        gap,_,_,_=f.geometry(5)
                        if np.any(f.data.eq_active[f.new]):f.stop('final registration angles incorrect')
                        elif (np.max(np.abs(f.data.qpos[f.qadr]))>=.15 or np.max(np.abs(f.data.qvel[f.dofs]))>=.05
                              or f.snapshot()['root_collinearity_error_m']>=.04 or gap<.05):
                            f.stop('final near-straight pose or separation tolerance not met')
                        else:f.transition('DONE')
                    return
                old_advance()
            f.advance=advance_reverse
            mujoco.mj_forward(m,f.data)
        old_measure=f.measure
        def measure():
            old_measure()
            active=f.connected().copy()
            if f.phase in ('LOAD','LOAD_HEADTAIL'):active[f.target]=False
            if np.any(f.loads[active]>NON_TARGET_LOAD_LIMIT):
                f.stop(f'non-target load limit violated: {f.loads.tolist()}')
        f.measure=measure
        return f
    except BaseException:
        f.close();raise


def reverse_from_current_refined(previous):
    f=original_reverse_from_current(previous)
    f.reverse_fast_unfold=True
    return f


def forward_from_current_refined(previous):
    if (not np.all(previous.connected()[:5]) or previous.connected()[5]
            or np.any(previous.data.eq_active[previous.new])
            or np.max(np.abs(previous.data.qpos[previous.qadr]))>=.15
            or np.max(np.abs(previous.data.qvel[previous.dofs]))>=.05):
        raise ValueError('Forward requires a stationary near-straight all-180 chain with head-tail open')
    f=create_refined('forward')
    try:
        mujoco.mj_copyData(f.data,f.model,previous.data)
        f.phase_start=float(f.data.time);f.initial_qpos=f.data.qpos.copy()
        f.qref=f.data.qpos[f.qadr].copy();f.next_log=f.phase_start;f.next_progress=f.phase_start
        f.set_contact_phase();f.set_pd();mujoco.mj_forward(f.model,f.data)
        f.cfg.speed_deg_s*=.8;f.cfg.acceleration_deg_s2*=.64
        return f
    except BaseException:
        f.close();raise


if __name__=='__main__':
    viewer.create=create_refined
    viewer.reverse_from_current=reverse_from_current_refined
    viewer.forward_from_current=forward_from_current_refined
    raise SystemExit(viewer.main())
