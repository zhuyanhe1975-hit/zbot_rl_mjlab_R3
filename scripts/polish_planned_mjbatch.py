"""Screen safe candidates and optimize full reconfiguration time."""
import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts import view_headtail_mjbatch_refined as refined
from scripts.optimize_headtail_mjbatch import lib_open, STATE
from scripts.planned_docking_feedback import (enable_curl_offset, enable_internal_docking_v1,
    enable_normal_guard, enable_predictive_docking_v1, enable_staged_feedback)
from scripts.run_planned_mjbatch import configure_servo
from scripts.view_planned_mjbatch import latest_run, load_run
from scripts.planned_run_index import record_execution
from scripts.polish_safety import CollisionAudit, source_joint_limits
from zbot_rl_mjlab.tasks.reconfiguration.closure_states import ClosedRegistrationAudit
from scripts.polish_headtail_axial import (prealign_goal, enable_axial_docking,
    enable_four_site_axial_docking, create_refined_with_landmarks)
from scripts.polish_release import enable_immediate_release
from scripts.polish_release_mpc import enable_release_mpc


def make_f(result, goal):
    f = refined.create_refined('forward')
    configure_servo(f, float(result['servo']['kp']), float(result['servo']['kd']))
    enable_staged_feedback(f, guarded=True)
    enable_normal_guard(f)
    f.closed_q = goal.copy()
    f.cfg.curl_speed_multiplier = float(result['candidate']['curl_speed_multiplier'])
    enable_curl_offset(f, result['candidate']['offset_rad'])
    enable_internal_docking_v1(f, early_switch=False)
    enable_predictive_docking_v1(f, alignment_clearance=.010)
    return f


def install_controls(f, controls):
    old = f.apply
    first = None
    def apply():
        nonlocal first
        old()
        if f.phase == 'APPROACH' and f.target == 3:
            if first is None:
                first = f.step_count
            i = min(f.step_count - first, len(controls) - 1)
            f.data.ctrl[f.actuators] = controls[i]
            demand = f.kp * (f.data.ctrl[f.actuators] - f.data.qpos[f.qadr]) - f.kd * f.data.qvel[f.dofs]
            if not np.isfinite(demand).all() or np.max(np.abs(demand)) > f.cfg.peak_torque:
                f.stop('polished control torque protection')
    f.apply = apply


def configure_full_candidate(f, settings):
    """Configure only the parameters explored by the full-cycle search."""
    f.cfg.control_dt = float(settings.get('control_dt_s', .02))
    f.cfg.acceleration_limit_enabled = not settings.get('no_acceleration_limit', False)
    f.cfg.raw_torque_limit_enabled = not settings.get('actuator_force_saturation', False)
    if settings.get('actuator_force_saturation', False):
        if not np.all(f.model.actuator_forcelimited[f.actuators]) or not np.allclose(
                f.model.actuator_forcerange[f.actuators], [-20., 20.]):
            raise ValueError('polish requires enabled actuator force saturation at ±20 Nm')
    f.cfg.speed_deg_s *= settings['internal_speed_scale']
    enable_staged_feedback(f, guarded=True)
    enable_normal_guard(f, current_geometry=settings.get('poststep_geometry', False))
    enable_curl_offset(f, settings['curl']['offset_rad'])
    enable_internal_docking_v1(f, early_switch=False,
                               separate_rotation=settings.get('separate_internal_rotation', False))
    enable_predictive_docking_v1(f, alignment_clearance=settings['alignment_clearance_m'])
    if settings.get('headtail_mode') == 'five_axis_exponential_then_z_v1':
        enable_axial_docking(f, settings['closure_goal_rad'], pre_gap=settings['prealign_gap_m'])
    elif settings.get('headtail_mode') == 'four_site_exponential_then_z_v2':
        enable_four_site_axial_docking(f, pre_gap=settings['prealign_gap_m'],
                                        quarter=settings['landmark_quarter'])
    if settings.get('release_mode') == 'immediate_substep_v1':
        enable_immediate_release(f)
    if 'release_mpc' in settings:
        enable_release_mpc(f, **settings['release_mpc'])


def run_candidate(task):
    import contextlib
    candidate, goal, servo, reference, out_path = task
    settings = candidate['settings']
    output = Path(out_path)
    with (output / f"candidate_{candidate['index']}.log").open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        refined.REFERENCE = Path(reference)
        f = (create_refined_with_landmarks(settings['landmark_radius_m'])
             if settings.get('headtail_mode') == 'four_site_exponential_then_z_v2'
             else refined.create_refined('forward'))
        try:
            configure_servo(f, servo['kp'], servo['kd'])
            f.closed_q = np.asarray(goal, dtype=float)
            f.cfg.curl_speed_multiplier = settings['curl']['curl_speed_multiplier']
            configure_full_candidate(f, settings)
            collision_audit = CollisionAudit(f, allow_aligned_mating=True, allow_old_mating=True,
                                             enforce_source_limits=settings.get('source_joint_limits', False))
            tracked = {'min_normal_m': None, 'normal_crossings': 0, 'samples': 0}
            original_measure = f.measure
            def measure():
                original_measure()
                collision_audit.sample()
                if f.phase in ('CURL', 'CAPTURE_HEADTAIL', 'APPROACH'):
                    eq = f.old[5] if f.phase in ('CURL', 'CAPTURE_HEADTAIL') else f.new[f.target]
                    a, b = int(f.model.eq_obj1id[eq]), int(f.model.eq_obj2id[eq])
                    rel = f.data.site_xmat[b].reshape(3, 3).T @ (f.data.site_xpos[a] - f.data.site_xpos[b])
                    if np.linalg.norm(rel) <= .03:
                        z = -float(rel[2]); tracked['samples'] += 1
                        tracked['min_normal_m'] = z if tracked['min_normal_m'] is None else min(tracked['min_normal_m'], z)
                        if z < 0.: tracked['normal_crossings'] += 1
                if np.any(f.data.warning.number):
                    f.stop('MuJoCo warning')
            f.measure = measure
            while f.phase not in ('DONE', 'FAILED') and f.data.time < 60:
                f.step()
            report = f.report()
            report['failure'] = f.failure
            report['phase_active_loads'] = f.phase_loads
            report['normal_audit'] = tracked
            report['headtail_axial'] = getattr(f, 'axial_metrics', None)
            report['release_mpc'] = getattr(f, 'release_mpc_stats', None)
            report['collision_audit'] = collision_audit.report()
            report['limit_scope'] = {'joint_position_limits_enabled_in_model': bool(np.any(f.model.jnt_limited[[f.model.joint(f'module_{i}/joint{i+1}').id for i in range(6)]])),
                                     'source_asset_joint_limits_rad':source_joint_limits().tolist(),
                                     'source_asset_limits_audited':bool(settings.get('source_joint_limits', False)),
                                     'note':'asset range is not independently verified as hardware limit'}
            report['polish_settings'] = settings
            events = report['events']
            times = {}
            for e in events:
                if e.get('from') is not None:
                    key = e['from']; times[key] = times.get(key, 0.) + float(e.get('elapsed_s', 0.))
            report['phase_durations_s'] = times
            (output / f"validation_{candidate['index']}.json").write_text(json.dumps(report, indent=2)+'\n')
            terminal = report['terminal']
            return {'index':candidate['index'], 'branch':candidate['branch'], 'success':bool(f.phase=='DONE' and tracked['normal_crossings']==0),
                    'failure':f.failure, 'simulated_time_s':float(f.data.time),
                    'peak_other_load_N':terminal['peak_other_load_N'],
                    'peak_torque_Nm':terminal['peak_actual_torque_Nm'],
                    'max_module_penetration_m':terminal['max_module_penetration_m'],
                    'max_ground_penetration_m':terminal['max_ground_penetration_m'],
                    'normal_audit':tracked, 'phase_durations_s':times,
                    'headtail_axial':report['headtail_axial'],
                    'release_mpc':report['release_mpc'],
                    'collision_audit':collision_audit.report(),
                    'limit_scope':report['limit_scope'],
                    'settings':settings, 'unfold_controls':reference,
                    'validation_report':f"validation_{candidate['index']}.json"}
        finally:
            f.close()


def main():
    import subprocess
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    p = argparse.ArgumentParser(description='Batch screen and optimize full reconfiguration cycles')
    p.add_argument('--result', type=Path, help='validated starting result')
    p.add_argument('--count', type=int, default=32, help='full-cycle candidates, including baseline')
    p.add_argument('--batch-count', type=int, default=256, help='CURL candidates in mjbatch screening')
    p.add_argument('--branches', type=int, choices=(1, 2), default=2, help='number of closed-ring IK branches')
    p.add_argument('--threads', type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument('--workers', type=int, default=max(1, min(2, len(os.sched_getaffinity(0)) // 8)))
    p.add_argument('--mpc-candidates', type=int, default=1024)
    p.add_argument('--mpc-horizon-ms', type=int, default=20)
    p.add_argument('--mpc-threads', type=int, default=4)
    p.add_argument('--mpc-offset-rad', type=float, default=.4)
    p.add_argument('--mpc-distance-margin-mm',type=float,default=.5)
    p.add_argument('--landmark-radius-mm',type=float,default=50.,
                   help='virtual four-magnet guidance radius in mm; measure real positions before physical use')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.count < 1 or args.batch_count < 2 or args.threads < 1 or args.workers < 1 or args.mpc_candidates < 2 or args.mpc_horizon_ms < 1 or args.mpc_threads < 1 or not np.isfinite(args.mpc_offset_rad) or args.mpc_offset_rad <= 0 or not np.isfinite(args.mpc_distance_margin_mm) or not 0 <= args.mpc_distance_margin_mm < 50 or not np.isfinite(args.landmark_radius_mm) or args.landmark_radius_mm <= 0:
        p.error('count, batch-count, threads and workers must be positive')
    source = args.result or latest_run()
    result, goal, _, reference = load_run(source)
    if result.get('internal_docking') != 'separate_rotate_predictive_feedback_v3':
        p.error('source must be the v3 predictive planned result')
    out = args.output_dir or Path('outputs/reconfiguration/full_polish') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False); out = out.resolve()
    record_execution(out, 'polish')
    print('RESULT_DIR', out, flush=True)
    seed = {'curl_speed_multiplier': result['candidate']['curl_speed_multiplier'], 'offset_rad':result['candidate']['offset_rad']}
    (out/'seed.json').write_text(json.dumps(seed)+'\n')
    native_library = out / 'libcurl_strict.so'
    if not np.allclose(source_joint_limits(), [-6.28, 6.28], atol=1e-12, rtol=0):
        raise ValueError('source asset joint limits changed; update strict native screening before running')
    subprocess.run([sys.executable, 'scripts/build_mjbatch_rollout.py', '--source',
                    'scripts/mjbatch_curl_rollout_strict.cpp', '--output', str(native_library)], check=True)
    goals=[goal]
    if args.branches == 2:
        audit=ClosedRegistrationAudit(face_angles_deg=[270,270,270,270,270,90])
        solution=audit.solve([0,0,0,0,0,1],[goal[::-1]])
        if solution['status']=='solution_found':
            alternate=np.asarray(solution['joint_angle_rad'])
            difference=(alternate-goal+np.pi)%(2*np.pi)-np.pi
            if np.linalg.norm(difference)>.1: goals.append(alternate)
    rng=np.random.default_rng(args.seed)
    groups=[]
    for branch,closure_goal in enumerate(goals):
        kinematic_f=refined.create_refined('forward')
        try:
            branch_goal=prealign_goal(kinematic_f,closure_goal,.03)
        finally:
            kinematic_f.close()
        directory=out / f'branch_{branch}'; directory.mkdir()
        (directory/'goals.json').write_text(json.dumps([branch_goal.tolist()])+'\n')
        cmd=[sys.executable, '-m', 'scripts.optimize_headtail_mjbatch', '--direction', 'forward',
            '--threads',str(args.threads),'--count',str(args.batch_count),'--low','5','--high','12',
            '--offset-rad','.5','--kp',str(result['servo']['kp']),'--kd',str(result['servo']['kd']),
            '--goal-file',str(directory/'goals.json'),'--output-dir',str(directory),'--native-library',str(native_library),
            '--control-dt','0.001','--no-acceleration-limit']
        if branch==0: cmd += ['--seed-candidate',str(out/'seed.json')]
        subprocess.run(cmd,check=True)
        search=json.loads((directory/'forward_curl_accel_1.json').read_text())
        feasible=[x for x in search['candidates'] if x['status']==0]
        print(f'STRICT_CURL branch={branch} passed={len(feasible)} / {len(search["candidates"])}',flush=True)
        if feasible:
            feasible=[feasible[i] for i in rng.permutation(len(feasible))]
            groups.append((branch,branch_goal,closure_goal,feasible))
    if not groups:
        (out/'result.json').write_text(json.dumps({'success':False,'reason':'no CURL candidate passed strict batch safety screening'},indent=2)+'\n')
        raise SystemExit(f'No safe CURL candidate found. Batch reports: {out}')
    with np.load(reference) as data: unfold=data['control'].copy()
    tasks=[]
    for i in range(args.count):
        branch, branch_goal, closure_goal, feasible=groups[i%len(groups)]
        speed_scale=1. if i==0 else float(rng.uniform(.8,1.3))
        clearance=.010 if i==0 else float(rng.uniform(.008,.015))
        unfold_speed=1. if i==0 else float(rng.uniform(1.,1.3))
        if i==0: candidate_ref=reference.resolve()
        else:
            indexes=np.minimum(np.arange(int(np.ceil(len(unfold)/unfold_speed)))*unfold_speed,len(unfold)-1)
            grid=np.arange(len(unfold)); control=np.stack([np.interp(indexes,grid,unfold[:,j]) for j in range(6)],axis=1)
            candidate_ref=out/f'unfold_{i}.npz'; np.savez(candidate_ref,control=control)
        settings={'curl':feasible[(i//len(groups))%len(feasible)],'internal_speed_scale':speed_scale,
                  'alignment_clearance_m':clearance,'unfold_speedup':unfold_speed,
                  'headtail_mode':'four_site_exponential_then_z_v2',
                  'landmark_quarter':1, 'landmark_radius_m':args.landmark_radius_mm / 1000.,
                  'landmark_radius_calibrated':False,
                  'closure_goal_rad':closure_goal.tolist(), 'prealign_gap_m':.03, 'control_dt_s':.001,
                  'poststep_geometry':True, 'source_joint_limits':True,
                  'no_acceleration_limit':True, 'actuator_force_saturation':True,
                  'separate_internal_rotation':True,
                  'release_mpc':{'library':str(native_library),'candidates':args.mpc_candidates,
                                 'horizon_ms':args.mpc_horizon_ms,'threads':args.mpc_threads,
                                 'seed':args.seed+i, 'action_span_rad':args.mpc_offset_rad,
                                 'replan_on_clear':True,'modes':['SEPARATE','ROTATE','APPROACH'],
                                 'distance_margin_m':args.mpc_distance_margin_mm / 1000.}}
        tasks.append(({'index':i,'settings':settings,'branch':branch},branch_goal.tolist(),result['servo'],str(candidate_ref),str(out)))
    print(f'FULL_SEARCH candidates={len(tasks)} workers={args.workers}',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(run_candidate,t):t[0]['index'] for t in tasks}
        for future in as_completed(futures):
            i=futures[future]
            try: row=future.result()
            except Exception as exc: row={'index':i,'success':False,'failure':f'{type(exc).__name__}: {exc}',
                                          'settings': tasks[i][0]['settings']}
            rows.append(row);print('CANDIDATE',json.dumps(row,ensure_ascii=False),flush=True)
    rows.sort(key=lambda x:x['index'])
    (out/'search.json').write_text(json.dumps({'source':str(source),'method':'mjbatch C++ ThreadPool CURL screening plus parallel full FSM simulations',
        'scope':'1ms inter-module geometry checks except aligned or connected mating pairs; ground support permitted; site normal guard limited to CURL, CAPTURE_HEADTAIL, APPROACH and 30mm',
        'workers':args.workers,'candidates':rows},indent=2)+'\n')
    valid=sorted((x for x in rows if x['success']),key=lambda x:x['simulated_time_s'])
    if not valid:
        from collections import Counter
        reasons = dict(Counter((r.get('failure') or 'unknown').split(':', 1)[0] for r in rows))
        (out/'result.json').write_text(json.dumps({'success':False,'candidates':len(rows), 'failure_reasons':reasons},indent=2)+'\n')
        raise SystemExit(f'No full-cycle candidate passed safety validation: {reasons}. Details: {out / "search.json"}')
    best=valid[0]; chosen=dict(result)
    chosen.update({'success':True,'simulated_time_s':best['simulated_time_s'], 'candidate':best['settings']['curl'],
                   'polish_settings':best['settings'],'unfold_controls':best['unfold_controls'],
                   'polish_safety': 'module_geom_1ms_speed_2pi_v4',
                   'validation_report':best['validation_report'],'polish_source':str(source)})
    chosen_goal=tasks[best['index']][1]
    (out/'kinematic_solution.json').write_text(json.dumps({'status':'solution_found','joint_angle_rad':chosen_goal},indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print('BEST',json.dumps(best,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
