"""Solve and screen a forward reconfiguration, then run the complete FSM.

This reproduces the refined viewer's MuJoCo 3.11 workflow. The native batch
uses mjbatch's C++ thread pool; it is not the mjbatch.Batch Python API.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts import view_headtail_bidirectional as viewer
from scripts import view_headtail_mjbatch_refined as refined
from scripts.optimize_headtail_mjbatch import LIB
from zbot_rl_mjlab.tasks.reconfiguration.closure_states import ClosedRegistrationAudit
import mujoco
from scripts.planned_docking_feedback import enable_staged_feedback, enable_normal_guard, enable_internal_docking_v1, enable_predictive_docking_v1
from scripts.planned_run_index import record_execution


def configure_servo(f, kp, kd):
    f.cfg.stiffness = f.cfg.outer_stiffness = kp
    f.cfg.damping = f.cfg.outer_damping = kd
    f.set_pd()
    mujoco.mj_forward(f.model, f.data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument('--candidates', type=int, default=21)
    p.add_argument('--min-speed', type=float, default=10.)
    p.add_argument('--max-speed', type=float, default=12.)
    p.add_argument('--max-validations', type=int, default=3)
    p.add_argument('--offset-rad', type=float, default=0.3, help='per-joint smooth CURL path perturbation')
    p.add_argument('--kp', type=float, default=15.)
    p.add_argument('--kd', type=float, default=1.5)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.threads < 1 or args.candidates < 2 or not 0 < args.min_speed < args.max_speed or args.max_validations < 1 or not np.isfinite([args.kp, args.kd,args.offset_rad]).all() or args.kp <= 0 or args.kd < 0 or args.offset_rad < 0:
        p.error('invalid search parameters')
    out = args.output_dir or Path('outputs/reconfiguration/planned_runs') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    out = out.resolve()
    record_execution(out, 'run')
    print(f'RESULT_DIR {out}', flush=True)
    seed_file = viewer.visual.out / 'verification_90_3_1245.json'
    seed = json.loads(seed_file.read_text())['branch_q']
    audit = ClosedRegistrationAudit(face_angles_deg=[270, 270, 270, 270, 270, 90])
    solution = audit.solve([0, 0, 0, 0, 0, 1], [seed])
    (out / 'kinematic_solution.json').write_text(json.dumps(solution, indent=2) + '\n')
    if solution['status'] != 'solution_found':
        raise RuntimeError('kinematic closure solution failed; see kinematic_solution.json')
    goal = solution['joint_angle_rad']
    (out / 'goals.json').write_text(json.dumps([goal]) + '\n')
    print('IK', json.dumps(solution, ensure_ascii=False), flush=True)
    if not LIB.is_file():
        subprocess.run([sys.executable, 'scripts/build_mjbatch_rollout.py'], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.optimize_headtail_mjbatch', '--direction', 'forward',
                    '--threads', str(args.threads), '--count', str(args.candidates),
                    '--low', str(args.min_speed), '--high', str(args.max_speed),
                    '--kp', str(args.kp), '--kd', str(args.kd),
                    '--offset-rad', str(args.offset_rad),
                    '--seed-candidate', 'scripts/planned_curl_seed.json',
                    '--goal-file', str(out / 'goals.json'), '--output-dir', str(out)], check=True)
    search = json.loads((out / 'forward_curl_accel_1.json').read_text())
    candidates = sorted((row for row in search['candidates'] if row['status'] == 0
                         and args.min_speed <= row['curl_speed_multiplier'] <= args.max_speed),
                        key=lambda row: (row is not search['candidates'][0], row['predicted_duration_s']))
    if not candidates:
        raise RuntimeError('no curl candidate passed batch screening')
    validations = []
    for candidate in candidates[:args.max_validations]:
        f = refined.create_refined('forward')
        try:
            configure_servo(f, args.kp, args.kd)
            enable_staged_feedback(f, guarded=True)
            enable_normal_guard(f)
            f.closed_q = np.asarray(goal)
            f.cfg.curl_speed_multiplier = candidate['curl_speed_multiplier']
            from scripts.planned_docking_feedback import enable_curl_offset
            enable_curl_offset(f, candidate['offset_rad'])
            enable_internal_docking_v1(f, early_switch=False)
            enable_predictive_docking_v1(f, alignment_clearance=.010)
            print('VALIDATE_FULL_FSM', candidate['curl_speed_multiplier'], flush=True)
            while f.phase not in ('DONE', 'FAILED') and f.data.time < 60:
                f.step()
            report = f.report()
            report['phase_active_loads'] = f.phase_loads
            report['candidate'] = candidate
            report['timeout'] = f.phase not in ('DONE', 'FAILED')
            report['failure'] = f.failure
            validations.append(report)
            (out / f'validation_{len(validations)}.json').write_text(json.dumps(report, indent=2) + '\n')
            if report['success']:
                result = {'success': True, 'candidate': candidate, 'simulated_time_s': float(f.data.time),
                          'servo': {'kp': args.kp, 'kd': args.kd},
                          'docking_feedback': 'site_xy_orientation_then_z_v2',
                          'normal_guard': 'site_negative_z_positive_clearance_30mm_1ms_v1',
                          'internal_docking': 'separate_rotate_predictive_feedback_v3',
                          'ik_seed': str(seed_file), 'unfold_controls': str(refined.REFERENCE),
                          'batch_method': 'mjbatch C++ ThreadPool and MuJoCo native rollouts',
                          'batch_scope': 'CURL only; subsequent full FSM separately validated',
                          'collision_note': 'viewer collision filtering applies; no strict head-tail pre-alignment collision audit',
                          'validation_report': f'validation_{len(validations)}.json'}
                (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                print('RESULT', json.dumps(result, ensure_ascii=False), flush=True)
                return
            print('VALIDATION_FAILED', f.failure, flush=True)
        finally:
            f.close()
    (out / 'result.json').write_text(json.dumps({'success': False, 'validations': len(validations)}, indent=2) + '\n')
    raise RuntimeError('no screened candidate passed full FSM validation')


if __name__ == '__main__':
    main()
