"""Re-simulate and visualize a successful planned mjbatch run."""
import argparse
import json
import math
import queue
import time
from pathlib import Path

import numpy as np

from scripts import view_headtail_bidirectional as bidirectional
from scripts import view_headtail_mjbatch_refined as refined
from scripts.run_planned_mjbatch import configure_servo
from scripts.planned_docking_feedback import enable_staged_feedback, enable_normal_guard, enable_curl_offset, enable_internal_docking_v1, enable_predictive_docking_v1
from scripts.planned_run_index import latest_execution

RUNS = Path('outputs/reconfiguration/planned_runs')


def load_run(path):
    if path.is_dir():
        path = path / 'result.json'
    result = json.loads(path.read_text())
    if not result.get('success'):
        raise ValueError(f'run did not pass full validation: {path}')
    solution = json.loads((path.parent / 'kinematic_solution.json').read_text())
    if solution.get('status') != 'solution_found':
        raise ValueError(f'no valid kinematic solution: {path.parent}')
    goal = np.asarray(solution['joint_angle_rad'], dtype=float)
    speed = float(result['candidate']['curl_speed_multiplier'])
    if goal.shape != (6,) or not np.isfinite(goal).all() or not math.isfinite(speed) or speed <= 0:
        raise ValueError(f'invalid solution or speed: {path.parent}')
    reference = Path(result['unfold_controls'])
    if not reference.is_file():
        raise FileNotFoundError(f'unfold controls not found: {reference}')
    return result, goal, speed, reference


def latest_run():
    for path in sorted(RUNS.glob('*/result.json'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            result, _, _, _ = load_run(path)
            if result.get('normal_guard') == 'site_negative_z_positive_clearance_30mm_1ms_v1':
                return path
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            continue
    raise FileNotFoundError(f'no successful result with the normal-plane guard in {RUNS}; run ./run_planned_mjbatch.sh first')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, help='run directory or its result.json; default: latest execution')
    parser.add_argument('--headless', action='store_true', help='re-run without a window')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed relative to simulated time')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive')
    try:
        path = args.result or latest_execution()
        if path.is_dir() and not (path / 'result.json').is_file():
            raise FileNotFoundError(f'latest execution has no result.json yet: {path}; it may still be running or have failed')
        result, goal, factor, reference = load_run(path)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    # create_refined reads this path when the controller is created.
    refined.REFERENCE = reference
    if 'normal_guard' not in result:
        print('WARNING: historical result did not enforce the normal-plane guard', flush=True)
    def create():
        settings = result.get('polish_settings', {})
        if settings.get('headtail_mode') == 'four_site_exponential_then_z_v2':
            from scripts.polish_headtail_axial import create_refined_with_landmarks
            f = create_refined_with_landmarks(settings['landmark_radius_m'])
        else:
            f = refined.create_refined('forward')
        if 'servo' in result:
            configure_servo(f, float(result['servo']['kp']), float(result['servo']['kd']))
        if 'polish_settings' in result:
            from scripts.polish_planned_mjbatch import configure_full_candidate
            configure_full_candidate(f, result['polish_settings'])
        elif result.get('docking_feedback') in ('site_xy_orientation_then_z_v1', 'site_xy_orientation_then_z_v2'):
            enable_staged_feedback(f, guarded=result['docking_feedback'].endswith('_v2'))
        if 'polish_settings' not in result and result.get('normal_guard') == 'site_negative_z_positive_clearance_30mm_1ms_v1':
            enable_normal_guard(f)
            enable_curl_offset(f, result['candidate']['offset_rad'])
        if 'polish_settings' in result:
            pass
        elif result.get('internal_docking') == 'separate_rotate_early_feedback_v1':
            enable_internal_docking_v1(f)
        elif result.get('internal_docking') == 'separate_rotate_predictive_feedback_v2':
            enable_internal_docking_v1(f, early_switch=False)
            enable_predictive_docking_v1(f)
        elif result.get('internal_docking') == 'separate_rotate_predictive_feedback_v3':
            enable_internal_docking_v1(f, early_switch=False)
            enable_predictive_docking_v1(f, alignment_clearance=.010)
        if 'polish_controls' in result:
            from scripts.polish_planned_mjbatch import install_controls
            with np.load(result['polish_controls']) as saved:
                install_controls(f, saved['controls'].copy())
        if result.get('polish_safety') in ('module_geom_1ms_speed_2pi_v1', 'module_geom_1ms_speed_2pi_v2', 'module_geom_1ms_speed_2pi_v3', 'module_geom_1ms_speed_2pi_v4'):
            from scripts.polish_safety import CollisionAudit
            audit = CollisionAudit(f, allow_aligned_mating=not result['polish_safety'].endswith('_v1'),
                                   allow_old_mating=result['polish_safety'].endswith(('_v3','_v4')),
                                   enforce_source_limits=result['polish_safety'].endswith('_v4'))
            previous_measure = f.measure
            def measure():
                previous_measure()
                audit.sample()
                if np.any(f.data.warning.number):
                    f.stop('MuJoCo warning')
            f.measure = measure
        f.closed_q = goal.copy()
        f.cfg.curl_speed_multiplier = factor
        return f
    f = create()
    print(f'REPLAY {path} curl_speed_multiplier={factor:g} unfold={reference}', flush=True)
    try:
        if args.headless:
            while f.phase not in ('DONE', 'FAILED') and f.data.time < 60:
                f.step()
            print(f'REPLAY_RESULT phase={f.phase} simulated_time_s={f.data.time:.3f} '
                  f'original_time_s={result["simulated_time_s"]:.3f} failure={f.failure}', flush=True)
            return 0 if f.phase == 'DONE' else 1
        import mujoco
        import mujoco.viewer
        keys = queue.SimpleQueue()
        def on_key(key):
            if key in (13, 257, 335):
                keys.put(key)
        with mujoco.viewer.launch_passive(f.model, f.data, key_callback=on_key) as viewer:
            viewer.cam.lookat[:] = [0., 0., .08]
            viewer.cam.distance = 1.4
            viewer.cam.elevation = -55
            anchor = time.perf_counter()
            anchor_sim = float(f.data.time)
            next_status = 0.
            while viewer.is_running():
                if not keys.empty():
                    keys.get()
                    fresh = create()
                    with viewer.lock():
                        f = bidirectional.swap(f, fresh)
                    anchor = time.perf_counter()
                    anchor_sim = float(f.data.time)
                if f.phase not in ('DONE', 'FAILED'):
                    desired = anchor_sim + (time.perf_counter() - anchor) * args.speed
                    deadline = time.perf_counter() + .012
                    while f.data.time < desired and time.perf_counter() < deadline and f.phase not in ('DONE', 'FAILED'):
                        f.step()
                if time.perf_counter() >= next_status:
                    peak = max((x['load_N'] for x in f.phase_loads.values()), default=0.)
                    labels = 'Time\nPhase\nTarget face\nFace loads 1-6 (N)\nPeak other (N)\nConnected 1-6\nControls'
                    values = (f'{f.data.time:.2f}s\n{f.phase}\n{f.target + 1}\n'
                              + ' '.join(f'{x:.1f}' for x in f.loads) + f'\n{peak:.2f}\n'
                              + ' '.join('1' if x else '0' for x in f.connected()) + '\nEnter: replay')
                    if f.failure:
                        labels += '\nFailure'
                        values += '\n' + f.failure[:160]
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                      mujoco.mjtGridPos.mjGRID_TOPRIGHT.value, labels, values))
                    next_status = time.perf_counter() + .1
                viewer.sync()
                time.sleep(.002)
        return 0
    finally:
        f.close()


if __name__ == '__main__':
    raise SystemExit(main())
