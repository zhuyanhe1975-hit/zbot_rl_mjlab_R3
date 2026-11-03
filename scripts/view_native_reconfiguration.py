"""Replay recorded native-adhesion reconstruction states in MuJoCo's viewer.

This reads the saved model and actual integration states from an execution;
it does not rerun the controller or infer a trajectory from summary events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import time
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_RESULT = Path('outputs/reconfiguration/native_fsm/result.json')
DEFAULT_INDEX = Path('outputs/reconfiguration/native_fsm/latest_execution.json')


def latest_recording(index=DEFAULT_INDEX, default=DEFAULT_RESULT):
    """Resolve the latest completed execution, including custom result paths."""
    if not index.exists():
        return default
    entry = json.loads(index.read_text())
    result_path = Path(entry['result_path'])
    if not result_path.is_absolute():
        raise ValueError(f'latest execution index must contain an absolute result path: {index}')
    report = json.loads(result_path.read_text())
    if not entry.get('run_id') or report.get('run_id') != entry['run_id']:
        raise ValueError(f'latest execution index does not match result: {index}')
    return result_path


def verify_file_hash(path, expected):
    if expected is None:
        return
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'recording checksum mismatch: {path}')


def load_recording(result_path):
    """Load and validate a recorded model and trajectory, including failed runs."""
    path = Path(result_path)
    if path.is_dir():
        path = path / 'result.json'
    report = json.loads(path.read_text())
    trajectory_file = report.get('trajectory_file')
    model_file = report.get('model_file')
    if not trajectory_file or not model_file:
        raise ValueError(f'{path} has no recorded trajectory and model; '
                         'historical summary events cannot be replayed. '
                         'Run ./run_native_reconfiguration.sh to record a new execution.')
    trajectory_path = path.parent / trajectory_file
    model_path = path.parent / model_file
    verify_file_hash(trajectory_path, report.get('trajectory_sha256'))
    verify_file_hash(model_path, report.get('model_sha256'))
    model = mujoco.MjModel.from_binary_path(str(model_path))
    with np.load(trajectory_path, allow_pickle=False) as recorded:
        required = ('time', 'qpos', 'qvel', 'ctrl', 'phase', 'target_interface')
        missing = [name for name in required if name not in recorded]
        if missing:
            raise ValueError(f'{trajectory_path} is missing arrays: {", ".join(missing)}')
        arrays = {name: np.array(recorded[name], copy=True) for name in required}
    time_values = arrays['time']
    if time_values.ndim != 1 or len(time_values) == 0 or not np.isfinite(time_values).all():
        raise ValueError('recorded time must be a nonempty finite 1-D array')
    n = len(time_values)
    if np.any(np.diff(time_values) <= 0):
        raise ValueError('recorded times must be strictly increasing')
    for name, width in (('qpos', model.nq), ('qvel', model.nv), ('ctrl', model.nu)):
        if arrays[name].shape != (n, width) or not np.isfinite(arrays[name]).all():
            raise ValueError(f'invalid {name} array for recorded model')
    if arrays['phase'].shape != (n,) or arrays['target_interface'].shape != (n,):
        raise ValueError('phase and target_interface must have one value per frame')
    if arrays['phase'].dtype.kind not in ('U', 'S'):
        raise ValueError('phase must use a string array, without pickle')
    if not np.isfinite(arrays['target_interface']).all():
        raise ValueError('invalid target interface values')
    return path, report, model, arrays


def set_frame(model, data, arrays, index):
    """Show saved states directly; no simulation steps or controller calls."""
    data.time = float(arrays['time'][index])
    data.qpos[:] = arrays['qpos'][index]
    data.qvel[:] = arrays['qvel'][index]
    data.ctrl[:] = arrays['ctrl'][index]
    mujoco.mj_forward(model, data)


def replay(path, speed=1., loop=False, headless=False):
    path, report, model, arrays = load_recording(path)
    times = arrays['time']
    print(f'REPLAY {path} frames={len(times)} time={times[0]:.3f}..{times[-1]:.3f}s '
          f'original_success={bool(report.get("success"))}', flush=True)
    data = mujoco.MjData(model)
    set_frame(model, data, arrays, 0)
    if headless:
        # Verify all saved frames can be restored and forwarded. This does not
        # rerun the original simulation or certify its dynamics or safety.
        for index in range(1, len(times)):
            set_frame(model, data, arrays, index)
        print(f'REPLAY_END phase={arrays["phase"][-1]} time={data.time:.3f}s '
              f'frames={len(times)}', flush=True)
        return 0

    from mujoco import viewer as mj_viewer
    keys = queue.SimpleQueue()
    def on_key(key):
        if key in (32, 82):  # Space pauses/resumes, R starts again.
            keys.put(key)

    with mj_viewer.launch_passive(model, data, key_callback=on_key,
                                   show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = [0., 0., .08]
        viewer.cam.distance = 1.4
        viewer.cam.elevation = -40
        clock_anchor = time.perf_counter()
        sim_anchor = float(times[0])
        current = 0
        paused = False
        while viewer.is_running():
            now = time.perf_counter()
            while not keys.empty():
                key = keys.get()
                if key == 82:
                    current = 0
                    sim_anchor = float(times[0])
                    clock_anchor = now
                    paused = False
                elif key == 32:
                    if paused:
                        sim_anchor = float(times[current])
                        clock_anchor = now
                        paused = False
                    else:
                        paused = True
            if not paused:
                target = sim_anchor + (now - clock_anchor) * speed
                if target > times[-1] and loop and len(times) > 1:
                    target = float(times[0]) + ((target - float(times[0])) % (float(times[-1]) - float(times[0])))
                    sim_anchor = target
                    clock_anchor = now
                current = min(int(np.searchsorted(times, target, side='right') - 1), len(times) - 1)
                current = max(0, current)
                if target >= times[-1] and not loop:
                    paused = True
            with viewer.lock():
                set_frame(model, data, arrays, current)
            phase = arrays['phase'][current]
            if isinstance(phase, bytes):
                phase = phase.decode('utf-8', errors='replace')
            viewer.set_texts((None, None,
                'Time\nPhase\nInterface\nPlayback',
                f'{times[current]:.3f} / {times[-1]:.3f} s\n{phase}\n'
                f'{int(arrays["target_interface"][current])}\n'
                f'{"paused" if paused else "running"}  Space: pause  R: restart'))
            viewer.sync()
            time.sleep(.01)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path,
                        help='recorded result JSON or its directory (default: latest execution)')
    parser.add_argument('--speed', type=float, default=1., help='playback speed; default 1')
    parser.add_argument('--loop', action='store_true', help='repeat when the last saved frame is reached')
    parser.add_argument('--headless', action='store_true', help='verify frame restore without opening a window')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    try:
        result = args.result if args.result is not None else latest_recording()
        return replay(result, speed=args.speed, loop=args.loop, headless=args.headless)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
