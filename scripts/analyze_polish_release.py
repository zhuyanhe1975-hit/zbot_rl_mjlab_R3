"""Audit one internal-release state using mjbatch native batched rollouts.

This is a diagnostic, not a planned or validated trajectory.
"""
import argparse
import collections
import json
import os
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from scripts.optimize_headtail_mjbatch import lib_open, STATE
from scripts.polish_planned_mjbatch import run_candidate
from scripts.polish_safety import CollisionAudit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-dir', type=Path, default=Path('outputs/reconfiguration/full_polish/20261102_153458_757945'))
    p.add_argument('--candidate', type=int, default=11)
    p.add_argument('--count', type=int, default=512)
    p.add_argument('--threads', type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument('--horizon-ms', type=int, default=80)
    p.add_argument('--offset-rad', type=float, default=.08)
    p.add_argument('--seed', type=int, default=142)
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    if a.count < 2 or a.threads < 1 or a.horizon_ms < 1 or not np.isfinite(a.offset_rad) or a.offset_rad < 0:
        p.error('invalid count, threads, horizon or offset')
    source = a.search_dir.resolve()
    rows = json.loads((source / 'search.json').read_text())['candidates']
    row = next((r for r in rows if r['index'] == a.candidate), None)
    if row is None or 'settings' not in row:
        p.error('candidate not found or its settings were not recorded')
    goal = json.loads((source / f"branch_{row['branch']}" / 'goals.json').read_text())[0]
    library = source / 'libcurl_strict.so'
    if not library.is_file():
        p.error(f'missing source strict native library: {library}')
    out = a.output_dir or Path('outputs/reconfiguration/release_analysis') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    out = out.resolve()
    print('RESULT_DIR', out, flush=True)
    settings = dict(row['settings'])
    settings.update(release_mode='immediate_substep_v1', separate_internal_rotation=True)
    captured = {}
    original_sample = CollisionAudit.sample
    def sample(audit):
        f = audit.f
        if not captured and f.phase == 'SEPARATE' and f.target == 0:
            state = np.empty(mujoco.mj_stateSize(f.model, STATE))
            mujoco.mj_getState(f.model, f.data, state, STATE)
            captured['state'] = state
            captured['q'] = f.data.qpos[f.qadr].copy()
            captured['time_s'] = float(f.data.time)
            mujoco.mj_saveModel(f.model, str(out / 'release_model.mjb'), None)
        original_sample(audit)
    CollisionAudit.sample = sample
    try:
        run_candidate(({'index':0, 'branch':row['branch'], 'settings':settings}, goal,
                       {'kp':15., 'kd':1.5}, row['unfold_controls'], str(out)))
    finally:
        CollisionAudit.sample = original_sample
    if not captured:
        raise SystemExit('Candidate did not reach first internal-release state; inspect candidate_0.log')
    state = captured['state']; q = captured['q']; n = a.count; steps = a.horizon_ms
    rng = np.random.default_rng(a.seed)
    offsets = rng.uniform(-a.offset_rad, a.offset_rad, (n, 6)); offsets[0] = 0.
    controls = np.ascontiguousarray(np.repeat((q + offsets)[:, None, :], steps, axis=1))
    lengths = np.full(n, steps, dtype=np.int32)
    results = np.zeros((n, 13)); endstates = np.zeros((n, len(state))); metrics = np.zeros((n, 4))
    lib = lib_open(library)
    ptr = lib.zbatch_create(os.fsencode(out / 'release_model.mjb'), a.threads)
    if not ptr:
        raise RuntimeError('cannot create native mjbatch pool')
    try:
        lib.zbatch_eval(ptr, state, n, steps, lengths, controls, results, endstates, 80., metrics, 0)
    finally:
        lib.zbatch_free(ptr)
    counts = {str(int(k)):int(v) for k,v in collections.Counter(results[:,0]).items()}
    accepted = np.flatnonzero(results[:,0] == 0)
    report = {'source':str(source), 'candidate':a.candidate, 'captured_time_s':captured['time_s'],
              'count':n, 'threads':a.threads, 'horizon_ms':steps, 'offset_rad':a.offset_rad,
              'status_counts':counts, 'status_codes':{'0':'passed','1':'load','2':'torque','3':'contact depth',
              '4':'nonfinite','5':'constraint','6':'module geometry','7':'joint speed',
              '8':'source joint position range','9':'MuJoCo warning'},
              'passed_indices':accepted.tolist(),
              'scope':'constant position commands from one release state; not a proof of impossibility or a full trajectory'}
    np.savez(out / 'samples.npz', offsets=offsets, results=results, endstates=endstates)
    (out / 'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RELEASE_ANALYSIS',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
