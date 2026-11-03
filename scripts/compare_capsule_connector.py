"""Experimental comparison of 4 mm pins projecting 2 mm from a disk.

All dimensions are assumptions. No magnetic forces or robot collision audit.
"""
import argparse
import math
from pathlib import Path
import statistics
import time

import mujoco
import numpy as np

from zbot_mjbatch.capsule_pin_prototype import pin_shape_spec, pin_shape_model


def benchmark(model, z, x, force, n, repeats):
    durations, peaks, depths = [], [], []
    warnings = 0
    body = model.body('pin_plate').id
    joint = model.body_jntadr[body]
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    for _ in range(repeats):
        data = mujoco.MjData(model)
        data.qpos[q:q+3] = [x, 0., z]
        data.qfrc_applied[v+2] = -force
        mujoco.mj_forward(model, data)
        peak, deepest = data.ncon, 0.
        start = time.perf_counter()
        for _ in range(n):
            mujoco.mj_step(model, data)
            peak = max(peak, data.ncon)
            if data.ncon:
                deepest = min(deepest, float(np.min(data.contact.dist)))
        durations.append((time.perf_counter() - start) / n * 1e6)
        peaks.append(peak); depths.append(-deepest * 1000)
        warnings += int(np.sum(data.warning.number))
    return statistics.median(durations), max(peaks), max(depths), warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', choices=['cylinder', 'sphere', 'capsule'], default='capsule')
    p.add_argument('--output', type=Path)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--gap-mm', type=float, default=20.)
    p.add_argument('--capsule-half-mm', type=float, default=.1)
    args = p.parse_args()
    if not args.output and not args.benchmark:
        p.error('provide --output or --benchmark')
    if args.steps < 1 or args.repeats < 1 or not math.isfinite(args.gap_mm) or args.gap_mm < 0:
        p.error('steps and repeats must be positive; gap must be nonnegative')
    half = args.capsule_half_mm / 1000.
    if args.output:
        spec = pin_shape_spec(args.shape, half)
        spec.body('pin_plate').pos = (0, 0, args.gap_mm / 1000.)
        spec.compile()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(spec.to_xml())
        suffix = (f' Capsule rear overhang is {2*args.capsule_half_mm:g} mm.'
                  if args.shape == 'capsule' else '')
        print(f'Wrote {args.output}, shape={args.shape}, relative protrusion 2 mm.{suffix}')
    if args.benchmark:
        print('EXPERIMENT: no magnets; capsule back overhang is 0.2 mm at default half-length.')
        for shape in ('cylinder', 'sphere', 'capsule'):
            model = pin_shape_model(shape, half)
            for name, z, x, force in [('press', 0., 0., 1.), ('free', .010, 0., 0.),
                                      ('side_initial_overlap', .0005, .0003, 0.)]:
                us, ncon, depth, warnings = benchmark(model, z, x, force, args.steps, args.repeats)
                print(f'{shape} {name} us_per_step={us:.1f} peak_contacts={ncon} '
                      f'max_depth_mm={depth:.4f} warnings={warnings}', flush=True)


if __name__ == '__main__':
    main()
