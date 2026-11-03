"""Generate a square-hole preview or benchmark independent connector models.

The square-hole design is an approximation missing most of the outer support
face. Neither model is connected to Zbot or includes magnets.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import mujoco
import numpy as np

from zbot_mjbatch.connector_geometry import ConnectorGeometry, make_standalone_model
from zbot_mjbatch.square_socket_prototype import SquareSocketGeometry, make_square_model, make_square_spec


def run(model, *, steps, repeats, x, z, pressure):
    durations, contacts, penetration = [], [], []
    warnings = 0
    body = model.body('pin_plate').id
    joint = model.body_jntadr[body]
    qadr, dof = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    for _ in range(repeats):
        data = mujoco.MjData(model)
        data.qpos[qadr:qadr+3] = [x, 0., z]
        data.qpos[qadr+3:qadr+7] = [1., 0., 0., 0.]
        data.qfrc_applied[dof+2] = -pressure
        mujoco.mj_forward(model, data)
        peak_contact = 0
        peak_depth = 0.
        start = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(model, data)
            peak_contact = max(peak_contact, data.ncon)
            if data.ncon:
                peak_depth = min(peak_depth, float(np.min(data.contact.dist)))
        durations.append(time.perf_counter()-start)
        contacts.append(peak_contact)
        penetration.append(-peak_depth)
        warnings += int(np.sum(data.warning.number))
    return {'ngeom':model.ngeom, 'median_us_per_step':1e6*statistics.median(durations)/steps,
            'peak_contacts':max(contacts), 'max_penetration_mm':1e3*max(penetration),
            'warnings':warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write square-socket MJCF for inspection')
    parser.add_argument('--initial-gap-mm', type=float, default=20.)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not args.output and not args.benchmark:
        parser.error('provide --output and/or --benchmark')
    if args.steps < 1 or args.repeats < 1 or not np.isfinite(args.initial_gap_mm) or args.initial_gap_mm < 0:
        parser.error('steps and repeats must be positive; initial gap must be nonnegative')
    if args.output:
        spec = make_square_spec()
        spec.body('pin_plate').pos = (0, 0, args.initial_gap_mm/1000.)
        spec.worldbody.add_light(pos=(0, 0, .3), dir=(0, 0, -1), diffuse=(.8, .8, .8))
        spec.worldbody.add_light(pos=(0, 0, -.3), dir=(0, 0, 1), diffuse=(.7, .7, .7))
        for geom in spec.geoms:
            if geom.name == 'pin_plate_disc':
                geom.rgba = (.1, .4, 1., .35)
            elif geom.name.startswith('pin_'):
                geom.rgba = (1., .75, .1, 1.)
            elif geom.name == 'support_disc':
                geom.rgba = (.7, .8, .9, 1.)
            elif geom.name.startswith('socket_wall_'):
                geom.rgba = (.95, .4, .15, 1.)
        spec.compile()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(spec.to_xml())
        print(f'Wrote {args.output}: isolated approximate geometry; no magnets, outer face incomplete.')
    if args.benchmark:
        models = {'round':make_standalone_model(ConnectorGeometry(.002,.035,.0001)),
                  'square':make_square_model()}
        for name, model in models.items():
            for case, (x, z, pressure) in {'free':(0., .003, 0.),
                                           'face_load':(0., 0., 1.),
                                           'pin_wall':(.0003, .0005, 0.)}.items():
                result = run(model, steps=args.steps, repeats=args.repeats, x=x, z=z, pressure=pressure)
                print(json.dumps({'model':name,'scenario':case,'steps':args.steps,
                                  'repeats':args.repeats, **result}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
