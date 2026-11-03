"""Research-only A/B for a four-hole SDF; not a safe collision model.

Run with ``./scripts/python.sh -m scripts.bench_connector_sdf``.  Compile a
private plugin in a temporary directory using the existing shared MuJoCo headers
and library. No package/environment changes. Fixed 100mm disc, 4.2mm holes,
4mm pins, 35mm pitch radius and 2mm disc/pin thickness are assumed dimensions.
The CSG field is not an exact SDF at intersections. The current experiment
underestimates penetration and returns the query cutoff for positive gaps, so it
must not be used for MPC clearance, collision acceptance or a robot model.
"""
import argparse
import ctypes
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import mujoco
import numpy as np

from zbot_mjbatch.connector_geometry import ConnectorGeometry, make_standalone_model
from zbot_mjbatch.full_square_connector import FullSquareConnector, make_full_square_model


XML = '''<mujoco>
<extension><plugin plugin="zbot.sdf.experiment"><instance name="disc"/></plugin></extension>
<asset><mesh name="disc"><plugin instance="disc"/></mesh></asset>
<option timestep="0.001" gravity="0 0 0"/>
<worldbody>
  <body name="socket"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia=".001 .001 .001"/>
    <geom name="socket" type="sdf" mesh="disc"><plugin instance="disc"/></geom>
  </body>
  <body name="pin_plate"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia=".001 .001 .001"/>
    <geom name="pin_plate_disc" type="cylinder" size=".05 .001" pos="0 0 .001"/>
    <geom name="pin_0" type="cylinder" size=".002 .001" pos=".035 0 -.001"/>
    <geom name="pin_1" type="cylinder" size=".002 .001" pos="0 .035 -.001"/>
    <geom name="pin_2" type="cylinder" size=".002 .001" pos="-.035 0 -.001"/>
    <geom name="pin_3" type="cylinder" size=".002 .001" pos="0 -.035 -.001"/>
  </body>
</worldbody></mujoco>'''


def compile_and_load(directory):
    root = Path(mujoco.__file__).resolve().parent
    libraries = list(root.glob('libmujoco.so.*'))
    if len(libraries) != 1:
        raise RuntimeError(f'Expected one MuJoCo shared library, found: {libraries}')
    source = Path(__file__).parent / 'experimental_sdf' / 'connector_sdf.cc'
    target = Path(directory) / 'connector_sdf.so'
    subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(source),
                    f'-I{root / "include"}', f'-L{root}', f'-Wl,-rpath,{root}',
                    f'-l:{libraries[0].name}', '-o', str(target)], check=True)
    library = ctypes.CDLL(str(target))
    if library.register_zbot_sdf() < 0:
        raise RuntimeError('MuJoCo plugin registration failed')
    return library


def measure(model, mode, steps, repeats):
    values = []
    peak = warnings = 0
    depth = 0.
    for _ in range(repeats):
        data = mujoco.MjData(model)
        if mode == 'away':
            data.qpos[9] = .010
        if mode == 'side':
            data.qpos[7] = .0003
            data.qpos[9] = .0005
        if mode == 'press':
            data.qfrc_applied[8] = -1.
        mujoco.mj_forward(model, data)
        start = time.perf_counter()
        for _ in range(steps):
            mujoco.mj_step(model, data)
            peak = max(peak, data.ncon)
            if data.ncon:
                depth = min(depth, float(np.min(data.contact.dist)))
        values.append((time.perf_counter() - start) * 1e6 / steps)
        warnings = max(warnings, int(np.sum(data.warning.number)))
    return statistics.median(values), peak, -1000 * depth, warnings


def probe(model):
    print('SDF GAP PROBE: actual aligned gap = 0.1 mm, 0.3 mm offset overlaps by 0.2 mm')
    for x in (0., .00005, .0002, .0003):
        data = mujoco.MjData(model)
        data.qpos[7] = x
        data.qpos[9] = .0005
        mujoco.mj_forward(model, data)
        rows = []
        for i in range(4):
            pin = model.geom(f'pin_{i}').id
            dist = mujoco.mj_geomDistance(model, data, pin, model.geom('socket').id, .01, None)
            contacts = [c.dist for c in data.contact if c.geom1 == pin or c.geom2 == pin]
            rows.append((round(dist*1000, 4), round(min(contacts)*1000, 4) if contacts else None))
        print(f'offset_mm={x*1000:g} per_pin=(distance_mm, minimum_contact_mm) {rows}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('steps and repeats must be positive')
    with tempfile.TemporaryDirectory(prefix='zbot_sdf_bench_') as directory:
        library = compile_and_load(directory)  # Keep the shared library loaded.
        model = mujoco.MjModel.from_xml_string(XML)
        print('EXPERIMENT ONLY: no collision-safety or robot-model validity')
        print(f'MuJoCo {mujoco.__version__}; SDF starts={model.opt.sdf_initpoints}, '
              f'iterations={model.opt.sdf_iterations}; steps={args.steps}, repeats={args.repeats}')
        models = [('round16', make_standalone_model(ConnectorGeometry(.002, .035, .0001))),
                  ('poly8', make_full_square_model(FullSquareConnector(hole_segments=8))),
                  ('poly4', make_full_square_model(FullSquareConnector(hole_segments=4))),
                  ('sdf', model)]
        for name, candidate in models:
            for mode in ('press', 'away', 'side'):
                us, contacts, depth, warnings = measure(candidate, mode, args.steps, args.repeats)
                print(f'{name} {mode} geom={candidate.ngeom} us_per_step={us:.1f} '
                      f'peak_contacts={contacts} max_depth_mm={depth:.4f} warnings={warnings}', flush=True)
        probe(model)
        _ = library


if __name__ == '__main__':
    main()
