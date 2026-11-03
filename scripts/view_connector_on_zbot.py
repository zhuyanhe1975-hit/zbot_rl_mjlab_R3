"""Visual-only inspection of an assumed connector on the existing Zbot assets.

This is a static preview: one module is displaced for visibility, and the
original collision hulls, exclusions and welds are NOT adjusted for physics.
Do not treat this model as a working replacement for the reconfiguration FSM.
"""
import argparse
import math
import time

import mujoco
import numpy as np
from mjlab.scene import Scene

from zbot_mjbatch.connector_geometry import ConnectorGeometry, inject_connector_collision
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu import add_full_cpu_connectors
from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg


def make_preview(face, geometry):
    cfg = reference_env_cfg().scene
    def decorate(spec):
        add_full_cpu_connectors(spec)
        inlet = spec.site(f'module_{(face+1)%6}/port_in')
        if inlet is None or not np.allclose(inlet.quat, [1, 0, 0, 0], atol=1e-9):
            raise ValueError('preview needs original identity-oriented inlet site')
        # The existing project uses this CAD mounting-origin correction in
        # docking_sites; this is not a measurement of actual magnet/pin sites.
        offset = -np.asarray(inlet.pos, dtype=float)
        inject_connector_collision(spec, f'module_{face}/port_out',
            f'module_{(face+1)%6}/port_in', geometry,
            pin_center_site=offset, socket_center_site=offset, prefix='preview')
        spec.worldbody.add_light(pos=(0, -1, 2), dir=(0, 0, -1), diffuse=(.8, .8, .8))
        spec.worldbody.add_light(pos=(0, 1, -1), dir=(0, 0, 1), diffuse=(.6, .6, .6))
    cfg.spec_fn = decorate
    scene = Scene(cfg, device='cpu')
    model = scene.compile()
    # This is a visual styling change only; no simulation steps are run.
    for i in range(model.ngeom):
        name = model.geom(i).name or ''
        if not name.startswith('preview_'):
            body_name = model.body(model.geom_bodyid[i]).name or ''
            is_neighbour = body_name.startswith((f'module_{face}/', f'module_{(face+1)%6}/'))
            model.geom_rgba[i, 3] = .15 if is_neighbour else 0.
    return model


def main():
    p = argparse.ArgumentParser(description='Static visual inspection of hypothetical connector on Zbot assets')
    p.add_argument('--face', type=int, choices=range(1, 7), default=1)
    p.add_argument('--gap-mm', type=float, default=20.)
    p.add_argument('--pin-radius-mm', type=float, default=2.)
    p.add_argument('--bolt-radius-mm', type=float, default=35.)
    p.add_argument('--clearance-mm', type=float, default=.1)
    p.add_argument('--headless', action='store_true')
    args = p.parse_args()
    if not math.isfinite(args.gap_mm) or args.gap_mm < 0:
        p.error('--gap-mm must be finite and nonnegative')
    face = args.face-1
    geom = ConnectorGeometry(args.pin_radius_mm/1000, args.bolt_radius_mm/1000, args.clearance_mm/1000)
    m = make_preview(face, geom)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    inlet = m.site(f'module_{(face+1)%6}/port_in').id
    out = m.site(f'module_{face}/port_out').id
    initial_gap = np.linalg.norm(d.site_xpos[out]-d.site_xpos[inlet])
    if initial_gap > 1e-5:
        raise ValueError('selected sites are not aligned in the reference scene')
    axis = d.site_xmat[inlet].reshape(3, 3)[:, 2]
    target_joint = m.joint(f'module_{(face+1)%6}/module_root').id
    qadr = int(m.jnt_qposadr[target_joint])
    d.qpos[qadr:qadr+3] -= axis * args.gap_mm / 1000
    mujoco.mj_forward(m, d)
    print('VISUAL ONLY: one module displaced; no physics is simulated; existing convex '
          'hulls/exclusions remain. Mounting center comes from existing project CAD '
          'offset; pin/hole dimensions are uncalibrated.', flush=True)
    print(f'face={args.face} pin geom={m.geom("preview_pin_0").id} '
          f'gap_mm={args.gap_mm} ngeom={m.ngeom}', flush=True)
    if args.headless:
        return
    from mujoco import viewer
    with viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (d.site_xpos[inlet] + d.site_xpos[out]) / 2
        v.cam.distance = .35
        v.cam.azimuth = 135
        v.cam.elevation = -25
        while v.is_running():
            v.sync()
            time.sleep(.02)


if __name__ == '__main__':
    main()
