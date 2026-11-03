"""Static exploded preview of a connector from the current native model.

This opens a separate MuJoCo model and moves two bodies for inspection. It
does not run dynamics, simulate magnetic attraction or validate a trajectory.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from zbot_mjbatch.native_batch import make_native_batch_model


def make_preview(face: int, gap_m: float):
    """Return a fresh model and static data, with one interface opened.

    Use the model's closed reference pose so interface 6, which is open in the
    actual initial straight configuration, can be inspected just like 1--5.
    Body positions and display settings below affect this preview only.
    """
    if face not in range(6) or not np.isfinite(gap_m) or gap_m < 0:
        raise ValueError('face must be 0..5 and gap must be finite and nonnegative')
    model, _ = make_native_batch_model()
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    source, target = face, (face + 1) % 6
    port = model.site(f'module_{target}/port_in').id
    normal = data.site_xmat[port].reshape(3, 3)[:, 2].copy()
    for module, direction in ((source, -1.), (target, 1.)):
        address = int(model.joint(f'module_{module}/module_root').qposadr[0])
        data.qpos[address:address + 3] += direction * normal * gap_m / 2.
    mujoco.mj_forward(model, data)

    # Retain the actual connector geoms while making the two adjoining module
    # shells translucent; hide unrelated modules and all other interfaces.
    prefix = f'physical_connector_{face}_'
    model.vis.headlight.ambient[:] = (.65, .65, .65)
    model.vis.headlight.diffuse[:] = (.8, .8, .8)
    # Sites on the other eleven connector sides are not part of this preview.
    model.site_rgba[:, 3] = 0.
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ''
        body = model.body(model.geom_bodyid[geom]).name or ''
        if name.startswith('physical_connector_'):
            if not name.startswith(prefix):
                model.geom_rgba[geom, 3] = 0.
            elif 'adhesion_' in name:
                # These spheres are numerical magnetic proxies, not the size
                # of real magnets; vivid color makes their locations visible.
                model.geom_rgba[geom, 3] = 1.
        elif body.startswith((f'module_{source}/', f'module_{target}/')):
            model.geom_matid[geom] = -1
            model.geom_rgba[geom] = (.35, .7, .45, .025)
        else:
            model.geom_rgba[geom, 3] = 0.

    pin_disc = model.geom(f'{prefix}pin_plate_disc').id
    target_magnets = [model.site(f'{prefix}magnet_target_{i}').id for i in range(4)]
    camera_center = (data.geom_xpos[pin_disc] + data.site_xpos[target_magnets].mean(axis=0)) / 2.
    return model, data, camera_center


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face', type=int, choices=range(1, 7), default=1,
                        help='interface number (1..6)')
    parser.add_argument('--gap-mm', type=float, default=20.,
                        help='added visual separation in mm (default: 20)')
    parser.add_argument('--headless', action='store_true', help='build and check without opening GUI')
    args = parser.parse_args()
    try:
        model, data, center = make_preview(args.face - 1, args.gap_mm / 1000.)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'STATIC CONNECTOR PREVIEW face={args.face} added_gap_mm={args.gap_mm:g}; '
          'current native model; magnet spheres are visualization proxies; '
          'no simulation steps or dynamics validation', flush=True)
    if args.headless:
        return 0
    from mujoco import viewer
    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as vis:
        vis.cam.lookat[:] = center
        vis.cam.distance = .25
        vis.cam.azimuth = 60
        vis.cam.elevation = -25
        while vis.is_running():
            vis.sync()
            time.sleep(.02)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
