"""Inspect the isolated six-module square-hole collision geometry.

Static exploded view only: modules are displaced for inspection. There are no
magnets, welds, state machine, simulation steps, or validated safety filtering.
"""
import argparse
import time

import mujoco
import numpy as np

from zbot_mjbatch.physical_zbot_geometry import make_physical_zbot_model


def make_view(face, gap_m, *, pin_shape='taper', pin_tip_radius_m=.0018, pin_sides=16):
    if not 0<=face<6 or not np.isfinite(gap_m) or gap_m<0:
        raise ValueError('face must be in 0..5 and gap must be nonnegative')
    model=make_physical_zbot_model(pin_shape=pin_shape, pin_tip_radius_m=pin_tip_radius_m,
                                   pin_sides=pin_sides)
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    source=face;target=(face+1)%6
    port=model.site(f'module_{target}/port_in').id
    axis=data.site_xmat[port].reshape(3,3)[:,2].copy()
    for i,sign in ((source,-1.),(target,1.)):
        adr=int(model.jnt_qposadr[model.joint(f'module_{i}/module_root').id])
        data.qpos[adr:adr+3]+=sign*axis*gap_m/2
    mujoco.mj_forward(model,data)
    for i in range(model.ngeom):
        name=model.geom(i).name or ''
        parent=model.body(model.geom_bodyid[i]).name or ''
        is_selected=parent.startswith((f'module_{source}/',f'module_{target}/'))
        if name.startswith('physical_connector_'):
            if not name.startswith(f'physical_connector_{face}_'):
                model.geom_rgba[i,3]=0.
        elif name.endswith('_clipped_shell'):
            model.geom_rgba[i]=(.3,.9,.5,.18 if is_selected else .03)
        elif parent.startswith('module_'):
            model.geom_rgba[i,3]=.12 if is_selected else .03
    return model,data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face',type=int,choices=range(1,7),default=1)
    parser.add_argument('--gap-mm',type=float,default=20.)
    parser.add_argument('--pin-shape',choices=('taper','cylinder'),default='taper')
    parser.add_argument('--tip-diameter-mm',type=float,default=3.6,
                        help='assumed tapered pin tip diameter; not measured')
    parser.add_argument('--pin-sides',type=int,default=16,
                        help='taper polygon resolution; multiple of four')
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args()
    model,data=make_view(args.face-1,args.gap_mm/1000.,pin_shape=args.pin_shape,
                         pin_tip_radius_m=args.tip_diameter_mm/2000.,pin_sides=args.pin_sides)
    a=model.site(f'module_{args.face-1}/port_out').id
    b=model.site(f'module_{args.face%6}/port_in').id
    print(f'STATIC PREVIEW ONLY: face={args.face}, gap={args.gap_mm}mm, '
          f'pin={args.pin_shape}, tip_diameter={args.tip_diameter_mm}mm (assumed), '
          f'geom={model.ngeom}; no magnets, connector constraints, '
          'physical steps or validated collision audit.',flush=True)
    if args.headless:
        return
    from mujoco import viewer
    with viewer.launch_passive(model,data) as vis:
        vis.cam.lookat[:]=(data.site_xpos[a]+data.site_xpos[b])/2
        vis.cam.distance=.35;vis.cam.azimuth=135;vis.cam.elevation=-25
        while vis.is_running():
            vis.sync();time.sleep(.02)


if __name__=='__main__':
    main()
