"""Isolated four-pin frustum study; no magnet or robot collision semantics."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from scripts.compare_capsule_connector import benchmark
from zbot_mjbatch.capsule_pin_prototype import pin_shape_model
from zbot_mjbatch.taper_pin_prototype import tapered_pin_model, tapered_pin_spec


def _distance_report(model, data, first, targets):
    vals=[]; invalid=[]
    for target in targets:
        fromto=np.zeros(6)
        value=float(mujoco.mj_geomDistance(model,data,first,target,.01,fromto))
        if value < .01:
            segment=float(np.linalg.norm(fromto[:3]-fromto[3:]))
            # For disjoint geometries the returned pair should be separated by
            # exactly the reported positive distance. On MuJoCo 3.11, this
            # convex mesh case can report zero with a nonzero segment.
            if value >= 0. and abs(value-segment)>1e-7:
                invalid.append(model.geom(target).name)
        vals.append(value)
    return {'raw_min_mm':round(min(vals)*1000,4), 'unreliable':bool(invalid),
            'inconsistent_targets':invalid}


def clearance(model, gap_m):
    data=mujoco.MjData(model)
    body=model.body('pin_plate').id; joint=model.body_jntadr[body]
    data.qpos[model.jnt_qposadr[joint]+2]=gap_m
    mujoco.mj_forward(model,data)
    sockets=[i for i in range(model.ngeom) if model.geom(i).name.startswith('socket_piece_')]
    return [_distance_report(model,data,model.geom(f'pin_{j}').id,sockets) for j in range(4)]


def backing_distances(gap_m, tip_radius_m):
    spec=tapered_pin_spec(tip_radius_m=tip_radius_m)
    # Isolated ideal backing surface at z=-2mm. This is a diagnostic fixture,
    # not a claim about existing robot shell shape or pocket depth.
    spec.body('socket').add_geom(name='test_backing',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(0.,0.,-.00205),size=(.049,.049,.00005),mass=0.)
    model=spec.compile(); data=mujoco.MjData(model)
    body=model.body('pin_plate').id; joint=model.body_jntadr[body]
    data.qpos[model.jnt_qposadr[joint]+2]=gap_m
    mujoco.mj_forward(model,data)
    return [_distance_report(model,data,model.geom(f'pin_{j}').id,[model.geom('test_backing').id])
            for j in range(4)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tip-diameter-mm',type=float,default=3.6,help='assumed tip diameter, not measured')
    p.add_argument('--output',type=Path)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--gap-mm',type=float,default=20.)
    args=p.parse_args()
    if args.steps<1 or args.repeats<1 or not np.isfinite(args.gap_mm) or args.gap_mm<0:
        p.error('steps/repeats must be positive and gap must be nonnegative')
    tip=args.tip_diameter_mm/2000.
    if args.output:
        spec=tapered_pin_spec(tip_radius_m=tip)
        spec.body('pin_plate').pos=(0.,0.,args.gap_mm/1000.)
        spec.compile();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(spec.to_xml())
        print(f'{args.output}: built-in 16-sided frustum, 4mm base, {args.tip_diameter_mm:g}mm tip, 2mm high. ASSUMED DIMENSIONS.')
    if args.benchmark:
        for shape in ('cylinder','sphere','capsule','taper'):
            model=tapered_pin_model(tip_radius_m=tip) if shape=='taper' else pin_shape_model(shape)
            for name,z,x,force in [('press',0.,0.,1.),('free',.010,0.,0.),
                                   ('side_initial_overlap',.0005,.0003,0.)]:
                us,ncon,depth,warnings=benchmark(model,z,x,force,args.steps,args.repeats)
                print(f'{shape} {name} geom={model.ngeom} us_per_step={us:.1f} '
                      f'peak_contacts={ncon} max_depth_mm={depth:.4f} warnings={warnings}',flush=True)
        for gap in (1.8,1.5,1.,.5,0.):
            distances=clearance(tapered_pin_model(tip_radius_m=tip),gap/1000.)
            back=backing_distances(gap/1000.,tip)
            print(f'taper gap_mm={gap:g} side_query={distances} ideal_backing_query={back}',flush=True)
        print('WARNING: queries marked unreliable have inconsistent nearest-point results; '
              'they are not a clearance or safety guarantee.',flush=True)


if __name__=='__main__':
    main()
