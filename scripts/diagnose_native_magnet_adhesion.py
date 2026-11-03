"""Isolated MuJoCo passive-adhesion diagnostic, not a calibrated magnet.

Four hypothetical paired magnet proxies are modeled as tiny spheres whose
centers lie inside the two facing plates. Only proxy-to-proxy collisions are
allowed by bit masks; the physical connector collision shapes remain intact.
MuJoCo geom adhesion is a per-contact tensile capacity, not a permanent-magnet
field law. It is constant within a finite contact gap and cannot distinguish
magnetic polarity. The default dimensions and 5 N are assumed, not measured.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import mujoco
import numpy as np

from zbot_mjbatch.taper_pin_prototype import tapered_pin_spec


def make_model(force_n=5., gap_m=.003, magnet_radius_m=.001, layout_radius_m=.035,
               timestep=.001, contact_timeconst=.004):
    values=(force_n,gap_m,magnet_radius_m,layout_radius_m,timestep,contact_timeconst)
    if not all(math.isfinite(v) and v>0 for v in values) or magnet_radius_m>.002:
        raise ValueError('expected positive finite parameters and magnet radius <= plate thickness')
    spec=tapered_pin_spec()
    for side, z, contype, affinity in (('pin_plate',magnet_radius_m,4,8),
                                      ('socket',-magnet_radius_m,8,4)):
        for i in range(4):
            theta=math.pi/4 + math.pi*i/2
            spec.body(side).add_geom(name=f'adhesion_proxy_{side}_{i}',
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=(layout_radius_m*math.cos(theta), layout_radius_m*math.sin(theta), z),
                size=(magnet_radius_m,0.,0.),mass=0.,condim=1,
                contype=contype,conaffinity=affinity,
                # Exactly one side contributes; both would double the pull.
                adhesion=force_n if side=='pin_plate' else 0.,
                gap=gap_m if side=='pin_plate' else 0.,
                rgba=(1.,.4,.1,.25))
    m=spec.compile();m.opt.timestep=timestep;m.geom_solref[:,0]=contact_timeconst
    return m


def probe(model, gap_m, yaw=0., sideways_m=0.):
    d=mujoco.MjData(model)
    a,b=model.body('socket').id,model.body('pin_plate').id
    q1=int(model.jnt_qposadr[model.body_jntadr[a]])
    q2=int(model.jnt_qposadr[model.body_jntadr[b]])
    d.qpos[q2:q2+3]=[sideways_m,0.,gap_m]
    d.qpos[q2+3:q2+7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    mujoco.mj_forward(model,d)
    proxies=[];unexpected=[]
    for c in d.contact:
        n1=model.geom(c.geom1).name or '';n2=model.geom(c.geom2).name or ''
        if n1.startswith('adhesion_proxy_') or n2.startswith('adhesion_proxy_'):
            if n1.startswith('adhesion_proxy_') and n2.startswith('adhesion_proxy_'):
                proxies.append([n1,n2,float(c.dist)])
            else:unexpected.append([n1,n2,float(c.dist)])
    u=int(model.jnt_dofadr[model.body_jntadr[a]])
    v=int(model.jnt_dofadr[model.body_jntadr[b]])
    return {'gap_mm':gap_m*1000,'yaw_deg':math.degrees(yaw),'sideways_mm':sideways_m*1000,
            'proxy_contacts':len(proxies),'proxy_pairs':proxies,
            'unexpected_proxy_contacts':unexpected,
            'source_adhesion_force_N':d.qfrc_adhesion[u:u+3].tolist(),
            'target_adhesion_force_N':d.qfrc_adhesion[v:v+3].tolist(),
            'ncon_total':int(d.ncon)}


def run(timestep=.001, duration=.1, force_n=5., gap_m=.003, start_gap_m=.002,
        contact_timeconst=.004):
    m=make_model(force_n,gap_m,timestep=timestep,contact_timeconst=contact_timeconst)
    d=mujoco.MjData(m)
    a,b=m.body('socket').id,m.body('pin_plate').id
    q1=int(m.jnt_qposadr[m.body_jntadr[a]]);q2=int(m.jnt_qposadr[m.body_jntadr[b]])
    d.qpos[q2+2]=start_gap_m;mujoco.mj_forward(m,d)
    npeak=0;deepest=0.;first=None;t0=time.perf_counter()
    for k in range(round(duration/timestep)):
        mujoco.mj_step(m,d)
        npeak=max(npeak,int(d.ncon))
        for c in d.contact:
            n1=m.geom(c.geom1).name or '';n2=m.geom(c.geom2).name or ''
            if not n1.startswith('adhesion_proxy_') and not n2.startswith('adhesion_proxy_'):
                if c.dist<0 and first is None:first=(k+1)*timestep
                deepest=min(deepest,float(c.dist))
    r=np.empty(9);mujoco.mju_quat2Mat(r,d.qpos[q1+3:q1+7])
    local_gap=float((r.reshape(3,3).T @ (d.qpos[q2:q2+3]-d.qpos[q1:q1+3]))[2])
    return {'timestep_ms':timestep*1000,'duration_s':float(d.time),'local_gap_mm':local_gap*1000,
            'physical_penetration_mm':-deepest*1000,'first_physical_contact_s':first,
            'peak_contacts':npeak,'warnings':int(sum(d.warning.number)),
            'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            'wall_s':time.perf_counter()-t0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--force-N',type=float,default=5.)
    p.add_argument('--gap-mm',type=float,default=3.)
    p.add_argument('--duration-s',type=float,default=.1)
    a=p.parse_args()
    model=make_model(a.force_N,a.gap_mm/1000)
    print('EXPERIMENT: passive per-contact adhesion, assumed proxy geometry, not a magnetic field',flush=True)
    for gap in (.004,.002,.0005,0.):
        print('PROBE',json.dumps(probe(model,gap)),flush=True)
    print('QUARTER',json.dumps(probe(model,.0005,math.pi/2)),flush=True)
    print('OFF_AXIS',json.dumps(probe(model,.0005,0.,.006)),flush=True)
    for dt in (.001,.0005,.00025):
        print('RUN',json.dumps(run(dt,a.duration_s,a.force_N,a.gap_mm/1000)),flush=True)


if __name__=='__main__':
    main()
