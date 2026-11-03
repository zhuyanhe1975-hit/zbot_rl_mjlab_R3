"""Independent six-module magnet/contact diagnostic; not a safe controller.

This scene has no weld, no MPC or collision safety guard. The magnet geometry
and per-pair force are illustrative and uncalibrated. The script applies the
magnetic model on every MuJoCo substep, with actual joint position servo gains
Kp=15, Kd=1.5 and each servo initially holding its current joint angle.
"""
import argparse
import json
import math
import time

import mujoco
import numpy as np

from zbot_mjbatch.four_magnet_force import FourMagnetParameters
from zbot_mjbatch.physical_zbot_geometry import make_physical_zbot_model
from zbot_mjbatch.six_interface_magnets import SixInterfaceMagnetForces


def run(duration=.1, timestep=.001, force_n=5., contact_timeconst=.02):
    vals=[duration,timestep,force_n,contact_timeconst]
    if not np.isfinite(vals).all() or any(v<=0 for v in vals):
        raise ValueError('all diagnostic values must be positive and finite')
    m=make_physical_zbot_model();m.opt.timestep=timestep
    m.geom_solref[:,0]=contact_timeconst
    d=mujoco.MjData(m)
    # Hold initial joint positions using native position servo law; no controls
    # are computed for the magnet itself and no qpos/qvel reset is performed.
    m.actuator_gainprm[:,0]=15.
    m.actuator_biasprm[:,1]=-15.;m.actuator_biasprm[:,2]=-1.5
    for act in range(m.nu):
        joint=int(m.actuator_trnid[act,0])
        if joint<0 or m.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('diagnostic requires hinge position actuators')
        d.ctrl[act]=d.qpos[m.jnt_qposadr[joint]]
    mujoco.mj_forward(m,d)
    p=FourMagnetParameters(force_n, .002, .001, .010, .020)
    magnet=SixInterfaceMagnetForces(m,d,[p]*6)
    wrench=np.empty(m.nv)
    deepest=0.;peak_contacts=0;peak_torque=0.;worst=None; peak_speed=0.
    category_depth={name:0. for name in ('ground','pin_wall','pin_backing','face','other_inter_module')}
    category_worst={name:None for name in category_depth}
    start=time.perf_counter()
    steps=round(duration/timestep)
    for i in range(steps):
        # Update each site's transform after the previous integration. The
        # array is rebuilt each step so no prior magnetic force accumulates.
        mujoco.mj_fwdPosition(m,d)
        magnet.compute(out=wrench)
        d.qfrc_applied[:]=wrench
        mujoco.mj_step(m,d)
        peak_contacts=max(peak_contacts,int(d.ncon))
        peak_torque=max(peak_torque,float(np.max(np.abs(d.qfrc_actuator))))
        peak_speed=max(peak_speed,float(np.max(np.abs(d.qvel))))
        if d.ncon:
            item=min(d.contact,key=lambda c:c.dist)
            if item.dist<deepest:
                deepest=float(item.dist)
                worst={'time_s':float(d.time),'geom1':m.geom(item.geom1).name,
                       'geom2':m.geom(item.geom2).name,'distance_mm':deepest*1000}
            for item in d.contact:
                g1,g2=int(item.geom1),int(item.geom2)
                n1,n2=(m.geom(g).name or '' for g in (g1,g2))
                body1,body2=(m.body(m.geom_bodyid[g]).name or '' for g in (g1,g2))
                pin1='_pin_' in n1 and n1.rsplit('_pin_',1)[-1].isdigit()
                pin2='_pin_' in n2 and n2.rsplit('_pin_',1)[-1].isdigit()
                if n1=='terrain' or n2=='terrain':category='ground'
                elif ((pin1 and '_socket_piece_' in n2) or
                      (pin2 and '_socket_piece_' in n1)):category='pin_wall'
                elif ((pin1 and n2.endswith('_clipped_shell')) or
                      (pin2 and n1.endswith('_clipped_shell'))):category='pin_backing'
                elif (('_pin_plate_disc' in n1 and '_socket_piece_' in n2) or
                      ('_pin_plate_disc' in n2 and '_socket_piece_' in n1)):category='face'
                elif (body1.startswith('module_') and body2.startswith('module_') and
                      body1.split('/')[0] != body2.split('/')[0]):category='other_inter_module'
                else:continue
                if item.dist<category_depth[category]:
                    category_depth[category]=float(item.dist)
                    category_worst[category]={'time_s':float(d.time), 'geom1':n1,'geom2':n2,
                                              'distance_mm':float(item.dist)*1000}
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            break
    return {'duration_s':duration,'simulated_s':float(d.time),'steps':steps,
            'force_N_at_assumed_2mm_reference':force_n,
            'geom_solref_timeconst_ms':contact_timeconst*1000,
            'timestep_ms':timestep*1000,'gravity_m_s2':m.opt.gravity.tolist(),
            'kp':15.,'kd':1.5,'peak_contacts':peak_contacts,
            'peak_penetration_mm':-deepest*1000,'worst_contact':worst,
            'category_peak_penetration_mm':{k:-v*1000 for k,v in category_depth.items()},
            'category_worst':category_worst,
            'peak_actuator_torque_Nm':peak_torque,
            'peak_generalized_speed':peak_speed,
            'warnings':int(np.sum(d.warning.number)),
            'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            'wall_s':time.perf_counter()-start,'num_geoms':m.ngeom,
            'caveat':'uncalibrated magnets, no weld or safety audit; peak generalized speed mixes units'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration-s',type=float,default=.1)
    p.add_argument('--force-N',type=float,default=5.)
    p.add_argument('--timestep-ms',type=float,default=1.)
    p.add_argument('--solref-ms',type=float,nargs='+',default=[20.,4.])
    a=p.parse_args()
    print('EXPERIMENT ONLY: uncalibrated force; no weld or MPC; not a safe completed maneuver.',flush=True)
    for tau_ms in a.solref_ms:
        print(json.dumps(run(a.duration_s,a.timestep_ms/1000,a.force_N,tau_ms/1000)),flush=True)


if __name__=='__main__':main()
