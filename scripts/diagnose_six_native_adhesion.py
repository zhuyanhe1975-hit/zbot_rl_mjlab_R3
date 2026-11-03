"""100 ms standalone six-interface native geom-adhesion diagnostic.

No custom magnetic forces, no weld, MPC, collision safety audit or FSM are
included. 5 N is an assumed per-contact tensile *capacity*, not a force curve
or real magnet calibration. Proxy spheres also participate in compression.
"""
import argparse
import json
import math
import time

import mujoco
import numpy as np

from zbot_mjbatch.native_adhesion import make_native_adhesion_zbot_model


def run(*, duration_s=.1, timestep_s=.001, capacity_n=5., gap_m=.003, solref_s=.004):
    values=(duration_s,timestep_s,capacity_n,gap_m,solref_s)
    if not all(math.isfinite(x) and x>0 for x in values):
        raise ValueError('duration, timestep, capacity, gap and solref must be positive and finite')
    model=make_native_adhesion_zbot_model(contact_force_n=capacity_n,interaction_gap_m=gap_m)
    model.opt.timestep=timestep_s
    model.geom_solref[:,0]=solref_s
    model.actuator_gainprm[:,0]=15.
    model.actuator_biasprm[:,1]=-15.;model.actuator_biasprm[:,2]=-1.5
    data=mujoco.MjData(model)
    for act in range(model.nu):
        joint=int(model.actuator_trnid[act,0])
        if joint<0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('diagnostic requires hinge position actuators')
        data.ctrl[act]=data.qpos[model.jnt_qposadr[joint]]
    mujoco.mj_forward(model,data)
    labels=('proxy','face','pin_wall','pin_backing','ground','other')
    depths={key:0. for key in labels};worst={key:None for key in labels}
    peak_contacts=0;peak_adhesion=0.;peak_actuator_torque=0.;peak_custom_force=0.
    start=time.perf_counter();steps=round(duration_s/timestep_s)
    for _ in range(steps):
        # MuJoCo's native geom adhesion is evaluated by the built-in solver.
        # No qfrc_applied/xfrc_applied custom magnetic model is used.
        mujoco.mj_step(model,data)
        peak_contacts=max(peak_contacts,int(data.ncon))
        peak_adhesion=max(peak_adhesion,float(np.max(np.abs(data.qfrc_adhesion))))
        peak_custom_force=max(peak_custom_force,float(np.max(np.abs(data.qfrc_applied))),
                              float(np.max(np.abs(data.xfrc_applied))))
        peak_actuator_torque=max(peak_actuator_torque,float(np.max(np.abs(data.qfrc_actuator))))
        for contact in data.contact:
            names=[model.geom(contact.geom1).name or '',model.geom(contact.geom2).name or '']
            if 'terrain' in names:category='ground'
            elif any('_adhesion_' in name for name in names):category='proxy'
            elif any('_pin_plate_disc' in name for name in names) and any('_socket_piece_' in name for name in names):category='face'
            elif any('_pin_' in name and name.rsplit('_pin_',1)[-1].isdigit() for name in names) and any('_socket_piece_' in name for name in names):category='pin_wall'
            elif any('_pin_' in name and name.rsplit('_pin_',1)[-1].isdigit() for name in names) and any('_clipped_shell' in name for name in names):category='pin_backing'
            else:category='other'
            if contact.dist<depths[category]:
                depths[category]=float(contact.dist)
                worst[category]={'time_s':float(data.time),'geom1':names[0],'geom2':names[1],
                                 'distance_mm':float(contact.dist)*1000.}
        if np.any(data.warning.number) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break
    return {'requested_s':duration_s,'simulated_s':float(data.time),'timestep_ms':1000*timestep_s,
            'solref_ms':1000*solref_s,'geom_adhesion_capacity_N_per_contact_assumed':capacity_n,
            'adhesion_range_mm_assumed':1000*gap_m,'proxy_radius_mm_assumed':1.,
            'magnet_layout_radius_mm_assumed':35.,'magnet_embed_depth_mm_assumed':1.,
            'magnet_phase_deg_assumed':45.,'kp':15.,'kd':1.5,'peak_contacts':peak_contacts,
            'max_penetration_mm':{key:-value*1000. for key,value in depths.items()},
            'worst_contact':worst,'peak_generalized_adhesion_component':peak_adhesion,
            'peak_custom_applied_component':peak_custom_force,
            'peak_actuator_torque_Nm':peak_actuator_torque,
            'warnings':int(np.sum(data.warning.number)),
            'finite':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            'wall_s':time.perf_counter()-start,
            'caveat':'uncalibrated passive contact adhesion; proxy balls also provide support; no MPC/FSM or safety guard'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-s',type=float,default=.1)
    parser.add_argument('--timestep-ms',type=float,default=1.)
    parser.add_argument('--capacity-n',type=float,default=5.)
    parser.add_argument('--gap-mm',type=float,default=3.)
    parser.add_argument('--solref-ms',type=float,nargs='+',default=[20.,4.])
    args=parser.parse_args()
    print('EXPERIMENT ONLY: native contact adhesion is a per-contact tensile capacity, not a measured magnet force curve.',flush=True)
    for ms in args.solref_ms:
        print(json.dumps(run(duration_s=args.duration_s,timestep_s=args.timestep_ms/1000.,
                             capacity_n=args.capacity_n,gap_m=args.gap_mm/1000.,solref_s=ms/1000.)),flush=True)


if __name__=='__main__':main()
