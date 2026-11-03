"""Local, damped quasi-static tapered-pin loading; NOT a safety proof.

No weld, magnet, robot geometry or controller is included. Equal and opposite
forces are applied to the two free bodies. A diagnostic axial PD and velocity
damping approximately hold the requested disc gap; no pose is overwritten.
"""
import argparse
import json

import mujoco
import numpy as np

from zbot_mjbatch.taper_pin_prototype import tapered_pin_model


def run(gap_m, timestep, load_N=.2, duration_s=.3):
    if (not np.isfinite([gap_m,timestep,load_N,duration_s]).all()
            or gap_m < 0 or timestep <= 0 or load_N < 0 or duration_s <= 0):
        raise ValueError('gap/load must be nonnegative, timestep/duration positive')
    model=tapered_pin_model();model.opt.timestep=timestep;data=mujoco.MjData(model)
    socket=model.body('socket').id;pinbody=model.body('pin_plate').id
    js=model.body_jntadr[socket]; jp=model.body_jntadr[pinbody]
    qs,qp=model.jnt_qposadr[js],model.jnt_qposadr[jp]
    vs,vp=model.jnt_dofadr[js],model.jnt_dofadr[jp]
    data.qpos[qp+2]=gap_m;mujoco.mj_forward(model,data)
    initial_contacts=int(data.ncon)
    count=0;peak_count=0;deepest=0.; peak_normal_N=0.; max_x=0.; max_gap_drift=0.
    normal_force=np.empty(6)
    # Half the nominal trial ramps the applied force; the second half holds it.
    for step in range(round(duration_s/timestep)):
        t=step*timestep
        displacement=data.qpos[qp:qp+3]-data.qpos[qs:qs+3]
        velocity=data.qvel[vp:vp+3]-data.qvel[vs:vs+3]
        ramp=min(t/(duration_s/2),1.)
        fx=load_N*ramp-10.*velocity[0]
        fy=-10.*velocity[1]
        fz=500.*(gap_m-displacement[2])-20.*velocity[2]
        data.qfrc_applied[vs:vs+3]=[-fx,-fy,-fz]
        data.qfrc_applied[vp:vp+3]=[fx,fy,fz]
        # Artificial damping prevents unbounded diagnostic rotations; this is
        # not intended to reproduce the physical fixture or magnet forces.
        data.qfrc_applied[vs+3:vs+6]=-0.05*data.qvel[vs+3:vs+6]
        data.qfrc_applied[vp+3:vp+6]=-0.05*data.qvel[vp+3:vp+6]
        mujoco.mj_step(model,data)
        max_x=max(max_x,abs(float(displacement[0])))
        max_gap_drift=max(max_gap_drift,abs(float(displacement[2]-gap_m)))
        active=[];normal_total=0.
        for i in range(data.ncon):
            contact=data.contact[i]
            a=model.geom(contact.geom1).name or '';b=model.geom(contact.geom2).name or ''
            if ((a.startswith('pin_') and b.startswith('socket_piece_')) or
                (b.startswith('pin_') and a.startswith('socket_piece_'))):
                active.append(contact)
                deepest=min(deepest,float(contact.dist))
                if contact.efc_address >= 0:
                    mujoco.mj_contactForce(model,data,i,normal_force)
                    normal_total+=max(0.,float(normal_force[0]))
        count+=bool(active);peak_count=max(peak_count,len(active))
        peak_normal_N=max(peak_normal_N,normal_total)
    relative=data.qpos[qp:qp+3]-data.qpos[qs:qs+3]
    return {'gap_mm':gap_m*1000,'timestep_ms':timestep*1000,'load_N':load_N,
            'duration_s':duration_s,'initial_contacts':initial_contacts,
            'contact_steps':count,'peak_pin_wall_contacts':peak_count,
            'max_pin_wall_penetration_mm':-deepest*1000,
            'peak_sum_pin_wall_normal_force_N':peak_normal_N,
            'max_abs_relative_x_mm':max_x*1000,'max_gap_drift_mm':max_gap_drift*1000,
            'final_relative_xyz_mm':(relative*1000).tolist(),
            'warnings':int(sum(data.warning.number)),
            'finite':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--load-N',type=float,default=.2)
    p.add_argument('--duration-s',type=float,default=.3)
    args=p.parse_args()
    print('EXPERIMENT: axial PD and damping; fixed assumed dimensions; no magnetic forces')
    for gap in (.0015,.001,.0005):
        for timestep in (.001,.0005,.00025):
            print(json.dumps(run(gap,timestep,args.load_N,args.duration_s)),flush=True)


if __name__=='__main__':main()
