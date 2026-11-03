"""Isolated four-magnet / taper-pin dynamics experiments, not safety validation.

The 35 mm magnet radius, 1 mm center setback and 5 N per pair at a 2 mm
center distance are *assumptions*, not measured parameters. No weld, robot
controller, MJBatch MPC or collision auditing is used. The step size is
configurable for sensitivity comparisons.
"""
from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np

from zbot_mjbatch.taper_pin_prototype import tapered_pin_spec


def make_scene(magnet_radius_m=.035, setback_m=.001):
    """Add an assumed symmetric four-magnet layout to the isolated connector.

    Magnets are offset by 45 degrees relative to the four pin centers so that
    they don't occupy the same nominal XY locations. Their centers are 1 mm
    behind each plate surface; neither this layout nor polarity is measured.
    """
    if not np.isfinite([magnet_radius_m, setback_m]).all() or not (0 < magnet_radius_m < .05 and 0 < setback_m <= .002):
        raise ValueError('invalid magnet location')
    spec = tapered_pin_spec()
    for side, z in (('socket', -setback_m), ('pin_plate', setback_m)):
        body = spec.body(side)
        for i in range(4):
            angle = math.pi/4 + i*math.pi/2
            body.add_site(name=f'{side}_magnet_{i}',
                          pos=(magnet_radius_m*math.cos(angle), magnet_radius_m*math.sin(angle), z),
                          size=(.001,)*3)
    return spec.compile()


def run(mode, timestep=.001, duration=.25, initial_gap_m=.003,
        force_n=5., radius_m=.035, setback_m=.001, tilt_nm=.2,
        contact_timeconst=.02, contact_impedance=(.9,.95), initial_closing_speed_m_s=0., trace=False):
    """Run free-body approach, hold, or applied-tilt experiment.

    The documented params are visual/test assumptions, not magnet or hardware
    calibration. ``hold`` and ``tilt`` start at nominal surface contact;
    ``approach`` starts at the specified positive initial gap. A torque in ``tilt`` is
    applied as equal-and-opposite body moments; no reference trajectory, force
    control, positional reset or dissipative fixture is introduced.
    """
    if mode not in ('approach','hold','tilt'):
        raise ValueError('mode must be approach, hold or tilt')
    values = [timestep, duration, initial_gap_m, force_n, radius_m, setback_m, tilt_nm,
              contact_timeconst, initial_closing_speed_m_s]
    if (not np.isfinite(values).all() or timestep <= 0 or duration <= 0 or initial_gap_m < 0
            or force_n <= 0 or tilt_nm < 0 or contact_timeconst <= 0 or initial_closing_speed_m_s < 0):
        raise ValueError('invalid diagnostic parameters')
    if len(contact_impedance)!=2 or not np.isfinite(contact_impedance).all() or not 0<contact_impedance[0]<=contact_impedance[1]<1:
        raise ValueError('contact impedance must be two increasing values in (0, 1)')
    from zbot_mjbatch.four_magnet_force import FourMagnetParameters, four_magnet_forces, magnetic_qfrc
    m = make_scene(radius_m, setback_m)
    m.opt.timestep = timestep
    m.geom_solref[:,0] = contact_timeconst
    m.geom_solimp[:,0:2] = contact_impedance
    d = mujoco.MjData(m)
    a, b = m.body('socket').id, m.body('pin_plate').id
    qa = int(m.jnt_qposadr[m.body_jntadr[a]])
    qb = int(m.jnt_qposadr[m.body_jntadr[b]])
    d.qpos[qb+2] = 0. if mode in ('hold','tilt') else initial_gap_m
    va=int(m.jnt_dofadr[m.body_jntadr[a]]);vb=int(m.jnt_dofadr[m.body_jntadr[b]])
    d.qvel[va+2]=initial_closing_speed_m_s/2
    d.qvel[vb+2]=-initial_closing_speed_m_s/2
    sa = [m.site(f'socket_magnet_{i}').id for i in range(4)]
    sb = [m.site(f'pin_plate_magnet_{i}').id for i in range(4)]
    params = FourMagnetParameters(contact_force_n=force_n, reference_distance_m=2*setback_m,
                                  softening_m=.001, switch_start_m=.010, cutoff_m=.020)
    mujoco.mj_forward(m, d)
    peak_penetration = 0.; peak_contacts = 0; first_contact = None
    def local_gap():
        # Measure plane origins along socket's current normal, not world Z.
        rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation,d.qpos[qa+3:qa+7])
        return float((rotation.reshape(3,3).T @ (d.qpos[qb:qb+3]-d.qpos[qa:qa+3]))[2])
    initial_gap = local_gap()
    minimum_gap = initial_gap; maximum_gap = initial_gap
    max_rel_speed = 0.; max_tilt_deg = 0.; magnet_norms=[]
    energies=[]; trace_rows=[]; max_gap_after_contact=0.;contact_seen=False
    wrench = np.zeros(m.nv)
    inertia=np.zeros((m.nv,m.nv))
    steps = round(duration/timestep)
    for i in range(steps):
        # mj_step has advanced qpos but its site transforms correspond to the
        # previous step. Refresh position/jacobians before recomputing forces.
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fullM(m,d,inertia)
        _,_,potential=four_magnet_forces(d.site_xpos[sb],d.site_xpos[sa],params)
        energies.append(float(.5*d.qvel@inertia@d.qvel+potential))
        magnetic_qfrc(m, d, sb, sa, params, out=wrench)
        d.qfrc_applied[:] = wrench
        magnet_norms.append(float(np.linalg.norm(wrench)))
        if mode == 'tilt':
            # Gradual load avoids an artificial one-step torque impulse.
            torque = np.array([tilt_nm * min(i*timestep/.1, 1.),0.,0.])
            mujoco.mj_applyFT(m, d, np.zeros(3), torque, d.xipos[b], b, d.qfrc_applied)
            mujoco.mj_applyFT(m, d, np.zeros(3), -torque, d.xipos[a], a, d.qfrc_applied)
        # Contacts after mj_step correspond to the pre-step state. Record the
        # local gap before advancing so the two describe the same instant.
        before_gap=local_gap()
        mujoco.mj_step(m, d)
        gap = local_gap()
        minimum_gap=min(minimum_gap,gap);maximum_gap=max(maximum_gap,gap)
        max_rel_speed=max(max_rel_speed,float(np.linalg.norm(d.qvel[vb:vb+3]-d.qvel[va:va+3])))
        ra=np.empty(9);rb=np.empty(9)
        mujoco.mju_quat2Mat(ra,d.qpos[qa+3:qa+7]);mujoco.mju_quat2Mat(rb,d.qpos[qb+3:qb+7])
        rel=ra.reshape(3,3).T@rb.reshape(3,3)
        max_tilt_deg=max(max_tilt_deg, math.degrees(math.acos(float(np.clip(rel[2,2],-1.,1.)))))
        if d.ncon:
            if first_contact is None:first_contact=i*timestep
            contact_seen=True
            peak_contacts=max(peak_contacts,int(d.ncon))
            peak_penetration=max(peak_penetration,-float(np.min(d.contact.dist)))
        if contact_seen:
            max_gap_after_contact=max(max_gap_after_contact,gap)
        if trace and (i < 5 or (i+1) % max(1,round(.002/timestep))==0 or d.ncon and len(trace_rows)<12):
            worst=min(d.contact,key=lambda item:item.dist,default=None)
            trace_rows.append({'time_s':i*timestep,'local_gap_mm':before_gap*1000,
                'contacts':int(d.ncon),
                'deepest_mm':(-float(worst.dist)*1000 if worst is not None else 0.),
                'geom1':(m.geom(worst.geom1).name if worst is not None else None),
                'geom2':(m.geom(worst.geom2).name if worst is not None else None),
                'normal':(np.asarray(worst.frame[:3]).tolist() if worst is not None else None)})
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or np.any(d.warning.number):
            break
    return {'mode':mode, 'timestep_ms':timestep*1000, 'requested_duration_s':duration,
            'simulated_time_s':float(d.time), 'initial_gap_mm':initial_gap*1000,
            'final_gap_mm':local_gap()*1000,
            'min_gap_mm':minimum_gap*1000,'max_gap_mm':maximum_gap*1000,
            'first_contact_s':first_contact, 'peak_contacts':peak_contacts,
            'max_gap_after_contact_mm':max_gap_after_contact*1000,
            'peak_penetration_mm':peak_penetration*1000, 'max_relative_speed_m_s':max_rel_speed,
            'max_relative_tilt_deg':max_tilt_deg, 'peak_generalized_force_norm':max(magnet_norms, default=0.),
            'magnetic_plus_kinetic_energy_J_initial_final':([energies[0],energies[-1]] if energies else []),
            'energy_note':'excludes contact compliance, gravity, and external torque work',
            'warnings':int(sum(d.warning.number)),
            'finite':bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
            'assumptions':{'magnet_radius_mm':radius_m*1000,'center_setback_mm':setback_m*1000,
                           'single_pair_force_at_2x_setback_N':force_n,'tilt_torque_Nm':tilt_nm if mode=='tilt' else 0.,
                           'all_magnets_attractive':True,'physical_calibration':False,
                           'contact_solref_timeconst_s':contact_timeconst,
                           'contact_solimp_min_max':list(contact_impedance),
                           'initial_closing_speed_m_s':initial_closing_speed_m_s},
            'trace':trace_rows if trace else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--duration-s',type=float,default=.25)
    p.add_argument('--force-N',type=float,default=5.)
    p.add_argument('--initial-gap-mm',type=float,default=3.)
    p.add_argument('--tilt-Nm',type=float,default=.2)
    args=p.parse_args()
    print('EXPERIMENT: hypothetical magnet positions/strength, no weld or robot; not validated hardware physics.', flush=True)
    for mode in ('approach','hold','tilt'):
        for timestep in (.001,.0005,.00025):
            print(json.dumps(run(mode,timestep,args.duration_s,args.initial_gap_mm/1000,
                                 args.force_N,tilt_nm=args.tilt_Nm)),flush=True)


if __name__ == '__main__':
    main()
