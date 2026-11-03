"""Plan and display the entire 180->270 sequence using FK/IK only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import KinematicRz270,EXPERIMENT_ID,KinematicPlanningError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--headless',dest='viewer',action='store_false');p.set_defaults(viewer=False)
    p.add_argument('--steps-per-phase',type=int,default=50);p.add_argument('--phase-duration',type=float,default=1.)
    p.add_argument('--order',type=int,nargs=5,default=[0,1,2,3,4])
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/kinematic_rz270/latest.json'))
    args=p.parse_args()
    planner=KinematicRz270(args.steps_per_phase,args.phase_duration,args.order)
    error=None
    try:planner.plan()
    except KinematicPlanningError as exc:
        error=str(exc)
        if planner.frames:
            last=planner.frames[-1];planner.q=last.q.copy();planner.roots=last.roots.copy();planner.active=last.eq_active.copy()
        planner.record('IK_FAILED',-1,event=True);print('KINEMATIC_FAILED',error,flush=True)
    reference=planner.validate_reference()
    constraints=planner.active_constraint_errors()
    success=(error is None and reference['max_root_position_error_m']<1e-5
             and reference['max_root_angle_error_rad']<1e-4 and reference['joint_zero_error_rad']<1e-6
             and constraints['max_active_weld_position_error_m']<2e-5 and constraints['max_active_weld_angle_error_rad']<2e-4)
    report={'experiment':EXPERIMENT_ID,'mode':'connected_chain_FK_IK','success':success,'error':error,
            'joint_limits_ignored':True,'not_validated':['torque','loads','dynamics','collisions','actuator_tracking','joint_speed_feasibility'],
            'connected_all_frames':all(planner.connectivity(f.eq_active)['components']==1 for f in planner.frames),
            'min_connected_faces':min(planner.connectivity(f.eq_active)['connected_faces'] for f in planner.frames),
            'phase_duration_s':args.phase_duration,'frames':len(planner.frames),'events':planner.events,
            'stages':planner.stage_errors,'reference_rz270':reference,'active_constraint_checks':constraints,
            'kinematic_basis':'One connected chain with one anchored root. Other module roots derive exclusively from joint FK; disconnected fragments are forbidden.',
            'root_pose_convention':'Head root remains fixed during final unfold; reference match is up to a common rigid transform.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print('REPORT',args.output,'success',success,reference,flush=True)
    if args.viewer and planner.frames:
        import mujoco.viewer
        control={'pause':False,'restart':False}
        def key(keycode):
            if keycode==32:control['pause']=not control['pause']
            elif keycode in (13,257):control['restart']=True
        with mujoco.viewer.launch_passive(planner.model,planner.data,key_callback=key) as viewer:
            viewer.cam.distance=1.5;viewer.cam.lookat[:]=planner.frames[0].roots[:,:3,3].mean(0)
            elapsed=0.;last=time.perf_counter();times=np.array([f.time for f in planner.frames]);previous=None
            while viewer.is_running():
                now=time.perf_counter()
                if not control['pause']:elapsed+=now-last
                last=now
                if control['restart']:elapsed=0.;control['restart']=False
                index=min(int(np.searchsorted(times,elapsed,side='right')-1),len(times)-1);index=max(index,0)
                frame=planner.frames[index];planner.apply_frame(frame)
                state=(frame.connector,frame.phase)
                if state!=previous:print('VIEW',state,flush=True);previous=state
                viewer.sync();time.sleep(.01)
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
