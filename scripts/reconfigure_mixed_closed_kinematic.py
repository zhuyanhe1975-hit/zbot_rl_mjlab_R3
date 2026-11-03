"""Connected kinematic FSM to internal angles 270,270,180,90,90; brace unchanged."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_mixed_closed import MixedClosedKinematics,EXPERIMENT_ID,TARGET_DEG
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import KinematicPlanningError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--headless',dest='viewer',action='store_false');p.set_defaults(viewer=False)
    p.add_argument('--steps-per-phase',type=int,default=20);p.add_argument('--phase-duration',type=float,default=1.)
    p.add_argument('--order',type=int,nargs=4,default=[1,2,4,5],help='1-based internal connections; connection 3 stays unchanged')
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/mixed_closed_kinematic/latest.json'))
    args=p.parse_args()
    if sorted(args.order)!=[1,2,4,5]:p.error('--order must contain 1 2 4 5 once each')
    planner=MixedClosedKinematics(args.steps_per_phase,args.phase_duration,tuple(i-1 for i in args.order))
    error=None
    try:planner.plan()
    except KinematicPlanningError as exc:
        error=str(exc)
        if planner.frames:
            last=planner.frames[-1];planner.q=last.q.copy();planner.roots=last.roots.copy();planner.active=last.eq_active.copy()
        planner.record('IK_FAILED',-1,event=True);print('KINEMATIC_FAILED',error,flush=True)
    checks=planner.target_checks()
    success=(error is None and checks['target_angles_match'] and checks['variant_selection_matches']
             and checks['head_tail_original_closed'] and checks['middle_connection_original']
             and checks['connected_all_frames'] and checks['final_connected_faces']==6
             and checks['max_active_weld_position_error_m']<2e-5 and checks['max_active_weld_angle_error_rad']<2e-4)
    report={'experiment':EXPERIMENT_ID,'mode':'connected_ring_kinematics_only','success':success,'error':error,
            'initial_internal_angles_deg':[180]*5,'target_internal_angles_deg':TARGET_DEG,'order_1_based':args.order,
            'head_tail_registration_deg':180,'joint_limits_ignored':True,
            'not_validated':['torque','loads','dynamics','collisions','actuator_tracking','joint_speed_feasibility'],
            'events':planner.events,'stages':planner.stage_errors,'frames':len(planner.frames),'target_checks':checks}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print('REPORT',args.output,'success',success,checks,flush=True)
    if args.viewer and planner.frames:
        import mujoco.viewer
        control={'pause':False,'restart':False}
        def key(code):
            if code==32:control['pause']=not control['pause']
            elif code in (13,257):control['restart']=True
        planner.apply_frame(planner.frames[0])
        with mujoco.viewer.launch_passive(planner.model,planner.data,key_callback=key) as viewer:
            viewer.cam.distance=1.5;viewer.cam.lookat[:]=planner.frames[0].roots[:,:3,3].mean(0)
            elapsed=0.;last=time.perf_counter();timestamps=np.array([f.time for f in planner.frames]);previous=None
            while viewer.is_running():
                now=time.perf_counter()
                if not control['pause']:elapsed+=now-last
                last=now
                if control['restart']:elapsed=0.;control['restart']=False
                index=max(0,min(int(np.searchsorted(timestamps,elapsed,side='right')-1),len(timestamps)-1))
                frame=planner.frames[index];planner.apply_frame(frame)
                current=(frame.connector,frame.phase)
                if current!=previous:print('VIEW',None if frame.connector<0 else frame.connector+1,frame.phase,flush=True);previous=current
                viewer.sync();time.sleep(.01)
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
