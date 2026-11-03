"""Straight -> curl/dock -> mixed reconfiguration -> release/unfold, CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,time,threading
from pathlib import Path
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu import FullCpuReconfiguration,FullCpuConfig


def make_config(args):
    return FullCpuConfig(dt=.001,stiffness=400.,speed_deg_s=args.speed_deg_s,acceleration_deg_s2=args.accel_deg_s2,
                      initial_hold_seconds=.25,settle_seconds=.25,capture_position_tolerance=.001,capture_angle_tolerance_deg=2.,capture_linear_speed=.1,capture_angular_speed=.6,motion_wait_seconds=10.,docking_stiffness=0.,approach_feedback=True,curl_speed_multiplier=2.,separation_gap=.021,overlap_rotation=True,headtail_parallel_gap=0.,break_load=100.,other_stop_load=100.,headtail_plan_load=100.,headtail_prediction_limit=100.,headtail_one_step_prediction_limit=100.,headtail_one_step_validation_limit=100.,headtail_validation_margin=0.,loading_other_prediction_limit=100.,one_shot_loading=True,headtail_model_horizon_s=.06,headtail_model_headroom=6.,loading_model_headroom=10.,post_break_hold_s=0.,headtail_break_hold_s=.5,loading_goal_margin=0.,unfold_prediction_limit=100.,closest_unfold=True,closest_unfold_gain=9.,closest_unfold_max_speed=7.2,closest_unfold_ramp_s=2.,direct_unfold=False,staged_unfold=False,
                      unfold_target_step=.008,unfold_target_step_max=.024,unfold_accel_delay_s=2.,unfold_accel_duration_s=8.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--headless',dest='viewer',action='store_false');p.set_defaults(viewer=False)
    p.add_argument('--max-time',type=float,default=180.)
    p.add_argument('--speed-deg-s',type=float,default=105.);p.add_argument('--accel-deg-s2',type=float,default=980.)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/mixed_full_cpu/latest.json'))
    args=p.parse_args()
    if not all(math.isfinite(x) and x>0 for x in (args.max_time,args.speed_deg_s,args.accel_deg_s2)):
        p.error('positive finite parameters required')
    cfg=make_config(args)
    fsm=FullCpuReconfiguration(cfg);start=time.perf_counter()
    try:
        if args.viewer:
            import mujoco
            import mujoco.viewer
            restart=threading.Event()
            def on_key(key):
                # GLFW Enter and keypad Enter; 13 also covers native backends.
                if key in (13,257,335):restart.set()
            while True:
                restart.clear()
                with mujoco.viewer.launch_passive(fsm.model,fsm.data,key_callback=on_key) as viewer:
                    viewer.cam.lookat[:]=[0.,0.,.08];viewer.cam.distance=1.4;viewer.cam.elevation=-55
                    anchor_wall=time.perf_counter();anchor_sim=float(fsm.data.time);next_status=0.
                    while viewer.is_running() and not restart.is_set():
                        if fsm.phase not in ('DONE','FAILED') and fsm.data.time<args.max_time:
                            desired=anchor_sim+(time.perf_counter()-anchor_wall)
                            deadline=time.perf_counter()+.012
                            while (fsm.data.time<desired and time.perf_counter()<deadline and
                                   fsm.phase not in ('DONE','FAILED') and not restart.is_set()):
                                fsm.step()
                        if time.perf_counter()>=next_status:
                            labels='Time\nPhase\nTarget\nFace loads 1-6 (N)\nPeak loads 1-6 (N)\nPeak other (N)\nConnected 1-6\nRestart'
                            values=(f'{fsm.data.time:.2f}s\n{fsm.phase}\n{fsm.target+1}\n'
                                    + ' '.join(f'{x:.1f}' for x in fsm.loads)+'\n'
                                    + ' '.join(f'{x:.1f}' for x in fsm.peak_load)+'\n'
                                    + f'{fsm.peak_other:.1f}\n'
                                    + ' '.join('1' if x else '0' for x in fsm.connected())+'\nEnter')
                            if fsm.failure:labels+='\nFailure';values+='\n'+fsm.failure[:120]
                            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                              mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,labels,values))
                            next_status=time.perf_counter()+.1
                        viewer.sync();time.sleep(.002)
                if not restart.is_set():break
                fsm.close()
                fsm=FullCpuReconfiguration(cfg)
                start=time.perf_counter()
                print('CPU_FSM restart: Enter',flush=True)

        else:
            while fsm.phase not in ('DONE','FAILED') and fsm.data.time<args.max_time:fsm.step()
        report=fsm.report();report['wall_time_s']=time.perf_counter()-start
        if fsm.phase not in ('DONE','FAILED'):report['stop_reason']='simulation time limit or viewer closed'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['terminal'],indent=2));print('REPORT',args.output,'success=',report['success'],'wall_s=',report['wall_time_s'])
        return 0 if args.viewer or report['success'] else 1
    finally:fsm.close()


if __name__=='__main__':raise SystemExit(main())
