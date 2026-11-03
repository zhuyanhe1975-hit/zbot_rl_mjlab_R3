"""Evaluate each loading target from the same initial closed ring; stop at rupture."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
from reconfigure_mixed_full_cpu import make_config
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu import FullCpuReconfiguration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=int,choices=range(1,7),action='append',help='1-based target; repeat for multiple faces (default: all six)')
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/static_loading/evaluation.json'))
    args=p.parse_args();cfg=make_config(SimpleNamespace(speed_deg_s=90.,accel_deg_s2=720.))
    baseline=FullCpuReconfiguration(cfg);start=time.perf_counter();results=[]
    try:
        # Stop as soon as the first closed-ring LOAD is entered: no loading or
        # disconnection is performed in the shared initial condition.
        while baseline.phase not in ('LOAD','FAILED') and baseline.data.time<30.:baseline.step()
        if baseline.phase!='LOAD' or not baseline.connected().all():
            raise RuntimeError(f'Could not prepare closed ring: {baseline.failure or baseline.phase}')
        for face in args.target or range(1,7):
            f=FullCpuReconfiguration(cfg)
            try:
                # Duplicate exactly the same physical state and mutable model
                # parameters; each target runs in an independent simulation.
                f.model.eq_solref[:]=baseline.model.eq_solref
                f.model.pair_geom1[:]=baseline.model.pair_geom1;f.model.pair_geom2[:]=baseline.model.pair_geom2
                mujoco.mj_copyData(f.data,f.model,baseline.data)
                f.phase='SETTLE_HEADTAIL';f.step_count=baseline.step_count;f.phase_start=float(f.data.time)
                f.start_load('LOAD_HEADTAIL' if face==6 else 'LOAD',face-1)
                f.peak_other=0.;f.peak_load[:]=0.;other=np.arange(6)!=(face-1);other_peak=0.;max_speed=0.;max_torque=0.
                while f.phase not in ('FAILED','DONE') and f.connected()[face-1] and f.phase_time<40.:
                    f.step()
                    other_peak=max(other_peak,float(f.loads[other].max()))
                    max_speed=max(max_speed,float(np.abs(f.data.qvel[f.dofs]).max()))
                    max_torque=max(max_torque,float(np.abs(f.data.qfrc_actuator[f.dofs]).max()))
                reached=not f.connected()[face-1] and not f.failure
                rec=f.loading_records[-1] if f.loading_records else {}
                report={'target':face,'reached':reached,'failure':f.failure or ('' if reached else 'loading timeout'),
                        'solver':rec,'actual_target_at_stop_N':float(f.loads[face-1]),
                        'actual_other_loads_at_stop_N':f.loads[other].tolist(),
                        'actual_other_peak_during_loading_N':other_peak,
                        'actual_margin_N':cfg.break_load-other_peak if reached else None,
                        'max_joint_speed_rad_s':max_speed,'max_actuator_torque_Nm':max_torque,
                        'elapsed_s':float(f.data.time-f.phase_start)}
                if rec.get('predicted_N') is not None:
                    predicted=np.asarray(rec['predicted_N']);report['predicted_margin_N']=cfg.break_load-float(predicted[other].max())
                results.append(report)
                print('STATIC_RESULT',json.dumps(report,ensure_ascii=False),flush=True)
            finally:f.close()
        out={'configuration':{'physics_dt_s':cfg.dt,'threshold_N':cfg.break_load,'torque_limit_Nm':cfg.continuous_torque,
              'initial_ring':'all 180 degrees; same captured physical state for each target'},
             'results':results,'wall_time_s':time.perf_counter()-start,
             'scope':'Each independent trial stops on target rupture or failure; no separation or reconnection. Margins are observed or model-predicted, not real-world guarantees.'}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
        print('REPORT',args.output,flush=True)
        return 0 if all(r['reached'] for r in results) else 1
    finally:baseline.close()


if __name__=='__main__':raise SystemExit(main())
