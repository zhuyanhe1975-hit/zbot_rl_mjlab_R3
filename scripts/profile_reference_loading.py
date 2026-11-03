"""Measure real inner-force planning calls separately from dynamics execution."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration import batched_loading
from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg,ReferencePhase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-time',type=float,default=60.)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/mixed_closed_reference/loading_profile.json'))
    args=p.parse_args()
    if args.max_time<=0:p.error('--max-time must be positive')
    rows=[];current=None
    original_search=batched_loading.projected_increments
    env=ManagerBasedRlEnv(reference_env_cfg(),device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();a=env.action_manager.get_term('joint_torque');c=a.distance_controller;s=env._disconnect_state
            original_response=a.loading.response;original_update=a.loading.update
            def measure(name,call):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();start=time.perf_counter();result=call();wall_ms=(time.perf_counter()-start)*1000;end.record()
                current[name]={'host_call_ms':wall_ms,'events':(begin,end)}
                return result
            def response(*v,**kw):return measure('response_matrix',lambda:original_response(*v,**kw))
            def search(*v,**kw):return measure('candidate_optimization',lambda:original_search(*v,**kw))
            def update(*v,**kw):
                nonlocal current
                if c.phase_host!=ReferencePhase.LOAD:return original_update(*v,**kw)
                current={'connection':c.events[-1]['connection'],'sim_time_s':env._sim_step_counter*env.physics_dt}
                result=measure('full_update',lambda:original_update(*v,**kw))
                if 'response_matrix' in current:rows.append(current)
                current=None
                return result
            a.loading.response=response;a.loading.update=update;batched_loading.projected_increments=search
            action=torch.zeros(1,6,device=env.device);start=time.perf_counter()
            for _ in range(max(1,round(args.max_time/env.step_dt))):
                env.step(action)
                if c.phase_host in (ReferencePhase.DONE,ReferencePhase.FAILED):break
            torch.cuda.synchronize();run_wall=time.perf_counter()-start
            for row in rows:
                for key in ('response_matrix','candidate_optimization','full_update'):
                    begin,end=row[key].pop('events');row[key]['cuda_stream_ms']=begin.elapsed_time(end)
            def stats(values):
                x=np.asarray(values,dtype=float)
                return {'count':len(x),'sum_ms':float(x.sum()),'mean_ms':float(x.mean()),'median_ms':float(np.median(x)),
                        'p95_ms':float(np.quantile(x,.95)),'max_ms':float(x.max())} if len(x) else {'count':0}
            def summarize(selected):
                return {key:{metric:stats([r[key][metric] for r in selected]) for metric in ('host_call_ms','cuda_stream_ms')}
                        for key in ('response_matrix','candidate_optimization','full_update')}
            report={'success':c.phase_host==ReferencePhase.DONE,'terminal_phase':c.phase_host.name,
                    'failure_message':c.failure_message,'sim_s':env._sim_step_counter*env.physics_dt,'run_wall_s':run_wall,
                    'gpu':torch.cuda.get_device_name(),'physics_dt_s':env.physics_dt,'control_dt_s':env.step_dt,
                    'candidate_directions':32,'projection_iterations':20,'safety_bisection_iterations':10,
                    'other_peak_N':float(s.max_other_load[0]),'connected':s.connected()[0].tolist(),
                    'all_calls':summarize(rows),'excluding_first_call':summarize(rows[1:]),
                    'per_connection':{str(i):summarize([r for r in rows if r['connection']==i]) for i in (1,2,4,5)},
                    'first_call':rows[0] if rows else None,'calls':rows,
                    'timing_notes':'Host-call wall time includes Python launch overhead and existing synchronizations; CUDA events measure stream elapsed time including host launch gaps. No synchronization is inserted between components. Component times are nested in full_update and must not be added to it. Run wall also includes dynamics, motion IK and normal managers.'}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:report[k] for k in ('success','sim_s','run_wall_s','all_calls','excluding_first_call')},indent=2))
            print('REPORT',args.output)
    finally:
        batched_loading.projected_increments=original_search;env.close()


if __name__=='__main__':main()
