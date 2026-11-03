"""Separate standalone FSM planning, dispatch, safety and dynamics costs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg,ReferencePhase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-time',type=float,default=60.)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/mixed_closed_reference/pipeline_profile.json'))
    args=p.parse_args();env=ManagerBasedRlEnv(reference_env_cfg(),device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();a=env.action_manager.get_term('joint_torque');c=a.distance_controller
            action=torch.zeros(1,6,device=env.device)
            # Isolate the solver on the real initialized GPU model, with its
            # native PD controls held constant. No FSM or protection dispatch.
            env.step(action);torch.cuda.synchronize();wp.synchronize()
            start=time.perf_counter()
            for _ in range(1000):env.sim.step()
            wp.synchronize();torch.cuda.synchronize();raw_wall=time.perf_counter()-start
            env.reset()
            totals={};samples={};plans=[];restores=[];prepare_serial=0
            wp_stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
            def wrap(obj,name,key,stream=None):
                original=getattr(obj,name);restores.append((obj,name,original));totals[key]={'count':0,'host_s':0.}
                def timed(*v,**kw):
                    sampled=stream is not None and env._sim_step_counter%50==0
                    serial=prepare_serial
                    if sampled:
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record(stream)
                    start=time.perf_counter();result=original(*v,**kw);wall=time.perf_counter()-start
                    totals[key]['count']+=1;totals[key]['host_s']+=wall
                    if sampled:
                        end.record(stream)
                        if serial==prepare_serial:samples.setdefault(key,[]).append((begin,end))
                    return result
                setattr(obj,name,timed)
            original_prepare=c.prepare
            def prepare(s):
                nonlocal prepare_serial
                prepare_serial+=1;start=time.perf_counter();result=original_prepare(s)
                plans.append({'connection':c.events[-1]['connection'],'wall_s':time.perf_counter()-start});return result
            c.prepare=prepare;restores.append((c,'prepare',original_prepare))
            wrap(c.planner,'plan','motion_ik_path')
            stream=torch.cuda.current_stream()
            for obj,name,key,st in [
                (env.action_manager,'process_action','process_control',stream),
                (env.action_manager,'apply_action','apply_control_and_fsm',stream),
                (env.scene,'write_data_to_sim','write_actuator_targets',wp_stream),
                (env.sim,'step','physics_step_launch',wp_stream),
                (env.scene,'update','scene_update',None),
                (env.metrics_manager,'compute_substep','torque_weld_collision_monitor',stream),
                (env.sim,'forward','control_end_forward',wp_stream),
                (env.sim,'sense','control_end_sense',wp_stream),
                (env.observation_manager,'compute','observations',stream),
                (env.metrics_manager,'compute','control_end_metrics',stream)]:wrap(obj,name,key,st)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(max(1,round(args.max_time/env.step_dt))):
                env.step(action)
                if c.phase_host in (ReferencePhase.DONE,ReferencePhase.FAILED):break
            wp.synchronize();torch.cuda.synchronize();wall=time.perf_counter()-start
            for key,items in samples.items():
                values=[b.elapsed_time(e) for b,e in items]
                totals[key]['sampled_stream_elapsed_ms']={'count':len(values),'mean':float(np.mean(values)),
                    'median':float(np.median(values)),'p95':float(np.quantile(values,.95))}
            for obj,name,original in reversed(restores):setattr(obj,name,original)
            report={'success':c.phase_host==ReferencePhase.DONE,'failure_message':c.failure_message,
                'sim_s':totals['physics_step_launch']['count']*env.physics_dt,'wall_s':wall,
                'raw_gpu_physics_1000_steps_wall_s':raw_wall,'per_method':totals,'motion_prepare':plans,
                'motion_prepare_total_s':sum(x['wall_s'] for x in plans),'other_peak_N':float(env._disconnect_state.max_other_load[0]),
                'streams':{'torch':stream.cuda_stream,'warp':wp_stream.cuda_stream},
                'timing_notes':'Host timings include launch overhead and waits encountered in each method, not pure GPU compute. CUDA events are sampled every 50 physics steps on the relevant stream, omit preparation-spanning samples, and include host launch gaps. motion_ik_path and motion_prepare are nested in apply_control_and_fsm; do not add twice. Raw GPU physics benchmark omits FSM/monitoring and is not a proposed execution mode.'}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2));print('REPORT',args.output)
    finally:env.close()


if __name__=='__main__':main()
