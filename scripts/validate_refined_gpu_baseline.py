"""Validate the fixed-initial-state refined teacher in parallel on the GPU.

This is a zero-policy diagnostic, not a trained policy or a robustness test.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_refined_teacher_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P

REFERENCE=Path('assets/reconfiguration/refined_forward_velocity_reference.npz')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--reference-rate',type=float,default=.2)
    parser.add_argument('--max-time',type=float,default=90.)
    parser.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/rl_incremental_baseline/gpu_validation.json'))
    args=parser.parse_args()
    if args.num_envs<1 or not 0<args.reference_rate<=1 or args.max_time<=0:
        parser.error('num-envs and max-time must be positive, and reference-rate must be in (0, 1]')
    cfg=make_refined_teacher_cfg(args.num_envs,REFERENCE,args.reference_rate)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    started=time.perf_counter()
    try:
        with torch.no_grad():
            env.reset();action=torch.zeros(args.num_envs,7,device=env.device)
            completed=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
            peak_other=torch.zeros(args.num_envs,device=env.device)
            peak_module=torch.zeros_like(peak_other);peak_ground=torch.zeros_like(peak_other)
            first_wrong=torch.zeros_like(completed);first_contact=torch.zeros_like(completed);first_margin=torch.zeros_like(completed)
            terminal_angle=torch.zeros(args.num_envs,device=env.device)
            terminal_speed=torch.zeros(args.num_envs,device=env.device)
            for step in range(int(args.max_time/env.step_dt)):
                _,_,terminated,truncated,_=env.step(action)
                s=env._disconnect_state;pending=terminated|truncated
                newly_done=env.termination_manager.get_term('success')&~completed
                failed=pending&~completed&~newly_done
                peak_other.copy_(torch.where(~completed,torch.maximum(peak_other,s.max_other_load),peak_other))
                peak_module.copy_(torch.where(~completed,torch.maximum(peak_module,s.max_module_depth),peak_module))
                peak_ground.copy_(torch.where(~completed,torch.maximum(peak_ground,s.max_ground_depth),peak_ground))
                first_wrong|=s.wrong_break&~completed;first_contact|=s.contact_violation&~completed
                first_margin|=s.safety_margin_violation&~completed
                q,v=env.action_manager.get_term('joint_torque').joint_state()
                terminal_angle.copy_(torch.where(newly_done,q.abs().amax(-1),terminal_angle))
                terminal_speed.copy_(torch.where(newly_done,v.abs().amax(-1),terminal_speed))
                completed|=newly_done
                if step%50==0 or bool(failed.any()) or bool(completed.all()):
                    print('PROGRESS',json.dumps({'time_s':round((step+1)*env.step_dt,3),
                        'phase_counts':torch.bincount(s.fsm.phase,minlength=len(P)).cpu().tolist(),
                        'failed':int(failed.sum()),'completed':int(completed.sum())}),flush=True)
                if bool(failed.any()) or bool(completed.all()):break
                # Worlds finish asynchronously. Reset only finished episodes
                # so MJLab can continue the remaining first episodes; their
                # subsequent results do not affect the latched first result.
                if bool(pending.any()):env.reset(env_ids=pending.nonzero(as_tuple=False).flatten())
            report={'num_envs':args.num_envs,'physics_dt_s':cfg.sim.mujoco.timestep,
                'control_dt_s':env.step_dt,'reference_rate':args.reference_rate,
                'terminal_angle_tolerance_rad':cfg.full_final_angle_tolerance,
                'simulation_time_s':(step+1)*env.step_dt,'wall_time_s':time.perf_counter()-started,
                'completed':int(completed.sum()),'failed':int(failed.sum()),
                'phase_counts':torch.bincount(s.fsm.phase,minlength=len(P)).cpu().tolist(),
                'failure_codes':torch.bincount(s.fsm.failure_code,minlength=6).cpu().tolist(),
                'peak_other_N':float(peak_other.max()),
                'completed_max_joint_angle_rad':float(terminal_angle.max()),
                'completed_max_joint_speed_rad_s':float(terminal_speed.max()),
                'max_module_penetration_m':float(peak_module.max()),
                'max_ground_penetration_m':float(peak_ground.max()),
                'wrong_break':int(first_wrong.sum()),'contact_violation':int(first_contact.sum()),
                'margin_violation':int(first_margin.sum()),
                'all_completed':bool(completed.all() and not failed.any())}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('RESULT',json.dumps(report),flush=True)
            return 0 if report['all_completed'] else 1
    finally:
        env.close();torch.cuda.set_stream(previous)

if __name__=='__main__':raise SystemExit(main())
