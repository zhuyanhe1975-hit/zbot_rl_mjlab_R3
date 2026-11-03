"""Visualize the refined zero-residual reference in the actual GPU simulator.

By default one GPU world runs; optional batching can display a selected world.
This is reference-controller inference, not a trained RL policy.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import copy
import json
import threading
import time
from pathlib import Path

import mujoco
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_refined_teacher_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES

REFERENCE=Path('assets/reconfiguration/refined_forward_velocity_reference.npz')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=1,help='independent GPU worlds (default: 1)')
    parser.add_argument('--world',type=int,default=0,help='world displayed in the window')
    parser.add_argument('--reference-rate',type=float,default=.2,help='reference time scale (default: 0.2)')
    parser.add_argument('--max-time',type=float,default=90.,help='simulation seconds before stopping')
    parser.add_argument('--headless',action='store_true',help='run without opening a window')
    args=parser.parse_args()
    if args.num_envs<1 or not 0<=args.world<args.num_envs or not 0<args.reference_rate<=1 or args.max_time<=0:
        parser.error('invalid batch size, world, reference rate, or time limit')
    cfg=make_refined_teacher_cfg(args.num_envs,REFERENCE,args.reference_rate)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    reset_event=threading.Event();pause_event=threading.Event()
    sim_time=0.;status='running';last_failed=0;last_completed=0
    try:
        with torch.no_grad():
            env.reset();actions=torch.zeros((args.num_envs,7),device=env.device)
            s=env._disconnect_state
            def step():
                nonlocal sim_time,status,last_failed,last_completed
                _,_,terminated,truncated,_=env.step(actions);sim_time+=env.step_dt
                done=s.fsm.phase==P['DONE'];failed=(terminated|truncated)&~done
                last_failed=int(failed.sum());last_completed=int(done.sum())
                if last_failed:status='failure'
                elif last_completed==args.num_envs:status='complete'
                elif sim_time>=args.max_time:status='time limit'

            def reset():
                nonlocal sim_time,status,last_failed,last_completed
                env.reset();sim_time=0.;status='running';last_failed=0;last_completed=0

            if args.headless:
                while status=='running':step()
            else:
                import mujoco.viewer
                from zbot_rl_mjlab.viewer import _AsyncOverlayHandle
                model=copy.copy(env.sim.mj_model);data=mujoco.MjData(model)
                root_addresses=[int(model.jnt_qposadr[model.joint(f'module_{i}/module_root').id]) for i in range(6)]
                def on_key(key):
                    if key in (13,257,335):reset_event.set()
                    elif key==32:pause_event.set()
                with mujoco.viewer.launch_passive(model,data,key_callback=on_key) as viewer:
                    overlay=_AsyncOverlayHandle(viewer)
                    viewer.cam.lookat[:]=[0.,0.,.08];viewer.cam.distance=1.4;viewer.cam.elevation=-55
                    anchor=time.perf_counter();paused=False;announced=False
                    while viewer.is_running():
                        if reset_event.is_set():
                            reset_event.clear();reset();anchor=time.perf_counter();paused=False
                        if pause_event.is_set():
                            pause_event.clear();paused=not paused;anchor=time.perf_counter()-sim_time
                        if not paused and status=='running':
                            desired=time.perf_counter()-anchor;deadline=time.perf_counter()+.012
                            while status=='running' and sim_time<desired and time.perf_counter()<deadline:step()
                        origin=env.scene.env_origins[args.world].cpu().numpy()
                        data.qpos[:]=env.sim.data.qpos[args.world].cpu().numpy()
                        for address in root_addresses:data.qpos[address:address+3]-=origin
                        data.qvel[:]=env.sim.data.qvel[args.world].cpu().numpy()
                        data.eq_active[:]=env.sim.data.eq_active[args.world].cpu().numpy()
                        mujoco.mj_forward(model,data)
                        phase=PHASES[int(s.fsm.phase[args.world])]
                        loads=(s.utilization[args.world]*cfg.break_force).cpu().tolist()
                        connected=s.connected()[args.world].cpu().tolist()
                        labels='Mode\nGPU worlds / displayed\nTime / reference rate\nPhase / target\nFace loads 1-6 (N)\nConnected 1-6\nOther peak (N)\nComplete / failed\nStatus\nKeys'
                        values=('Reference, zero residual (not RL)\n'
                            f'{args.num_envs} / {args.world}\n{sim_time:.2f}s / {args.reference_rate:.2f}x\n'
                            f'{phase} / {int(s.fsm.target[args.world])+1}\n'
                            +' '.join(f'{v:.1f}' for v in loads)+'\n'
                            +' '.join('1' if v else '0' for v in connected)+'\n'
                            +f'{float(s.max_other_load[args.world]):.1f}\n{last_completed} / {last_failed}\n'
                            +('paused' if paused else status)+'\nEnter reset, Space pause')
                        overlay.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                          mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,labels,values))
                        viewer.sync();time.sleep(.002)
                        if not announced:
                            print('GPU viewer ready; Enter resets, Space pauses.',flush=True);announced=True
                    overlay.close()
            print('GPU_VIEW',json.dumps({'mode':'reference_zero_residual','num_envs':args.num_envs,
                  'displayed_world':args.world,'simulation_time_s':sim_time,'status':status,
                  'completed':last_completed,'failed':last_failed,'physics_dt_s':cfg.sim.mujoco.timestep}),flush=True)
            return 1 if status=='failure' else 0
    finally:
        env.close();torch.cuda.set_stream(previous)

if __name__=='__main__':raise SystemExit(main())
