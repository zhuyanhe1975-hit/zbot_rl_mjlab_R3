"""Visualize a trained initial-closure RSL policy in one 2ms GPU world."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import copy
import hashlib
import json
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from scripts.train_refined_gpu_initial_closure import DEFAULT_LOG_DIR,InitialClosureRunner,make_agent_cfg,make_initial_closure_cfg,planner_signature
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES


def find_checkpoint(log_dir):
    log_dir=Path(log_dir)
    evaluation=log_dir/'deterministic_evaluation.json';best=log_dir/'best.pt'
    if evaluation.is_file():
        selected=json.loads(evaluation.read_text()).get('selected')
        if selected is None:
            raise FileNotFoundError('No checkpoint passed deterministic validation; use --checkpoint-file to inspect a candidate explicitly')
        if best.is_file() and hashlib.sha256(best.read_bytes()).hexdigest()==selected['sha256']:
            return best
    candidates=[]
    for path in Path(log_dir).glob('model_*.pt'):
        match=re.fullmatch(r'model_(\d+)\.pt',path.name)
        if match and path.is_file():candidates.append((int(match.group(1)),path))
    if not candidates:raise FileNotFoundError(f'No RSL checkpoint in {log_dir}; train first or pass --checkpoint-file')
    return max(candidates,key=lambda item:item[0])[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs','--env.scene.num-envs',type=int,default=1,help='viewer uses one GPU world')
    parser.add_argument('--log-dir',type=Path,default=DEFAULT_LOG_DIR)
    parser.add_argument('--checkpoint-file',type=Path,help='explicit RSL checkpoint; otherwise use latest')
    parser.add_argument('--max-time',type=float,default=20.)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args(argv)
    if args.num_envs!=1 or args.max_time<=0:parser.error('the viewer requires one environment and positive max-time')
    checkpoint=args.checkpoint_file or find_checkpoint(args.log_dir)
    cfg=make_initial_closure_cfg(1);cfg.auto_reset=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0');previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    reset_event=threading.Event();pause_event=threading.Event();sim_time=0.;status='running'
    try:
        with torch.no_grad():
            wrapped=RslRlVecEnvWrapper(env);agent=make_agent_cfg(1)
            reference_hash=planner_signature()
            runner=InitialClosureRunner(wrapped,asdict(agent),None,device='cuda:0',reference_sha256=reference_hash)
            runner.load(str(checkpoint),map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
            obs=wrapped.get_observations();s=env._disconnect_state
            print(f'GPU closure policy: {checkpoint} (deterministic)',flush=True)
            def step():
                nonlocal obs,sim_time,status
                action=policy(obs);obs,_,done,_=wrapped.step(action);sim_time+=env.step_dt
                if bool(done[0]):status='success' if bool(env.termination_manager.get_term('success')[0]) else 'failure'
                elif sim_time>=args.max_time:status='time limit'
            def reset():
                nonlocal obs,sim_time,status
                obs,_=wrapped.reset();sim_time=0.;status='running'
            if args.headless:
                while status=='running':step()
            else:
                import mujoco.viewer
                from zbot_rl_mjlab.viewer import _AsyncOverlayHandle
                model=copy.copy(env.sim.mj_model);data=mujoco.MjData(model)
                roots=[int(model.jnt_qposadr[model.joint(f'module_{i}/module_root').id]) for i in range(6)]
                def on_key(key):
                    if key in (13,257,335):reset_event.set()
                    elif key==32:pause_event.set()
                with mujoco.viewer.launch_passive(model,data,key_callback=on_key) as viewer:
                    overlay=_AsyncOverlayHandle(viewer)
                    viewer.cam.lookat[:]=[0.,0.,.08];viewer.cam.distance=1.4;viewer.cam.elevation=-55
                    anchor=time.perf_counter();paused=False;announced=False
                    while viewer.is_running():
                        if reset_event.is_set():reset_event.clear();reset();anchor=time.perf_counter();paused=False
                        if pause_event.is_set():pause_event.clear();paused=not paused;anchor=time.perf_counter()-sim_time
                        if not paused and status=='running':
                            desired=time.perf_counter()-anchor;deadline=time.perf_counter()+.012
                            while status=='running' and sim_time<desired and time.perf_counter()<deadline:step()
                            if status!='running':
                                print(f'Episode {status} at {sim_time:.2f}s; resetting.',flush=True)
                                reset();anchor=time.perf_counter()
                        origin=env.scene.env_origins[0].cpu().numpy()
                        data.qpos[:]=env.sim.data.qpos[0].cpu().numpy()
                        for address in roots:data.qpos[address:address+3]-=origin
                        data.qvel[:]=env.sim.data.qvel[0].cpu().numpy()
                        data.eq_active[:]=env.sim.data.eq_active[0].cpu().numpy()
                        mujoco.mj_forward(model,data)
                        pose=s.capture_geometry();loads=(s.utilization[0]*cfg.break_force).cpu().tolist()
                        connected=s.connected()[0].cpu().tolist();phase=PHASES[int(s.fsm.phase[0])]
                        labels='Mode\nCheckpoint\nTime / physics dt\nPhase\nCapture gap / angle\nFace loads 1-6 (N)\nConnected 1-6\nPeak other (N)\nStatus\nKeys'
                        values=(f'RSL policy, deterministic\n{checkpoint.name}\n{sim_time:.2f}s / 2ms\n{phase}\n'
                                f'{1000*float(pose["gap"][0]):.2f}mm / {float(pose["angle"][0])*180/3.141592653589793:.2f}deg\n'
                                +' '.join(f'{v:.1f}' for v in loads)+'\n'
                                +' '.join('1' if v else '0' for v in connected)+'\n'
                                +f'{float(s.max_other_load[0]):.1f}\n'+('paused' if paused else status)+'\nEnter reset, Space pause')
                        overlay.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,labels,values))
                        viewer.sync();time.sleep(.002)
                        if not announced:print('GPU closure viewer ready; episodes reset automatically; Enter resets, Space pauses.',flush=True);announced=True
                    overlay.close()
            print('GPU_CLOSURE_VIEW',json.dumps({'checkpoint':str(checkpoint),'mode':'rsl_policy_deterministic',
                'num_envs':1,'time_s':sim_time,'status':status,'physics_dt_s':cfg.sim.mujoco.timestep,
                'peak_other_N':float(s.max_other_load[0])}),flush=True)
            return 1 if status=='failure' else 0
    finally:
        env.close();torch.cuda.set_stream(previous)


if __name__=='__main__':raise SystemExit(main())
