"""Evaluate a full-FSM residual checkpoint with deterministic policy actions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import torch
from train_full_student import ActorCritic
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import FullCpuStudent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--max-time',type=float,default=120.)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/full_cpu_student/evaluation.json'))
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--headless',dest='viewer',action='store_false')
    p.set_defaults(viewer=False)
    a=p.parse_args()
    if a.max_time<=0:p.error('--max-time must be positive')
    torch.set_num_threads(1);ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    env=FullCpuStudent(ckpt['residual_scale'],ckpt['reference']);obs=env.observation()
    if len(obs)!=ckpt['obs_dim']:
        env.close();raise ValueError('checkpoint observation mismatch')
    net=ActorCritic(len(obs));net.load_state_dict(ckpt['model']);net.eval();start=time.perf_counter();rewards=0.
    def step_policy():
        nonlocal obs,rewards
        with torch.no_grad():action=torch.tanh(net.actor(torch.from_numpy(obs)[None]))[0].numpy()
        obs,reward,_,_=env.step(action);rewards+=reward
    try:
        if a.viewer:
            import mujoco
            import mujoco.viewer
            f=env.fsm
            with mujoco.viewer.launch_passive(f.model,f.data) as viewer:
                viewer.cam.lookat[:]=[0.,0.,.08];viewer.cam.distance=1.4;viewer.cam.elevation=-55
                anchor_wall=time.perf_counter();anchor_sim=float(f.data.time);next_status=0.
                while viewer.is_running():
                    if f.phase not in ('DONE','FAILED') and f.data.time<a.max_time:
                        desired=anchor_sim+(time.perf_counter()-anchor_wall)
                        deadline=time.perf_counter()+.012
                        while f.data.time<desired and time.perf_counter()<deadline and f.phase not in ('DONE','FAILED') and f.data.time<a.max_time:
                            step_policy()
                    if time.perf_counter()>=next_status:
                        labels='Time\nPhase\nTarget\nFace loads 1-6 (N)\nPeak other (N)\nConnected 1-6'
                        values=(f'{f.data.time:.2f}s\n{f.phase}\n{f.target+1}\n'
                                +' '.join(f'{x:.1f}' for x in f.loads)+'\n'
                                +f'{f.peak_other:.1f}\n'+' '.join('1' if x else '0' for x in f.connected()))
                        if f.failure:labels+='\nFailure';values+='\n'+f.failure[:120]
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100.value,
                                          mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,labels,values))
                        next_status=time.perf_counter()+.1
                    viewer.sync();time.sleep(.002)
        else:
            while env.fsm.phase not in ('DONE','FAILED') and env.fsm.data.time<a.max_time:step_policy()
        report=env.fsm.report();report.update(checkpoint=str(a.checkpoint),episode_return=rewards,wall_time_s=time.perf_counter()-start)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
        print('EVALUATION',json.dumps({'success':report['success'],'terminal':report['terminal'],'report':str(a.output)}),flush=True)
        return 0 if a.viewer or report['success'] else 1
    finally:env.close()
if __name__=='__main__':raise SystemExit(main())
