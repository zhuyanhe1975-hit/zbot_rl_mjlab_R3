"""Evaluate or view the independent mjbatch head-tail closure task."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from train_mjbatch import MODE,Policy,signature
from zbot_mjbatch.env import HeadTailBatch
from zbot_mjbatch.mpc import PredictiveController

def select_checkpoint(log):
    """Prefer a verified best; otherwise allow inspecting the newest model."""
    log=Path(log);report=log/'deterministic_evaluation.json';best=log/'best.pt'
    reason=None
    if report.is_file():
        selected=json.loads(report.read_text()).get('selected')
        if selected is None:
            reason='尚无通过独立验证的检查点'
        elif best.is_file() and hashlib.sha256(best.read_bytes()).hexdigest()==selected['sha256']:
            return best,None
        else:
            reason='已验证的 best.pt 缺失或校验不匹配'
    files=[x for x in log.glob('model_*.pt') if x.stem[6:].isdigit()]
    if not files:raise FileNotFoundError('没有检查点；请先训练，或使用 --zero-policy')
    newest=max(files,key=lambda x:int(x.stem[6:]))
    return newest,reason or '最新检查点尚未经过独立验证'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--zero-policy',action='store_true')
    p.add_argument('--headless',action='store_true');p.add_argument('--max-steps',type=int,default=1000)
    p.add_argument('--mpc-candidates',type=int,default=0);p.add_argument('--mpc-horizon',type=int,default=2)
    args=p.parse_args()
    if not args.checkpoint and not args.zero_policy:
        try:args.checkpoint,notice=select_checkpoint('logs/mjbatch_headtail_no_collision')
        except FileNotFoundError:
            args.zero_policy=True;notice=None
            print('回放提示：当前实验尚无检查点，使用零残差控制器。',flush=True)
        if notice:print(f'回放提示：{notice}；使用最新模型 {args.checkpoint}，其表现未获验证。',flush=True)
    env=HeadTailBatch(1,1);obs=env.observation();policy=None;mpc=None
    try:
        if args.checkpoint:
            checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            if checkpoint['mode']!=MODE or checkpoint['signature']!=signature() or checkpoint['obs_dim']!=obs.shape[1]:
                raise ValueError('checkpoint does not match the current mjbatch task')
            policy=Policy(obs.shape[1]);policy.load_state_dict(checkpoint['model']);policy.eval()
            args.mpc_candidates=checkpoint['config']['training']['mpc_candidates']
            args.mpc_horizon=checkpoint['config']['training']['mpc_horizon']
        if args.mpc_candidates:mpc=PredictiveController(1,1,args.mpc_candidates,args.mpc_horizon)
        viewer=None
        if not args.headless:
            import mujoco.viewer
            data=mujoco.MjData(env.model)
            viewer=mujoco.viewer.launch_passive(env.model,data)
        try:
            for step in range(args.max_steps):
                started=time.perf_counter()
                with torch.inference_mode():action=np.zeros((1,7)) if policy is None else torch.tanh(policy.actor(torch.from_numpy(obs))).numpy()
                if mpc is not None:action,_=mpc.select(env,action)
                obs,reward,done,info=env.step(action)
                if viewer is not None:
                    with viewer.lock():
                        mujoco.mj_setState(env.model,data,env.state[0],mujoco.mjtState.mjSTATE_INTEGRATION)
                        mujoco.mj_forward(env.model,data)
                    viewer.sync()
                    if not viewer.is_running():break
                    time.sleep(max(0.,env.control_dt-(time.perf_counter()-started)))
                if done[0]:
                    result={k:float(v[0]) for k,v in info.items()};result['steps']=step+1
                    print('MJBATCH_EVAL',json.dumps(result),flush=True)
                    return 0 if info['success'][0] else 1
            print('MJBATCH_EVAL',json.dumps({'status':'unfinished','steps':step+1}),flush=True)
            return 2
        finally:
            if viewer is not None:viewer.close()
    finally:
        if mpc is not None:mpc.close()
        env.close()

if __name__=='__main__':raise SystemExit(main())
