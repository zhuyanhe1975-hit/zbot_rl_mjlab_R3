"""Evaluate saved mean policies independently and select a successful best.pt."""
import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
import warp as wp
from mjlab.rl import RslRlVecEnvWrapper

from scripts.train_refined_gpu_initial_closure import (
    DEFAULT_LOG_DIR, InitialClosureRunner, make_agent_cfg, make_initial_closure_cfg, planner_signature,
)
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv


def select_successful_checkpoint(rows):
    """Require every evaluated world to succeed, then prefer shorter duration."""
    passed=[r for r in rows if r['successes']==r['worlds'] and r['unfinished']==0]
    return min(passed,key=lambda r:(r['mean_success_time_s'],-r['iteration'])) if passed else None


def evaluate(runner,wrapped,checkpoint,max_steps):
    runner.load(str(checkpoint),map_location=wrapped.device)
    policy=runner.get_inference_policy(device=wrapped.device)
    before={k:v.clone() for k,v in policy.obs_normalizer.state_dict().items()}
    obs,_=wrapped.reset()
    n=wrapped.num_envs
    ended=torch.zeros(n,dtype=torch.bool,device=wrapped.device)
    successful=torch.zeros_like(ended)
    duration=torch.zeros(n,device=wrapped.device)
    returns=torch.zeros_like(duration)
    term_names=wrapped.unwrapped.reward_manager.active_terms
    term_sums=torch.zeros((n,len(term_names)),device=wrapped.device)
    for step in range(max_steps):
        active=~ended
        obs,reward,done,_=wrapped.step(policy(obs))
        returns+=reward*active
        term_sums+=wrapped.unwrapped.reward_manager._step_reward*wrapped.unwrapped.step_dt*active[:,None]
        first_done=done.bool()&active
        successful[first_done]=wrapped.unwrapped.termination_manager.get_term('success')[first_done]
        duration[first_done]=(step+1)*wrapped.unwrapped.step_dt
        ended|=first_done
        if bool(ended.all()):break
    if not all(torch.equal(v,policy.obs_normalizer.state_dict()[k]) for k,v in before.items()):
        raise RuntimeError('evaluation changed observation normalization')
    return {'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'iteration':int(checkpoint.stem[6:]),'worlds':n,'successes':int(successful.sum()),
            'failures':int((ended&~successful).sum()),'unfinished':int((~ended).sum()),
            'mean_success_time_s':float(duration[successful].mean()) if bool(successful.any()) else None,
            'mean_return':float(returns.mean()),'normalizer_unchanged':True,
            'reward_components':dict(zip(term_names,term_sums.mean(0).tolist()))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir',type=Path,default=DEFAULT_LOG_DIR)
    parser.add_argument('--num-envs',type=int,default=16)
    args=parser.parse_args()
    if args.num_envs<1:parser.error('num-envs must be positive')
    checkpoints=sorted((p for p in args.log_dir.glob('model_*.pt') if p.stem[6:].isdigit()),
                       key=lambda p:int(p.stem[6:]))
    if not checkpoints:parser.error(f'no model checkpoints in {args.log_dir}; train first')
    torch.set_num_threads(1)
    env=FullBatchedGpuEnv(make_initial_closure_cfg(args.num_envs),device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    try:
        with torch.inference_mode():
            wrapped=RslRlVecEnvWrapper(env)
            runner=InitialClosureRunner(wrapped,asdict(make_agent_cfg(1)),None,device=env.device,
                                        reference_sha256=planner_signature())
            rows=[]
            for checkpoint in checkpoints:
                row=evaluate(runner,wrapped,checkpoint,round(env.cfg.episode_length_s/env.step_dt)+1)
                rows.append(row);print('DETERMINISTIC_EVAL',json.dumps(row),flush=True)
        selected=select_successful_checkpoint(rows)
        if selected:
            temp=args.log_dir/'best.pt.tmp'
            shutil.copyfile(selected['checkpoint'],temp);temp.replace(args.log_dir/'best.pt')
        report={'policy':'Bounded Beta mean, no sampling','results':rows,'selected':selected,
                'limits':'Fixed initial state only; this is not randomized robustness evaluation.'}
        temp=args.log_dir/'deterministic_evaluation.json.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n')
        temp.replace(args.log_dir/'deterministic_evaluation.json')
        print('Selected:',selected['checkpoint'] if selected else 'none; no policy passed every world',flush=True)
        return 0 if selected else 1
    finally:
        env.close();torch.cuda.set_stream(previous)


if __name__=='__main__':raise SystemExit(main())
