"""Measure initial-closure sampling without training or changing run logs."""
import argparse
import json
import time

import torch
import warp as wp

from scripts.train_refined_gpu_initial_closure import make_initial_closure_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=8192)
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--unpruned',action='store_true')
    args=parser.parse_args()
    if min(args.num_envs,args.steps,args.repeats)<1:
        parser.error('num-envs, steps and repeats must be positive')
    torch.set_num_threads(1)
    cfg=make_initial_closure_cfg(args.num_envs)
    cfg.full_initial_closure_collision_pruning=not args.unpruned
    env=FullBatchedGpuEnv(cfg,device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device)
    torch.cuda.set_stream(stream)
    try:
        times=[]
        with torch.inference_mode():
            action=torch.zeros((args.num_envs,7),device=env.device)
            for _ in range(args.repeats):
                env.reset()
                for _ in range(4):env.step(action)
                torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(args.steps):env.step(action)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
        mean=sum(times)/len(times)
        print('CLOSURE_SAMPLING_BENCHMARK',json.dumps({
            'num_envs':args.num_envs,'steps':args.steps,'pruned':not args.unpruned,
            'wall_seconds':times,'mean_seconds':mean,
            'transitions_per_second':args.num_envs*args.steps/mean,
            'scope':'zero-action early-episode sampling only; excludes PPO update and initialization'}))
    finally:
        env.close();torch.cuda.set_stream(previous)


if __name__=='__main__':main()
