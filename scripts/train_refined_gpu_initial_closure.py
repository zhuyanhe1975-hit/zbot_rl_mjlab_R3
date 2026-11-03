"""Bounded, control-phase PPO for safe and fast initial head-tail closure."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import warp as wp
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper

from scripts.train_refined_gpu_incremental import make_online_initial_closure_cfg as make_initial_closure_cfg
from zbot_rl_mjlab.experiments.common import ppo_cfg
from zbot_rl_mjlab.initial_closure_logging import InitialClosureLogger
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv

MODE='initial_closure_rsl_online_jacobian_2ms_v11'
DEFAULT_LOG_DIR=Path('logs/refined_gpu_initial_closure_rsl_2ms_online_jacobian')

def planner_signature():
    root=Path(__file__).resolve().parents[1]
    paths=('src/zbot_rl_mjlab/tasks/reconfiguration/online_headtail_planner.py',
           'src/zbot_rl_mjlab/tasks/reconfiguration/full_batched_action.py',
           'src/zbot_rl_mjlab/tasks/reconfiguration/full_batched_scene.py',
           'assets/reconfiguration/full_straight_start.json','assets/zbot_6/ZBOT6.xml',
           'assets/zbot_6/meshes/a.STL','assets/zbot_6/meshes/b.STL')
    digest=hashlib.sha256()
    for path in paths:digest.update((root/path).read_bytes())
    return digest.hexdigest()


class InitialClosureRunner(MjlabOnPolicyRunner):
    """Keep this task's native RSL checkpoints separate from other experiments."""
    def __init__(self,*args,reference_sha256,evaluation_interval=50,**kwargs):
        self.reference_sha256=reference_sha256
        self.evaluation_interval=evaluation_interval
        self._validator=None;self._final_iteration=None;self._last_validated_iteration=None
        super().__init__(*args,**kwargs)
        self.logger=InitialClosureLogger(log_dir=self.logger.log_dir,cfg=self.cfg,
            env_cfg=self.env.cfg,num_envs=self.env.num_envs,is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,gpu_global_rank=self.gpu_global_rank,
            device=self.device,step_dt=self.env.unwrapped.step_dt)
        # Distribution initializes a zero-mean, bounded residual policy.

    def save(self,path,infos=None):
        infos={**(infos or {}),'task_mode':MODE,'planner_sha256':self.reference_sha256}
        result=super().save(path,infos=infos)
        iteration=self.current_learning_iteration
        if (self.logger.log_dir and self.evaluation_interval>0
                and self._last_validated_iteration!=iteration
                and (iteration%self.evaluation_interval==0 or iteration==self._final_iteration)):
            from zbot_rl_mjlab.closure_validation import ClosureValidator
            if self._validator is None:
                self._validator=ClosureValidator(self,make_initial_closure_cfg,make_agent_cfg)
            self._validator.check(path)
            self._last_validated_iteration=iteration
        return result

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        self._final_iteration=self.current_learning_iteration+num_learning_iterations-1
        try:
            return super().learn(num_learning_iterations,init_at_random_ep_len=init_at_random_ep_len)
        finally:
            if self._validator is not None:self._validator.close()

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        infos=checkpoint.get('infos') or {}
        if infos.get('task_mode')!=MODE or infos.get('planner_sha256')!=self.reference_sha256:
            raise ValueError('checkpoint is not for online-jacobian v11 and this planner; start a fresh run')
        return super().load(path,load_cfg=load_cfg,strict=strict,map_location=map_location)


def make_agent_cfg(iterations):
    """Bounded residual policy and conservative, longer-horizon PPO."""
    cfg=ppo_cfg()
    cfg.experiment_name='refined_gpu_initial_closure_rsl_2ms_online_jacobian'
    cfg.actor.distribution_cfg={'class_name':'zbot_rl_mjlab.closure_policy:ClosureBetaDistribution',
                                'initial_concentration':6.}
    cfg.algorithm.class_name='zbot_rl_mjlab.closure_policy:ClosurePPO'
    cfg.algorithm.schedule='fixed'
    cfg.algorithm.learning_rate=3e-4
    cfg.algorithm.clip_param=.1
    cfg.algorithm.num_learning_epochs=3
    cfg.algorithm.entropy_coef=0.
    cfg.algorithm.gamma=.997
    cfg.algorithm.lam=.97
    cfg.save_interval=10
    cfg.max_iterations=iterations
    cfg.logger='tensorboard'
    cfg.upload_model=False
    return cfg


def prepare_fresh_log_dir(log_dir):
    """Preserve an earlier run before starting fresh at the same viewer path."""
    if not log_dir.exists() or not any(log_dir.iterdir()):
        return None
    stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    archived=log_dir.with_name(f'{log_dir.name}_archived_{stamp}')
    log_dir.rename(archived)
    return archived


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs','--env.scene.num-envs',type=int,default=4096)
    parser.add_argument('--iterations','--agent.max-iterations',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--log-dir',type=Path,default=DEFAULT_LOG_DIR)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--auto-resume',action='store_true',help='resume the latest checkpoint in the log directory if present')
    parser.add_argument('--eval-interval',type=int,default=50,help='independent validation interval; 0 disables')
    args=parser.parse_args(argv)
    if args.num_envs<1 or args.iterations<1:parser.error('num-envs and iterations must be positive')
    if args.eval_interval<0:parser.error('eval-interval must be nonnegative')
    if args.eval_interval and args.eval_interval%10:
        parser.error('eval-interval must be 0 or a multiple of the 10-iteration save interval')
    if args.auto_resume and args.resume is None and args.log_dir.exists():
        checkpoints=[p for p in args.log_dir.glob('model_*.pt') if p.stem[6:].isdigit()]
        if checkpoints:
            args.resume=max(checkpoints,key=lambda p:int(p.stem[6:]))
    if args.resume is None:
        archived=prepare_fresh_log_dir(args.log_dir)
        if archived is not None:
            print(f'Previous run preserved at {archived}; starting fresh in {args.log_dir}.',flush=True)
    reference_hash=planner_signature()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    cfg=make_initial_closure_cfg(args.num_envs);cfg.seed=args.seed
    agent=make_agent_cfg(args.iterations);agent.seed=args.seed
    env=FullBatchedGpuEnv(cfg,device='cuda:0');previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    try:
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        args.log_dir.mkdir(parents=True,exist_ok=True)
        runner=InitialClosureRunner(wrapped,asdict(agent),str(args.log_dir),device='cuda:0',
                                    reference_sha256=reference_hash,evaluation_interval=args.eval_interval)
        if args.resume:
            runner.load(str(args.resume),map_location='cuda:0')
            runner.current_learning_iteration+=1
            print(f'Resuming from {args.resume}; next iteration {runner.current_learning_iteration}.',flush=True)
        snapshot={'mode':MODE,'planner_sha256':reference_hash,'seed':args.seed,
            'agent':runner.cfg,'evaluation_interval':args.eval_interval,
            'resume_from':str(args.resume) if args.resume else None,
            'environment':{'num_envs':args.num_envs,'physics_dt':cfg.sim.mujoco.timestep,
                'decimation':cfg.decimation,'episode_length_s':cfg.episode_length_s,
                'observation_dim':runner.alg.get_policy().obs_dim,
                'headtail_alignment_position_tolerance_m':cfg.full_headtail_alignment_position_tolerance,
                'headtail_alignment_angle_tolerance_rad':cfg.full_headtail_alignment_angle_tolerance,
                'headtail_contact_sample_period_s':cfg.sim.mujoco.timestep,
                'other_load_guard_N':cfg.break_force*(1.-cfg.minimum_other_margin),
                'rewards':{name:{'weight':term.weight,'function':term.func.__module__+'.'+term.func.__name__}
                           for name,term in cfg.rewards.items()}}}
        config_path=args.log_dir/'run_config.json'
        if config_path.exists():
            config_path=args.log_dir/f'resume_config_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}.json'
        config_path.write_text(json.dumps(snapshot,indent=2)+'\n')
        print(f'Task: {MODE}; 2ms physics, 20ms control; {args.num_envs} GPU worlds.',flush=True)
        print('Failures receive a penalty and reset; no zero-policy baseline worlds are reserved.',flush=True)
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=False)
    finally:
        env.close();torch.cuda.set_stream(previous)


if __name__=='__main__':raise SystemExit(main())
