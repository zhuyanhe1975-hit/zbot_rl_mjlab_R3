"""Small CURL-only residual PPO on the refined GPU reference controller.

This is an experimental pilot, not an assertion of learned full-task success.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import warp as wp
from torch import nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from zbot_rl_mjlab.full_student_logging import format_rsl_log
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_refined_teacher_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv

MODE='refined_gpu_curl_residual_ppo_v1'
REFERENCE=Path('assets/reconfiguration/refined_forward_velocity_reference.npz')


class ActorCritic(nn.Module):
    def __init__(self,obs_dim,action_dim=6,action_std=.1):
        super().__init__()
        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,out))
        self.actor=mlp(action_dim);self.critic=mlp(1);self.action_std=float(action_std)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    def distribution(self,obs):
        mean=self.actor(obs)
        return Normal(mean,torch.full_like(mean,self.action_std))


def make_pilot_cfg(num_envs,reference=REFERENCE):
    cfg=make_refined_teacher_cfg(num_envs,reference,.2)
    cfg.auto_reset=True
    cfg.full_velocity_scale=.02
    cfg.full_torque_scale=0.
    cfg.full_policy_curl_only=True
    cfg.full_curl_offset_limit=.02
    # The old tracking loss compares to unscaled qvel, incompatible with the
    # 0.2x reference; progress, time, safety and terminal rewards remain.
    cfg.rewards.pop('penalty_tracking',None)
    return cfg


def make_initial_closure_cfg(num_envs,reference=REFERENCE):
    from zbot_rl_mjlab.tasks.reconfiguration.initial_closure_pilot import configure_initial_closure
    cfg=configure_initial_closure(make_pilot_cfg(num_envs,reference))
    cfg.sim.mujoco.timestep=.002
    cfg.decimation=10
    cfg.full_curl_clock_fraction=.5
    cfg.full_velocity_scale=.2
    cfg.full_curl_offset_limit=.15
    return cfg


def make_online_initial_closure_cfg(num_envs):
    """Experimental real-time planner; no trajectory file is read or sampled."""
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_scene import add_online_headtail_connectors
    from zbot_rl_mjlab.tasks.reconfiguration.initial_closure_pilot import configure_initial_closure
    cfg=make_full_batched_env_cfg(num_envs,None,0.,0.)
    cfg.full_refined_reference=True;cfg.scene.spec_fn=add_online_headtail_connectors
    cfg.auto_reset=True;cfg.break_force=100.;cfg.minimum_other_margin=.2;cfg.full_enforce_other_margin=True
    cfg.full_outer_stiffness=80.;cfg.full_outer_damping=4.;cfg.full_inner_stiffness=400.;cfg.full_inner_damping=4.
    cfg.full_reference_rate=1.;cfg.full_teacher_load_replay=False;cfg.full_teacher_headtail_replay=False
    cfg.full_teacher_separation_position=False;cfg.full_final_angle_tolerance=.04;cfg.full_terminal_recheck=True
    cfg.full_policy_curl_only=True;cfg.full_curl_offset_limit=.15;cfg.full_velocity_scale=.2;cfg.full_torque_scale=0.
    cfg.full_online_headtail=True
    configure_initial_closure(cfg)
    cfg.sim.mujoco.timestep=.002;cfg.decimation=10
    return cfg


def policy_action(raw,phase,baseline=None):
    """Mask policy influence outside CURL; a seventh value controls tempo."""
    enabled=(phase==P['CURL']) if baseline is None else ((phase==P['CURL'])&~baseline)
    enabled=enabled[:,None]
    velocity=torch.where(enabled,torch.tanh(raw[:,:6]),torch.zeros_like(raw[:,:6]))
    clock=torch.where(enabled,torch.tanh(raw[:,6:7]),torch.zeros_like(raw[:,6:7])) if raw.shape[-1]==7 else torch.zeros_like(velocity[:,:1])
    return torch.cat((velocity,clock),-1)


def episode_outcomes(done,success,baseline):
    failed=done&~success
    return failed,success&baseline,success&~baseline,failed&baseline,failed&~baseline


def main(argv=None, *, initial_closure=False):
    mode='refined_gpu_initial_closure_progress_ppo_v3' if initial_closure else MODE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--rollout-steps',type=int,default=64 if initial_closure else 128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--log-dir',type=Path,default=Path('logs/refined_gpu_initial_closure_ppo_v3' if initial_closure else 'logs/refined_gpu_curl_pilot'))
    parser.add_argument('--resume',type=Path)
    if initial_closure:parser.add_argument('--baseline-envs',type=int,default=64)
    args=parser.parse_args(argv)
    if args.num_envs<1 or args.iterations<1 or args.rollout_steps<2:
        parser.error('num-envs and iterations must be positive; rollout-steps must be at least 2')
    if initial_closure and not 0<=args.baseline_envs<args.num_envs:
        parser.error('baseline-envs must be nonnegative and smaller than num-envs')
    if args.log_dir.exists() and any(args.log_dir.iterdir()) and args.resume is None:
        parser.error('log directory is not empty; choose a new --log-dir or specify --resume')
    reference_hash=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    cfg=make_initial_closure_cfg(args.num_envs) if initial_closure else make_pilot_cfg(args.num_envs)
    cfg.seed=args.seed
    env=FullBatchedGpuEnv(cfg,device='cuda:0');previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    writer=None
    try:
        with torch.no_grad():obs,_=env.reset();x=obs['actor']
        action_dim=7 if initial_closure else 6;action_std=.4 if initial_closure else .1
        net=ActorCritic(x.shape[-1],action_dim,action_std).to(env.device);optimizer=torch.optim.Adam(net.parameters(),lr=3e-4 if initial_closure else 1e-4)
        start_iteration=0;total_steps=0;total_time=0.
        if args.resume:
            state=torch.load(args.resume,map_location=env.device,weights_only=False)
            if (state.get('mode')!=mode or state.get('reference_sha256')!=reference_hash
                or state.get('obs_dim')!=x.shape[-1] or state.get('num_envs')!=args.num_envs
                or (initial_closure and state.get('baseline_envs')!=args.baseline_envs)):
                raise ValueError('checkpoint is incompatible with this pilot or reference')
            net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            start_iteration=state['iteration']+1;total_steps=state['total_steps']
            total_time=state.get('total_time',0.)
        args.log_dir.mkdir(parents=True,exist_ok=True);writer=SummaryWriter(str(args.log_dir))
        print('CONFIG',json.dumps({'mode':mode,'num_envs':args.num_envs,'physics_dt_s':cfg.sim.mujoco.timestep,
              'control_dt_s':env.step_dt,'reference_rate':.2,'phase':'CURL','residual_speed_limit_rad_s':cfg.full_velocity_scale,
              'residual_offset_limit_rad':cfg.full_curl_offset_limit,'action_std':action_std,'break_force_N':100.,'other_limit_N':80.,
              'clock_fraction':.5 if initial_closure else 0.,'baseline_envs':args.baseline_envs if initial_closure else 0,
              'reference_sha256':reference_hash,'full_4096_baseline_validated':False}),flush=True)
        episode_steps=torch.zeros(args.num_envs,device=env.device,dtype=torch.long)
        episode_return=torch.zeros(args.num_envs,device=env.device)
        baseline=(torch.arange(args.num_envs,device=env.device)<args.baseline_envs) if initial_closure else torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
        pose_term=env.reward_manager.active_terms.index('pose_nearness') if initial_closure else None
        progress_term=env.reward_manager.active_terms.index('pose_progress') if initial_closure else None
        final_iteration=start_iteration+args.iterations;session_time=0.
        for iteration in range(start_iteration,start_iteration+args.iterations):
            start=time.perf_counter();rows=[];success_total=torch.zeros((),device=env.device,dtype=torch.long);success_steps=torch.zeros_like(success_total)
            failure_total=torch.zeros_like(success_total);completed_steps=torch.zeros_like(success_total)
            completed_return=torch.zeros((),device=env.device)
            timeout_total=torch.zeros_like(success_total)
            reference_success=torch.zeros_like(success_total);reference_steps=torch.zeros_like(success_total);reference_failure=torch.zeros_like(success_total)
            policy_success=torch.zeros_like(success_total);policy_steps=torch.zeros_like(success_total);policy_failure=torch.zeros_like(success_total)
            residual_sum=torch.zeros((),device=env.device);clock_sum=torch.zeros_like(residual_sum);active_total=torch.zeros_like(success_total)
            collection_start=torch.cuda.Event(enable_timing=True);collection_end=torch.cuda.Event(enable_timing=True);learning_end=torch.cuda.Event(enable_timing=True)
            collection_start.record()
            pose_rate_sum=torch.zeros((),device=env.device);progress_rate_sum=torch.zeros_like(pose_rate_sum)
            env.interval_peak_loads.zero_();env.interval_peak_other.zero_();env.interval_invalid_load_world_steps.zero_()
            for _ in range(args.rollout_steps):
                with torch.no_grad():
                    current=x.clone();dist=net.distribution(current);raw=dist.sample();logp=dist.log_prob(raw).sum(-1)
                    value=net.critic(current).squeeze(-1);phase=env._disconnect_state.fsm.phase.clone()
                    active=(phase==P['CURL'])&~baseline;action=policy_action(raw,phase,baseline)
                    obs,reward,terminated,truncated,_=env.step(action);done=terminated|truncated
                    if pose_term is not None:pose_rate_sum+=env.reward_manager._step_reward[:,pose_term].sum()
                    if progress_term is not None:progress_rate_sum+=env.reward_manager._step_reward[:,progress_term].sum()
                    success=env.termination_manager.get_term('success').clone()
                    failed,ref_ok,policy_ok,ref_fail,policy_fail=episode_outcomes(done,success,baseline)
                    failure_total+=failed.sum();reference_failure+=ref_fail.sum();policy_failure+=policy_fail.sum()
                    timeout_total+=env.termination_manager.get_term('phase_timeout').sum()
                    episode_steps+=1;success_steps+=torch.where(success,episode_steps,0).sum();completed_steps+=torch.where(done,episode_steps,0).sum()
                    reference_success+=ref_ok.sum();reference_steps+=torch.where(ref_ok,episode_steps,0).sum()
                    policy_success+=policy_ok.sum();policy_steps+=torch.where(policy_ok,episode_steps,0).sum()
                    episode_steps.masked_fill_(done,0)
                    episode_return+=reward;completed_return+=torch.where(done,episode_return,0.).sum();episode_return.masked_fill_(done,0.)
                    active &= (env._disconnect_state.fsm.phase==P['CURL'])|failed
                    if initial_closure:
                        action_term=env.action_manager.get_term('joint_torque')
                        residual_sum+=(action_term.curl_residual_velocity.abs().mean(-1)*active).sum()
                        clock_sum+=((1.+cfg.full_curl_clock_fraction*action_term.policy[:,6])*active).sum();active_total+=active.sum()
                    rows.append((current,raw,logp,value,reward.clone(),done.clone(),active));x=obs['actor']
                    success_total+=success.sum()
                    # Failed episodes are normal training data and are reset
                    # by the GPU mask. Non-finite loads still abort the batch.
                    if not initial_closure and bool(failed.any()):break
                    if (not initial_closure or len(rows)%16==0) and bool(env.interval_invalid_load_world_steps):break
            collection_end.record();steps=len(rows);total_steps+=steps*args.num_envs
            success_count=int(success_total);failure_count=int(failure_total);done_count=success_count+failure_count
            if int(env.interval_invalid_load_world_steps) or (not initial_closure and failure_count):
                stopped={'iteration':iteration,'transitions':steps*args.num_envs,
                      'success':success_count,'failure':failure_count,
                      'invalid_load_world_steps':int(env.interval_invalid_load_world_steps),
                      'peak_other_N':float(env.interval_peak_other),
                      'reason':'invalid or failed rollout; no update from this batch'}
                (args.log_dir/'stop.json').write_text(json.dumps(stopped,indent=2)+'\n')
                print('STOP',json.dumps(stopped),flush=True)
                return 1
            with torch.no_grad():bootstrap=net.critic(x).squeeze(-1);gae=torch.zeros(args.num_envs,device=env.device);advantages=[]
            for i in reversed(range(steps)):
                _,_,_,value,reward,done,_=rows[i];next_value=bootstrap if i==steps-1 else rows[i+1][3]
                keep=(~done).float();gae_lambda=.99 if initial_closure else .95
                gae=reward+.995*next_value*keep-value+.995*gae_lambda*keep*gae;advantages.append(gae)
            advantage=torch.stack(advantages[::-1]);values=torch.stack([row[3] for row in rows]);returns=advantage+values
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            X=torch.stack([row[0] for row in rows]).flatten(0,1);A=torch.stack([row[1] for row in rows]).flatten(0,1)
            old_logp=torch.stack([row[2] for row in rows]).flatten();G=advantage.flatten();R=returns.flatten()
            mask=torch.stack([row[6] for row in rows]).flatten();loss_sums=torch.zeros(2,device=env.device);updates=0
            for _ in range(2):
                for ids in torch.randperm(len(X),device=env.device).split(16384):
                    dist=net.distribution(X[ids]);ratio=(dist.log_prob(A[ids]).sum(-1)-old_logp[ids]).exp()
                    weights=mask[ids].float();denom=weights.sum().clamp_min(1.)
                    low,high=(.8,1.2) if initial_closure else (.9,1.1)
                    actor_loss=-(torch.minimum(ratio*G[ids],ratio.clamp(low,high)*G[ids])*weights).sum()/denom
                    value_loss=(net.critic(X[ids]).squeeze(-1)-R[ids]).square().mean()
                    loss=actor_loss+.5*value_loss;optimizer.zero_grad();loss.backward()
                    if not bool(mask[ids].any()):
                        for parameter in net.actor.parameters():parameter.grad=None
                    nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
                    loss_sums+=torch.stack((actor_loss.detach(),value_loss.detach()));updates+=1
            learning_end.record();torch.cuda.synchronize();wall=time.perf_counter()-start
            total_time+=wall;session_time+=wall;done_iterations=iteration-start_iteration+1
            metrics={'iteration':iteration,'total_steps':total_steps,'num_envs':args.num_envs,'rollout_steps':steps,
                     'policy_samples':int(mask.sum()),'success':success_count,'failure':failure_count,
                     'mean_success_time_s':float(success_steps)*env.step_dt/success_count if success_count else None,
                     'done':done_count,'mean_episode_reward':float(completed_return)/done_count if done_count else None,
                     'mean_episode_length':float(completed_steps)/done_count if done_count else None,
                     'final_iteration':final_iteration,'phase_timeout':int(timeout_total),
                     'collection_stream_s':collection_start.elapsed_time(collection_end)/1000.,
                     'learning_stream_s':collection_end.elapsed_time(learning_end)/1000.,
                     'surrogate_loss':float(loss_sums[0]/updates),'entropy':action_dim*(.5*math.log(2*math.pi*math.e)+math.log(action_std)),
                     'action_std':action_std,'invalid_load_world_steps':int(env.interval_invalid_load_world_steps),
                     'elapsed_s':total_time,'eta_s':session_time/done_iterations*(final_iteration-iteration-1),
                     'peak_other_N':float(env.interval_peak_other),'peak_connector_loads_N':env.interval_peak_loads.tolist(),
                     'actor_loss':float(loss_sums[0]/updates),'value_loss':float(loss_sums[1]/updates),
                     'wall_s':wall,'fps':steps*args.num_envs/wall,'checkpoint_status':'candidate_not_full_task_validated'}
            if pose_term is not None:
                metrics['mean_pose_reward_rate']=float(pose_rate_sum)/(steps*args.num_envs)
                metrics['mean_pose_progress_rate']=float(progress_rate_sum)/(steps*args.num_envs)
                n_reference=int(reference_success);n_policy=int(policy_success);n_active=int(active_total)
                metrics.update(reference_success=n_reference,policy_success=n_policy,
                    reference_failure=int(reference_failure),policy_failure=int(policy_failure),
                    mean_reference_time_s=float(reference_steps)*env.step_dt/n_reference if n_reference else None,
                    mean_policy_time_s=float(policy_steps)*env.step_dt/n_policy if n_policy else None,
                    mean_abs_curl_residual_rad_s=float(residual_sum)/n_active if n_active else None,
                    mean_curl_clock_scale=float(clock_sum)/n_active if n_active else None)
            with (args.log_dir/'metrics.jsonl').open('a') as output:output.write(json.dumps(metrics)+'\n')
            for key in ('policy_samples','success','peak_other_N','actor_loss','value_loss','wall_s','fps'):
                writer.add_scalar(key,metrics[key],iteration)
            if pose_term is not None:writer.add_scalar('mean_pose_reward_rate',metrics['mean_pose_reward_rate'],iteration)
            if progress_term is not None:writer.add_scalar('mean_pose_progress_rate',metrics['mean_pose_progress_rate'],iteration)
            if initial_closure:
                for key in ('mean_reference_time_s','mean_policy_time_s','mean_abs_curl_residual_rad_s','mean_curl_clock_scale'):
                    if metrics[key] is not None:writer.add_scalar(key,metrics[key],iteration)
            candidate={'mode':mode,'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                       'obs_dim':x.shape[-1],'num_envs':args.num_envs,'reference_sha256':reference_hash,
                       'iteration':iteration,'total_steps':total_steps,'total_time':total_time,'baseline_envs':args.baseline_envs if initial_closure else 0,'config':{'reference_rate':.2,
                       'residual_speed_rad_s':cfg.full_velocity_scale,'residual_offset_rad':cfg.full_curl_offset_limit,'clock_fraction':cfg.full_curl_clock_fraction if initial_closure else 0.,'action_std':action_std,'phase':'CURL'},
                       'validation':'candidate_not_full_task_validated'}
            tmp=args.log_dir/'latest.pt.tmp';torch.save(candidate,tmp);tmp.replace(args.log_dir/'latest.pt')
            print(format_rsl_log(metrics),flush=True)
        return 0
    finally:
        if writer is not None:writer.close()
        env.close();torch.cuda.set_stream(previous)


if __name__=='__main__':raise SystemExit(main())
