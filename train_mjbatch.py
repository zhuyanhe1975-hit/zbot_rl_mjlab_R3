"""Train standalone PPO for Zbot head-tail closure using mjbatch.Batch."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from zbot_mjbatch.env import HeadTailBatch
from zbot_mjbatch.mpc import PredictiveController

MODE='zbot_mjbatch_headtail_no_collision_ppo_v4'
ROOT=Path(__file__).resolve().parent

def signature():
    files=('src/zbot_mjbatch/model.py','src/zbot_mjbatch/env.py','src/zbot_mjbatch/mpc.py','src/zbot_mjbatch/native/monitor.cpp',
           'assets/zbot_6/ZBOT6.xml','assets/zbot_6/meshes/a.STL','assets/zbot_6/meshes/b.STL',
           'assets/reconfiguration/closed_loop_start.json','assets/reconfiguration/full_straight_start.json')
    h=hashlib.sha256()
    for name in files:h.update((ROOT/name).read_bytes())
    return h.hexdigest()

def available_cpu_count():
    """Respect the process CPU affinity when choosing the simulation pool."""
    try:return max(1,len(os.sched_getaffinity(0)))
    except (AttributeError,OSError):return max(1,os.cpu_count() or 1)

class Policy(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        def mlp(n):return nn.Sequential(nn.Linear(obs_dim,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,n))
        self.actor=mlp(7);self.critic=mlp(1);self.log_std=nn.Parameter(torch.full((7,),float(np.log(.25))))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    def distribution(self,obs):return Normal(self.actor(obs),self.log_std.clamp(-4.,-.5).exp())
    def value(self,obs):return self.critic(obs).squeeze(-1)


def evaluate_policy(net,num_envs,threads,device,candidates,horizon):
    """Independent first-episode deterministic evaluation, without learning."""
    env=HeadTailBatch(num_envs,threads);mpc=PredictiveController(num_envs,threads,candidates,horizon) if candidates else None;obs=env.observation()
    ended=np.zeros(num_envs,bool);success=np.zeros(num_envs,bool);duration=np.zeros(num_envs)
    peak=np.zeros(num_envs);torque=np.zeros(num_envs);reason=np.zeros(num_envs,dtype=int)
    try:
        for _ in range(round(env.max_time/env.control_dt)+1):
            with torch.no_grad():actions=torch.tanh(net.actor(torch.as_tensor(obs,device=device))).cpu().numpy()
            if mpc is not None:actions,_=mpc.select(env,actions)
            obs,_,done,info=env.step(actions);first=done&~ended
            success[first]=info['success'][first];duration[first]=info['duration_s'][first]
            peak[first]=info['peak_load_N'][first];torque[first]=info['peak_torque_Nm'][first]
            reason[first]=info['reason'][first];ended|=first
            if ended.all():break
        return {'worlds':num_envs,'successes':int(success.sum()),'failures':int((ended&~success).sum()),
                'unfinished':int((~ended).sum()),'mean_success_time_s':float(duration[success].mean()) if success.any() else None,
                'peak_other_load_N':float(peak.max()),'peak_joint_torque_Nm':float(torque.max()),'reasons':reason.tolist(),
                'limits':'fixed initial state; collision detection disabled; no perturbation or real-world validation'}
    finally:
        if mpc is not None:mpc.close()
        env.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs',type=int,default=256);p.add_argument('--threads',type=int,default=available_cpu_count())
    p.add_argument('--iterations',type=int,default=200);p.add_argument('--horizon',type=int,default=24)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--log-dir',type=Path,default=Path('logs/mjbatch_headtail_no_collision'))
    p.add_argument('--resume',type=Path);p.add_argument('--save-interval',type=int,default=10)
    p.add_argument('--eval-interval',type=int,default=50);p.add_argument('--eval-envs',type=int,default=4)
    p.add_argument('--mpc-candidates',type=int,default=0,help='0 disables experimental predictive sampling')
    p.add_argument('--mpc-horizon',type=int,default=2)
    args=p.parse_args(argv)
    if min(args.num_envs,args.threads,args.iterations,args.horizon,args.epochs,args.minibatches,args.save_interval)<1:p.error('positive sizes required')
    if args.eval_interval<0 or args.eval_envs<1:p.error('invalid evaluation settings')
    if args.mpc_candidates<0 or args.mpc_candidates==1 or args.mpc_horizon<1:p.error('MPC requires zero or at least two candidates and a positive horizon')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    if args.log_dir.exists() and any(args.log_dir.iterdir()) and not args.resume:
        stamp=time.strftime('%Y%m%d-%H%M%S');previous=args.log_dir.with_name(args.log_dir.name+'_archived_'+stamp)
        if previous.exists():previous=previous.with_name(previous.name+f'-{time.time_ns()}')
        args.log_dir.rename(previous);print('PRESERVED',previous,flush=True)
    args.log_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    env=HeadTailBatch(args.num_envs,args.threads,args.seed)
    mpc=PredictiveController(args.num_envs,args.threads,args.mpc_candidates,args.mpc_horizon) if args.mpc_candidates else None
    try:
        obs=env.observation();net=Policy(obs.shape[1]).to(args.device);optim=torch.optim.Adam(net.parameters(),lr=3e-4)
        start_iteration=0;sig=signature()
        if args.resume:
            saved=torch.load(args.resume,map_location=args.device,weights_only=False)
            if saved['mode']!=MODE or saved['signature']!=sig or saved['obs_dim']!=obs.shape[1]:raise ValueError('incompatible mjbatch checkpoint')
            prior=saved['config']['training']
            if prior['mpc_candidates']!=args.mpc_candidates or (args.mpc_candidates and prior['mpc_horizon']!=args.mpc_horizon):
                raise ValueError('MPC settings differ from the checkpoint')
            net.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer']);start_iteration=saved['iteration']+1
        config={'mode':MODE,'signature':sig,'num_envs':args.num_envs,'threads':args.threads,'obs_dim':obs.shape[1],
                'action_dim':7,'physics_dt_s':env.dt,'control_dt_s':env.control_dt,'episode_limit_s':env.max_time,
                'joint_speed_limit_rad_s':env.speed_limit,'joint_acceleration_limit_rad_s2':None,
                'joint_servo_kp':env.kp,'joint_servo_kd':env.kd,'joint_torque_limit_Nm':env.torque_limit,
                'collision_detection':'disabled','collision_velocity_projection':'disabled',
                'training':vars(args)|{'log_dir':str(args.log_dir),'resume':str(args.resume) if args.resume else None},
                'reward':{'time':'20 / duration on safe success','pose_nearness':'0.1 exp(-gap/.15-angle/.75) per second',
                          'failure':-20.,'elapsed_per_second':-.2},'backend':'mjbatch.Batch CPU, MuJoCo 3.14'}
        config_path=args.log_dir/('resume_config_'+str(time.time_ns())+'.json' if args.resume else 'run_config.json')
        config_path.write_text(json.dumps(config,indent=2)+'\n')
        print('TASK',json.dumps(config,ensure_ascii=False),flush=True)
        episode_return=np.zeros(args.num_envs);episode_pose=np.zeros(args.num_envs);episode_elapsed=np.zeros(args.num_envs)
        evaluation_path=args.log_dir/'deterministic_evaluation.json'
        evaluations=json.loads(evaluation_path.read_text()) if args.resume and evaluation_path.exists() else {'results':[],'selected':None}
        for iteration in range(start_iteration,start_iteration+args.iterations):
            t0=time.perf_counter();records=[];episodes=[];mpc_stats=[]
            for _ in range(args.horizon):
                x=torch.as_tensor(obs,device=args.device)
                with torch.no_grad():
                    dist=net.distribution(x);raw=dist.sample();logp=dist.log_prob(raw).sum(-1);value=net.value(x)
                proposed=torch.tanh(raw).cpu().numpy()
                if mpc is None:selected=proposed
                else:selected,stats=mpc.select(env,proposed);mpc_stats.append(stats)
                next_obs,reward,done,info=env.step(selected)
                episode_return+=reward;episode_pose+=info['reward_pose'];episode_elapsed+=info['penalty_elapsed']
                records.append((x,raw,logp,value,torch.as_tensor(reward,device=args.device),torch.as_tensor(done,device=args.device)))
                if done.any():
                    for i in np.flatnonzero(done):
                        item={k:float(v[i]) for k,v in info.items()}
                        item.update(episode_return=float(episode_return[i]),episode_pose=float(episode_pose[i]),
                                    episode_elapsed=float(episode_elapsed[i]));episodes.append(item)
                    episode_return[done]=0.;episode_pose[done]=0.;episode_elapsed[done]=0.
                obs=next_obs
            with torch.no_grad():bootstrap=net.value(torch.as_tensor(obs,device=args.device))
            advantages=[];gae=torch.zeros(args.num_envs,device=args.device)
            for i in range(len(records)-1,-1,-1):
                _,_,_,value,reward,done=records[i];nxt=bootstrap if i==len(records)-1 else records[i+1][3]
                delta=reward+.997*nxt*(~done)-value;gae=delta+.997*.97*(~done)*gae;advantages.append(gae)
            adv=torch.stack(advantages[::-1]);values=torch.stack([r[3] for r in records]);returns=adv+values
            X=torch.stack([r[0] for r in records]).reshape(-1,obs.shape[1]);A=torch.stack([r[1] for r in records]).reshape(-1,7)
            old=torch.stack([r[2] for r in records]).reshape(-1);adv=adv.flatten();returns=returns.flatten()
            mask=X[:,env.policy_active_observation_index]>.5
            if mask.any():
                selected=adv[mask];adv=torch.where(mask,(adv-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8),0.)
            else:adv.zero_()
            policy_losses=[];value_losses=[]
            for _ in range(args.epochs):
                for idx in torch.randperm(len(X),device=args.device).tensor_split(args.minibatches):
                    if not len(idx):continue
                    active=mask[idx];optim.zero_grad(set_to_none=True)
                    value_loss=(net.value(X[idx])-returns[idx]).square().mean();loss=.5*value_loss
                    if active.any():
                        ratio=(net.distribution(X[idx]).log_prob(A[idx]).sum(-1)-old[idx]).exp()
                        policy=-torch.minimum(ratio*adv[idx],ratio.clamp(.9,1.1)*adv[idx])[active].mean()
                        loss=loss+policy;policy_losses.append(float(policy.detach()))
                    loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step();value_losses.append(float(value_loss.detach()))
            elapsed=time.perf_counter()-t0;n=len(episodes);s=sum(x['success'] for x in episodes)
            report={'iteration':iteration,'Perf/steps':args.num_envs*args.horizon,'Perf/wall_s':elapsed,
                    'Perf/steps_per_s':args.num_envs*args.horizon/elapsed,'Policy/active_fraction':float(mask.float().mean()),
                    'Loss/value':float(np.mean(value_losses)),'Loss/policy':float(np.mean(policy_losses)) if policy_losses else None,
                    'Metrics/episodes':n,'Metrics/success_fraction':s/n if n else None,'Metrics/failure_fraction':(n-s)/n if n else None,
                    'Metrics/success_duration_s':float(np.mean([x['duration_s'] for x in episodes if x['success']])) if s else None,
                    'Metrics/peak_other_load_N':max((x['peak_load_N'] for x in episodes),default=None),
                    'Metrics/peak_joint_torque_Nm':max((x['peak_torque_Nm'] for x in episodes),default=None),
                    'Metrics/load_guard_fraction':sum(x['reason']==1 for x in episodes)/n if n else None,
                    'Metrics/physics_warning_fraction':sum(x['reason']==7 for x in episodes)/n if n else None,
                    'Metrics/timeout_fraction':sum(x['reason']==5 for x in episodes)/n if n else None,
                    'Reward/time':float(np.mean([x['reward_time'] for x in episodes])) if n else None,
                    'Reward/pose_nearness':float(np.mean([x['episode_pose'] for x in episodes])) if n else None,
                    'Penalty/failure':float(np.mean([x['penalty_failure'] for x in episodes])) if n else None,
                    'Penalty/elapsed_time':float(np.mean([x['episode_elapsed'] for x in episodes])) if n else None,
                    'Reward/episode_return':float(np.mean([x['episode_return'] for x in episodes])) if n else None}
            if mpc_stats:
                report.update({'MPC/safe_fraction':float(np.mean([x['safe_fraction'] for x in mpc_stats])),
                               'MPC/nominal_fraction':float(np.mean([x['nominal_fraction'] for x in mpc_stats])),
                               'MPC/predicted_failure_fraction':float(np.mean([x['predicted_failure_fraction'] for x in mpc_stats]))})
            with (args.log_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            print('ITERATION',json.dumps(report,ensure_ascii=False),flush=True)
            should_eval=args.eval_interval>0 and ((iteration+1)%args.eval_interval==0 or iteration==start_iteration+args.iterations-1)
            if iteration%args.save_interval==0 or iteration==start_iteration+args.iterations-1 or should_eval:
                checkpoint={'mode':MODE,'signature':sig,'iteration':iteration,'obs_dim':obs.shape[1],
                            'model':net.state_dict(),'optimizer':optim.state_dict(),'config':config}
                path=args.log_dir/f'model_{iteration}.pt';torch.save(checkpoint,path)
                if should_eval:
                    result=evaluate_policy(net,args.eval_envs,min(args.threads,args.eval_envs),args.device,args.mpc_candidates,args.mpc_horizon)
                    result.update(iteration=iteration,checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    evaluations['results'].append(result)
                    passed=result['successes']==args.eval_envs and result['unfinished']==0
                    selected=evaluations['selected']
                    if passed and (selected is None or result['mean_success_time_s']<selected['mean_success_time_s']):
                        shutil.copyfile(path,args.log_dir/'best.pt');evaluations['selected']=result
                    evaluation_path.write_text(json.dumps(evaluations,indent=2)+'\n')
                    print('DETERMINISTIC_EVAL',json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if mpc is not None:mpc.close()
        env.close()

if __name__=='__main__':main()
