"""Experimental GPU PPO for full reconstruction, with RSL-style logging."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,hashlib,json,time
from pathlib import Path
import torch
import numpy as np
import warp as wp
from torch import nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from zbot_rl_mjlab.full_student_logging import format_rsl_log
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P

MODE='experimental_full_gpu_velocity_torque_v8'
REFINED_MODE='experimental_full_gpu_refined90_velocity_torque_v9'
DEFAULT_REFERENCE=Path('assets/reconfiguration/refined_forward_velocity_reference.npz')

def reference_identity(path):
    path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as data:
        refined='reference_version' in data and str(data['reference_version'])=='refined_forward_90_overlap_v1'
    return (REFINED_MODE if refined else MODE),digest

class ActorCritic(nn.Module):
    def __init__(self,n):
        super().__init__()
        def net(out):return nn.Sequential(nn.Linear(n,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,out))
        self.actor=net(7);self.critic=net(1);self.log_std=nn.Parameter(torch.full((7,),-2.))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    def dist(self,x):return Normal(self.actor(x),self.log_std.clamp(-5.,0.).exp())

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=1)
    p.add_argument('--rollout-steps',type=int,default=128);p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    p.add_argument('--velocity-scale',type=float,default=.5);p.add_argument('--torque-scale',type=float,default=2.);p.add_argument('--log-dir',type=Path,default=Path('logs/full_gpu_student_prototype'))
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',type=Path)
    a=p.parse_args(argv)
    if a.num_envs<=0 or a.iterations<=0 or a.rollout_steps<2 or not 0<a.velocity_scale<=3. or not 0<a.torque_scale<=20.:
        p.error('positive num-envs/iterations and rollout-steps >= 2 required')
    mode,reference_sha256=reference_identity(a.reference)
    torch.manual_seed(a.seed);cfg=make_full_batched_env_cfg(a.num_envs,a.reference,a.velocity_scale,a.torque_scale);cfg.seed=a.seed;a.log_dir.mkdir(parents=True,exist_ok=True)
    print('WARNING: Experimental full-FSM GPU PPO; complete reconstruction has not been validated. Console uses RSL-RL style; algorithm is custom PPO.',flush=True)
    env=FullBatchedGpuEnv(cfg,device='cuda:0');writer=None
    previous_stream=torch.cuda.current_stream(env.device)
    shared_stream=torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device)
    torch.cuda.set_stream(shared_stream)
    try:
        with torch.no_grad():obs,_=env.reset();x=obs['actor']
        net=ActorCritic(x.shape[-1]).to(env.device);opt=torch.optim.Adam(net.parameters(),lr=3e-4);n=a.num_envs;start_iteration=0
        total_steps=0;total_time=0.
        if a.resume:
            ckpt=torch.load(a.resume,map_location=env.device,weights_only=False)
            if ckpt.get('mode')!=mode or ckpt['obs_dim']!=x.shape[-1] or ckpt.get('reference')!=str(a.reference) or ckpt.get('velocity_scale')!=a.velocity_scale or ckpt.get('torque_scale')!=a.torque_scale or (mode==REFINED_MODE and ckpt.get('reference_sha256')!=reference_sha256):
                raise ValueError('incompatible full GPU student checkpoint')
            net.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer']);start_iteration=ckpt.get('iteration',-1)+1
            total_steps=ckpt.get('total_steps',0);total_time=ckpt.get('total_time',0.)
            if 'total_steps' not in ckpt:print('Previous checkpoint has no recorded step total; totals start from this resume.',flush=True)
        writer=SummaryWriter(log_dir=str(a.log_dir));episode_return=torch.zeros(n,device=env.device);episode_length=torch.zeros(n,device=env.device)
        final_iteration=start_iteration+a.iterations;session_time=0.
        for it in range(start_iteration,final_iteration):
            started=time.perf_counter();collection_start=torch.cuda.Event(enable_timing=True);collection_end=torch.cuda.Event(enable_timing=True);learning_end=torch.cuda.Event(enable_timing=True)
            collection_start.record();rows=[];ended=torch.zeros((),device=env.device,dtype=torch.long);succeeded=torch.zeros_like(ended);failed=torch.zeros_like(ended);timeouts=torch.zeros_like(ended)
            return_sum=torch.zeros((),device=env.device);length_sum=torch.zeros_like(return_sum);env.interval_peak_loads.zero_();env.interval_invalid_load_world_steps.zero_()
            for _ in range(a.rollout_steps):
                with torch.no_grad():
                    dist=net.dist(x);raw=dist.sample();logp=dist.log_prob(raw).sum(-1);value=net.critic(x).squeeze(-1)
                    phase=env._disconnect_state.fsm.phase
                    mask=(phase==P['CURL'])|(phase==P['CAPTURE_HEADTAIL'])|(phase==P['SEPARATE'])|(phase==P['ROTATE'])|(phase==P['APPROACH'])|(phase==P['UNFOLD'])|(phase==P['LOAD'])|(phase==P['LOAD_HEADTAIL'])
                    obs,reward,terminated,truncated,_=env.step(torch.tanh(raw));done=terminated|truncated
                    rows.append((x,raw,logp,value,reward.clone(),done.clone(),mask.clone()));x=obs['actor']
                    episode_return+=reward;episode_length+=1.;return_sum+=torch.where(done,episode_return,0.).sum();length_sum+=torch.where(done,episode_length,0.).sum()
                    episode_return.masked_fill_(done,0.);episode_length.masked_fill_(done,0.)
                    ended+=done.sum();succeeded+=env.termination_manager.get_term('success').sum();failed+=env.termination_manager.get_term('failure').sum();timeouts+=env.termination_manager.get_term('phase_timeout').sum()
            collection_end.record()
            with torch.no_grad():bootstrap=net.critic(x).squeeze(-1);gae=torch.zeros(n,device=env.device);advantages=[]
            for i in reversed(range(len(rows))):
                _,_,_,v,r,done,_=rows[i];nv=bootstrap if i==len(rows)-1 else rows[i+1][3];keep=(~done).float()
                gae=r+.995*nv*keep-v+.995*.95*keep*gae;advantages.append(gae)
            adv=torch.stack(advantages[::-1]);values=torch.stack([r[3] for r in rows]);returns=adv+values
            adv=(adv-adv.mean())/(adv.std()+1e-8);X=torch.stack([r[0] for r in rows]).flatten(0,1);A=torch.stack([r[1] for r in rows]).flatten(0,1)
            L=torch.stack([r[2] for r in rows]).flatten();G=adv.flatten();R=returns.flatten();M=torch.stack([r[6] for r in rows]).flatten()
            loss_sums=torch.zeros(3,device=env.device);updates=0
            for _ in range(2):
                for ids in torch.randperm(len(X),device=env.device).split(16384):
                    dist=net.dist(X[ids]);ratio=(dist.log_prob(A[ids]).sum(-1)-L[ids]).exp();weights=M[ids].float();denom=weights.sum().clamp_min(1.)
                    surrogate=-(torch.minimum(ratio*G[ids],ratio.clamp(.8,1.2)*G[ids])*weights).sum()/denom
                    value_loss=(net.critic(X[ids]).squeeze(-1)-R[ids]).square().mean();entropy=(dist.entropy().sum(-1)*weights).sum()/denom
                    loss=surrogate+.5*value_loss-.001*entropy
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    loss_sums+=torch.stack((value_loss.detach(),surrogate.detach(),entropy.detach()));updates+=1
            learning_end.record();torch.cuda.synchronize();wall=time.perf_counter()-started;session_time+=wall;total_time+=wall;total_steps+=n*a.rollout_steps
            count=int(ended);loss_values=(loss_sums/updates).tolist();peak_loads=env.interval_peak_loads.tolist();done_count=it-start_iteration+1
            report={'collection_stream_s':collection_start.elapsed_time(collection_end)/1000.,'learning_stream_s':collection_end.elapsed_time(learning_end)/1000.,
                'iteration':it,'final_iteration':final_iteration,'num_envs':n,'rollout_steps':a.rollout_steps,'transitions':n*a.rollout_steps,
                'total_steps':total_steps,'fps':n*a.rollout_steps/wall,'done':count,'success':int(succeeded),'failure':int(failed),'phase_timeout':int(timeouts),
                'mean_episode_reward':float(return_sum)/count if count else None,'mean_episode_length':float(length_sum)/count if count else None,
                'phase_counts':torch.bincount(env._disconnect_state.fsm.phase,minlength=len(P)).tolist(),'motion_samples':int(M.sum()),'wall_s':wall,
                'elapsed_s':total_time,'eta_s':session_time/done_count*(final_iteration-it-1),'value_loss':loss_values[0],'surrogate_loss':loss_values[1],
                'entropy':loss_values[2],'action_std':float(net.log_std.detach().clamp(-5.,0.).exp().mean()),'peak_connector_loads_N':peak_loads,
                'invalid_load_world_steps':int(env.interval_invalid_load_world_steps),'mode':mode}
            with (a.log_dir/'metrics.jsonl').open('a') as out:out.write(json.dumps(report)+'\n')
            for tag,value in {'Perf/total_fps':report['fps'],'Perf/collection_time':report['collection_stream_s'],'Perf/learning_time':report['learning_stream_s'],
                              'Loss/value_function':report['value_loss'],'Loss/surrogate':report['surrogate_loss'],'Loss/learning_rate':opt.param_groups[0]['lr'],
                              'Policy/mean_std':report['action_std'],'Episode_Termination/failure':report['failure'],'Episode_Termination/success':report['success'],
                              'Episode_Termination/phase_timeout':report['phase_timeout'],'Diagnostics/invalid_load_world_steps':report['invalid_load_world_steps']}.items():writer.add_scalar(tag,value,it)
            if count:writer.add_scalar('Train/mean_reward',report['mean_episode_reward'],it);writer.add_scalar('Train/mean_episode_length',report['mean_episode_length'],it)
            for i,value in enumerate(peak_loads):writer.add_scalar(f'Connector/peak_load_{i+1}_N',value,it)
            tmp=a.log_dir/'latest.pt.tmp';torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'mode':mode,'num_envs':n,'obs_dim':x.shape[-1],
                'iteration':it,'reference':str(a.reference),'reference_sha256':reference_sha256,'velocity_scale':a.velocity_scale,'torque_scale':a.torque_scale,'total_steps':total_steps,'total_time':total_time},tmp);tmp.replace(a.log_dir/'latest.pt')
            print(format_rsl_log(report),flush=True)
    finally:
        if writer is not None:writer.close()
        env.close();torch.cuda.set_stream(previous_stream)
if __name__=='__main__':main()
