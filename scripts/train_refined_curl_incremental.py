"""Small, gated first-stage PPO experiment over the verified CPU FSM.

The policy controls only a bounded velocity residual while curling.  After
curling, the original FSM captures the head-tail connector without policy
intervention.  Every update is accepted only after a deterministic physical
validation; this is a single-CPU-world pilot, not 4096-world GPU training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from zbot_rl_mjlab.tasks.reconfiguration.refined_curl_rl import RefinedCurlEnv

MODE='refined_cpu_curl_residual_ppo_v1'


class Net(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,6))
        self.value=nn.Sequential(nn.Linear(n,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        self.log_std=nn.Parameter(torch.full((6,),float(np.log(.1))))
    def distribution(self,x):return Normal(self.actor(x),self.log_std.clamp(-5.,-1.).exp())


def evaluate(net,speed,offset):
    env=RefinedCurlEnv(speed,offset)
    try:
        obs=env.reset()
        for _ in range(500):
            with torch.no_grad():action=torch.tanh(net.actor(torch.from_numpy(obs))).numpy()
            obs,reward,done,info=env.step(action)
            if done:
                info['stage_success']=info['success']
                if info['success']:
                    f=env.f
                    while f.phase not in ('DONE','FAILED') and f.data.time<120:
                        f.step()
                    info['success']=f.phase=='DONE'
                    info['phase']=f.phase
                    info['failure']=f.failure or ('' if info['success'] else 'full evaluation timeout')
                    info['elapsed_s']=float(f.data.time)
                    info['peak_other_N']=max(info['peak_other_N'],max((v['load_N'] for v in f.phase_loads.values()),default=0.))
                return info
        return {'success':False,'failure':'evaluation timeout','peak_other_N':env.peak_other}
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--rollout-steps',type=int,default=128)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--residual-speed',type=float,default=.02)
    p.add_argument('--residual-offset',type=float,default=.02)
    p.add_argument('--log-dir',type=Path,default=Path('logs/refined_curl_incremental_cpu'))
    args=p.parse_args()
    if args.iterations<1 or args.rollout_steps<2:p.error('positive iterations and rollout steps >=2 required')
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    env=RefinedCurlEnv(args.residual_speed,args.residual_offset);args.log_dir.mkdir(parents=True,exist_ok=True)
    try:
        obs=env.reset();net=Net(len(obs));optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
        baseline=evaluate(net,args.residual_speed,args.residual_offset)
        if not baseline['success'] or baseline['peak_other_N']>80.:
            raise RuntimeError(f'zero-residual baseline did not pass: {baseline}')
        print('ZERO_BASELINE',json.dumps(baseline),flush=True)
        for iteration in range(args.iterations):
            records=[];episodes=[]
            for _ in range(args.rollout_steps):
                x=torch.from_numpy(obs.copy())
                with torch.no_grad():
                    dist=net.distribution(x);raw=dist.sample();logp=dist.log_prob(raw).sum();value=net.value(x).squeeze()
                next_obs,reward,done,info=env.step(torch.tanh(raw).numpy())
                records.append((x,raw,logp,value,reward,done,info['policy_active']))
                obs=next_obs
                if done:episodes.append(info);obs=env.reset()
            with torch.no_grad():bootstrap=net.value(torch.from_numpy(obs)).item()
            advantage=[];gae=0.
            for i in reversed(range(len(records))):
                x,raw,logp,value,reward,done,mask=records[i]
                next_value=bootstrap if i==len(records)-1 else records[i+1][3].item()
                gae=reward+.995*next_value*(not done)-value.item()+.995*.95*(not done)*gae
                advantage.append(gae)
            advantage=torch.tensor(advantage[::-1],dtype=torch.float32);values=torch.stack([r[3] for r in records]);returns=advantage+values
            advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
            X=torch.stack([r[0] for r in records]);A=torch.stack([r[1] for r in records]);old=torch.stack([r[2] for r in records]);mask=torch.tensor([r[6] for r in records],dtype=torch.bool)
            previous_model=copy.deepcopy(net.state_dict());previous_optimizer=copy.deepcopy(optimizer.state_dict())
            for _ in range(2):
                dist=net.distribution(X);ratio=(dist.log_prob(A).sum(-1)-old).exp()
                surrogate=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage)
                actor=surrogate[mask].mean() if mask.any() else surrogate.sum()*0.
                loss=actor+.5*(net.value(X).squeeze(-1)-returns).square().mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            check=evaluate(net,args.residual_speed,args.residual_offset)
            accepted=bool(check['success'] and check['peak_other_N']<=80.)
            if not accepted:net.load_state_dict(previous_model);optimizer.load_state_dict(previous_optimizer)
            report={'iteration':iteration,'samples':len(records),'policy_samples':int(mask.sum()),'episodes':len(episodes),
                    'sampled_success':sum(e['success'] for e in episodes),'sampled_failure':sum(not e['success'] for e in episodes),
                    'accepted':accepted,'deterministic_validation':check,'residual_speed_rad_s':args.residual_speed,
                    'residual_offset_rad':args.residual_offset,'mode':MODE}
            with (args.log_dir/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
            source=Path('scripts/view_headtail_mjbatch_refined.py')
            torch.save({'mode':MODE,'model':net.state_dict(),'optimizer':optimizer.state_dict(),'obs_dim':len(obs),
                        'residual_speed':args.residual_speed,'residual_offset':args.residual_offset,'iteration':iteration,
                        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},args.log_dir/'latest.pt')
            print('ITERATION',json.dumps(report),flush=True)
    finally:env.close()

if __name__=='__main__':main()
