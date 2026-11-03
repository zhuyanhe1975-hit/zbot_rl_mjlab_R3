"""Train a PPO residual student over the full CPU reconstruction FSM."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import FullCpuStudent

class ActorCritic(nn.Module):
    def __init__(self, n):
        super().__init__()
        def net(out):return nn.Sequential(nn.Linear(n,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,out))
        self.actor=net(7);self.critic=net(1);self.log_std=nn.Parameter(torch.full((7,),-2.))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
    def dist(self,x):return Normal(self.actor(x),self.log_std.clamp(-5.,0.).exp())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--rollout-steps',type=int)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--residual-scale',type=float,default=.01)
    p.add_argument('--velocity-scale',type=float,default=.5);p.add_argument('--torque-scale',type=float,default=2.)
    p.add_argument('--env.scene.num-envs', '--num-envs', dest='num_envs', type=int, default=1)
    p.add_argument('--reference',type=Path);p.add_argument('--log-dir',type=Path,default=Path('logs/full_cpu_residual_student'));p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if args.num_envs<1 or args.iterations<1 or (args.rollout_steps is not None and args.rollout_steps<2) or not 0<args.residual_scale<=.1:p.error('invalid training parameters')
    if args.reference is None:
        args.reference=Path('assets/reconfiguration/refined_forward_velocity_reference.npz' if args.num_envs>1 else 'outputs/reconfiguration/full_cpu_student/teacher_reference.npz')
    if args.num_envs>1:
        from train_full_batched_student import main as gpu_main
        log_dir=args.log_dir if str(args.log_dir)!='logs/full_cpu_residual_student' else Path('logs/full_gpu_reference_student')
        gpu_args=['--num-envs',str(args.num_envs),'--iterations',str(args.iterations),'--rollout-steps',str(args.rollout_steps or 128),'--seed',str(args.seed),'--velocity-scale',str(args.velocity_scale),'--torque-scale',str(args.torque_scale),'--reference',str(args.reference),'--log-dir',str(log_dir)]
        if args.resume:gpu_args.extend(['--resume',str(args.resume)])
        return gpu_main(gpu_args)
    if args.rollout_steps is None:args.rollout_steps=512
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);args.log_dir.mkdir(parents=True,exist_ok=True)
    env=FullCpuStudent(args.residual_scale,args.reference);obs=env.observation();model=ActorCritic(len(obs));opt=torch.optim.Adam(model.parameters(),lr=3e-4);start_iter=0
    if args.resume:
        ckpt=torch.load(args.resume,map_location='cpu',weights_only=False)
        if ckpt.get('reference')!=str(args.reference) or ckpt['obs_dim']!=len(obs) or ckpt['residual_scale']!=args.residual_scale:raise ValueError('checkpoint incompatible with observation or action scale')
        model.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer']);start_iter=ckpt['iteration']+1
    ep_return=0.;episode=0
    try:
        for iteration in range(start_iter,start_iter+args.iterations):
            start=time.time();rows=[];completed=[]
            for _ in range(args.rollout_steps):
                x=torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    dist=model.dist(x);raw=dist.sample();logp=dist.log_prob(raw).sum(-1);value=model.critic(x).squeeze(-1)
                nxt,reward,done,info=env.step(torch.tanh(raw[0]).numpy());ep_return+=reward
                rows.append((obs,raw[0].numpy(),float(logp),float(value),reward,done))
                obs=nxt
                if done:
                    completed.append({'return':ep_return,'success':info['phase']=='DONE',**info});episode+=1;ep_return=0.;env.close();env=FullCpuStudent(args.residual_scale,args.reference);obs=env.observation()
            with torch.no_grad(): bootstrap=float(model.critic(torch.from_numpy(obs).unsqueeze(0))[0,0])
            adv=np.zeros(len(rows),np.float32);gae=0.
            for i in reversed(range(len(rows))):
                _,_,_,v,r,done=rows[i];nv=bootstrap if i==len(rows)-1 else rows[i+1][3];mask=1.-float(done)
                gae=r+.995*nv*mask-v+.995*.95*mask*gae;adv[i]=gae
            ret=adv+np.array([r[3] for r in rows],np.float32);adv=(adv-adv.mean())/(adv.std()+1e-8)
            X=torch.tensor(np.stack([r[0] for r in rows]));A=torch.tensor(np.stack([r[1] for r in rows]));L=torch.tensor([r[2] for r in rows]);G=torch.tensor(adv);R=torch.tensor(ret)
            for _ in range(3):
                for ids in torch.randperm(len(rows)).split(128):
                    dist=model.dist(X[ids]);ratio=(dist.log_prob(A[ids]).sum(-1)-L[ids]).exp();policy=-torch.minimum(ratio*G[ids],ratio.clamp(.8,1.2)*G[ids]).mean()
                    loss=policy+.5*(model.critic(X[ids]).squeeze(-1)-R[ids]).square().mean()-.001*dist.entropy().sum(-1).mean()
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            report={'iteration':iteration,'steps':len(rows),'episodes_total':episode,'completed':completed,'phase':env.fsm.phase,'sim_time':env.fsm.data.time,'peak_other_N':env.fsm.peak_other,'wall_s':time.time()-start,'loss':float(loss.detach())}
            with (args.log_dir/'metrics.jsonl').open('a') as out:out.write(json.dumps(report)+'\n')
            print('PPO',json.dumps(report),flush=True)
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'iteration':iteration,'obs_dim':len(obs),'residual_scale':args.residual_scale,'reference':str(args.reference)},args.log_dir/'latest.pt')
    finally:env.close()
if __name__=='__main__':main()
