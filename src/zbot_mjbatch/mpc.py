"""Receding-horizon predictive sampling with actual mjbatch rollouts."""
import numpy as np
from .env import HeadTailBatch

class PredictiveController:
    """Select a safe proposal from parallel, short physics rollouts.

    PPO outputs the central proposal; alternatives change the residual and
    clock. Every candidate runs the same 1 ms safety checks as the real world.
    """
    def __init__(self,num_envs,threads,candidates=4,horizon=2):
        if candidates<2 or horizon<1:raise ValueError('MPC requires >=2 candidates and positive horizon')
        self.num_envs=num_envs;self.candidates=candidates;self.horizon=horizon
        self.prediction=HeadTailBatch(num_envs*candidates,threads)

    def select(self,env,proposal):
        proposal=np.clip(np.asarray(proposal,dtype=float),-1.,1.)
        if proposal.shape!=(self.num_envs,7):raise ValueError('MPC proposal shape mismatch')
        if not np.any(env.phase==1):
            return proposal,{'safe_fraction':1.,'nominal_fraction':1.,'predicted_failure_fraction':0.}
        k=self.candidates;sim=self.prediction;n=self.num_envs
        sim.state[:]=np.repeat(env.state,k,axis=0)
        sim.eq[:]=np.repeat(env.eq,k,axis=0)
        sim.warnings[:]=np.repeat(env.warnings,k,axis=0)
        sim.warning_baseline[:]=np.repeat(env.warning_baseline,k,axis=0)
        for name in ('qtarget','velocity','action','elapsed','capture_time','settle_time','phase','overload','loads','last_peak_N','last_peak_torque'):
            getattr(sim,name)[:]=np.repeat(getattr(env,name),k,axis=0)
        sim.batch.forward()
        choices=np.repeat(proposal[:,None,:],k,axis=1)
        choices[:,1,:]=0.
        direction=np.sign(env.goal[None,:]-env.qpos[:,env.qids])
        for i in range(2,k):
            if i%3==2:choices[:,i,:6]=np.clip(proposal[:,:6]+.4*direction,-1.,1.)
            elif i%3==0:choices[:,i,:6]=np.clip(proposal[:,:6]-.4*direction,-1.,1.)
            else:choices[:,i,6]=1.
        choices=choices.reshape(n*k,7)
        predicted_failure=np.zeros(n*k,bool);predicted_success=np.zeros(n*k,bool)
        peaks=np.zeros(n*k)
        for _ in range(self.horizon):
            _,_,done,info=sim.step(choices)
            active=~predicted_failure&~predicted_success
            predicted_failure|=active&done&info['failed']
            predicted_success|=active&done&info['success']
            peaks=np.maximum(peaks,info['peak_load_N'])
        delta,rot,_=sim.geometry();gap=np.linalg.norm(delta,axis=1);angle=np.linalg.norm(rot,axis=1)
        q=sim.qpos[:,sim.qids]
        # Joint endpoint supplies global direction when initial site distance is
        # locally flat; near the endpoint the site pose takes over.
        score=np.linalg.norm(q-sim.goal,axis=1)+.3*gap/.5+.08*angle
        score+=.01*peaks/80.+.0001*np.sum((choices-np.repeat(proposal,k,axis=0))**2,axis=1)
        score=np.where(predicted_failure,np.inf,np.where(predicted_success,-10.,score)).reshape(n,k)
        choice=np.argmin(score,axis=1);safe=np.isfinite(score).any(axis=1)
        result=choices.reshape(n,k,7)[np.arange(n),choice].copy()
        result[~safe]=0.
        # Only motion worlds are planned. In the hold/settle phases the action
        # is physically ignored, so keep the policy proposal unchanged.
        result[env.phase!=1]=proposal[env.phase!=1]
        return result,{'safe_fraction':float(np.mean(safe)),
                       'nominal_fraction':float(np.mean((choice==0)&safe)),
                       'predicted_failure_fraction':float(np.mean(predicted_failure))}

    def close(self):self.prediction.close()
