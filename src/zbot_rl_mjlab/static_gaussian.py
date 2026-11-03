"""Gaussian mean policy gradients for loading and event-switched separation."""
from __future__ import annotations

import json
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg

MODE='static_gaussian_tilt_mean_v1'
SEPARATION_MODE='gaussian_load_then_separate_v1'


@dataclass
class GaussianMeanCfg:
    initial_checkpoint: str=''
    separation_initial_scale: float=0.
    exploration_std: float=.02
    initial_mean_std: float=.02
    minimum_exploration_std: float=.002
    exploration_retries: int=4
    mean_step_size: float=.02
    sample_steps: int=10
    validation_interval: int=10
    validation_steps: int=200
    backtrack_attempts: int=8


@dataclass
class StaticGaussianRunnerCfg(RslRlOnPolicyRunnerCfg):
    algorithm: GaussianMeanCfg=field(default_factory=GaussianMeanCfg)
    actor: None=None
    critic: None=None


class GaussianTorquePolicy(torch.nn.Module):
    """Six learned means and an explicit exploration radius, no observation network."""
    def __init__(self,mean,std,separation_phase=False,target_connector=5):
        super().__init__()
        self.register_buffer('mean',mean.clone())
        self.register_buffer('std',torch.full_like(mean,std))
        self.register_buffer('loading_mean',torch.zeros_like(mean))
        self.separation_phase=separation_phase
        self.target_connector=target_connector

    def forward(self,obs,stochastic_output=False):
        x=obs['actor']
        mean=self.mean.expand(x.shape[0],-1)
        if stochastic_output:mean=mean+torch.randn_like(mean)*self.std
        if self.separation_phase:
            active=x[:,24+self.target_connector:25+self.target_connector]>.5
            return torch.where(active,self.loading_mean,mean)
        return mean

    def reset(self,*args,**kwargs):
        pass


def score_gradient(noise,scores,std):
    """Gaussian score estimator; antithetic pairs cancel the common baseline."""
    if scores.numel()<2 or not torch.isfinite(scores).all():
        raise ValueError('Need at least two finite reward samples')
    centered=scores-scores.mean()
    return (noise*centered[:,None]/std**2).mean(0)


class StaticGaussianRunner:
    def __init__(self,env,train_cfg,log_dir=None,device='cpu',**kwargs):
        self.env=env;self.cfg=train_cfg;self.device=str(device)
        self.log_dir=Path(log_dir) if log_dir else None
        self.settings=GaussianMeanCfg(**train_cfg['algorithm'])
        c=self.settings
        if not math.isfinite(c.separation_initial_scale):
            raise ValueError('separation_initial_scale must be finite')
        for name in ('exploration_std','initial_mean_std','mean_step_size'):
            if not math.isfinite(getattr(c,name)) or getattr(c,name)<=0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('sample_steps','validation_interval','validation_steps','backtrack_attempts','exploration_retries'):
            if getattr(c,name)<1:
                raise ValueError(f'{name} must be positive')
        if not 0<c.minimum_exploration_std<=c.exploration_std:
            raise ValueError('minimum_exploration_std must be positive and no larger than exploration_std')
        core=env.unwrapped
        self.separation_phase=core.cfg.enable_weld_failure
        self.mode=SEPARATION_MODE if self.separation_phase else MODE
        if log_dir is not None and (core.num_envs<2 or core.num_envs%2):
            raise ValueError('Training needs an even number of Gaussian samples >=2')
        self.generator=torch.Generator(device=self.device).manual_seed(train_cfg.get('seed',42))
        mean=torch.randn(6,generator=self.generator,device=self.device)*c.initial_mean_std
        self.policy=GaussianTorquePolicy(mean.clamp(-1.,1.),c.exploration_std,self.separation_phase,
                                          getattr(core.cfg,'target_connector',5)).to(self.device)
        self.completed_updates=0;self.current_learning_iteration=0
        self.best_mean=None;self.best_report=None;self.center_report=None
        self.writer=None
        self.initialized=False
        self.revalidate_on_learn=False
        self.initialization_report={}

    @torch.inference_mode()
    def initialize_mean(self):
        if self.initialized:return
        path=self.settings.initial_checkpoint
        if path:
            saved=torch.load(path,map_location=self.device,weights_only=False)
            mode=saved.get('infos',{}).get('disconnect_control_mode')
            sd=saved['actor_state_dict']
            if mode==MODE:
                point=sd['mean'].to(self.device)
            elif mode=='torque_feedforward_p0_static_tilt_loading_obs31_v10':
                obs,_=self.env.reset();x=obs['actor'][:1]
                if sd['mlp.0.weight'].shape[1]!=31:raise ValueError('Expected 31-channel source policy')
                x=(x-sd['obs_normalizer._mean'])/(sd['obs_normalizer._std']+.01)
                for layer in (0,2,4):
                    x=torch.nn.functional.linear(x,sd[f'mlp.{layer}.weight'],sd[f'mlp.{layer}.bias'])
                    if layer!=4:x=torch.nn.functional.elu(x)
                point=x[0]
            else:
                raise ValueError('Initial checkpoint must be a compatible static loading policy')
            if point.shape!=(6,) or not torch.isfinite(point).all():
                raise ValueError('Loading initial point must have six finite actions')
            if self.separation_phase:
                self.policy.loading_mean.copy_(point.clamp(-1.,1.))
                self.policy.mean.copy_((point*self.settings.separation_initial_scale).clamp(-1.,1.))
            else:self.policy.mean.copy_(point.clamp(-1.,1.))
            self.initialization_report={'checkpoint':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        elif self.separation_phase:
            raise ValueError('Separation training needs a verified loading initial_checkpoint')
        self.initialized=True

    def add_git_repo_to_log(self,*args,**kwargs):
        pass

    def get_inference_policy(self,device=None):
        return self.policy.to(device or self.device).eval()

    @torch.inference_mode()
    def evaluate(self,actions,steps):
        """Each trial starts from the same physical reset; stop scoring after done.

        The actual environment supplies every reward and safety termination.
        A candidate is held constant, never teleported into a loaded state.
        """
        from .tasks.reconfiguration.disconnect.state import state,target_tilt_load
        core=self.env.unwrapped
        if not core.cfg.auto_reset:raise ValueError('Trial evaluation requires auto_reset=true')
        obs,_=self.env.reset()
        actions=actions.expand(core.num_envs,-1)
        alive=torch.ones(core.num_envs,dtype=torch.bool,device=self.device)
        failed=torch.zeros_like(alive)
        scores=torch.zeros(core.num_envs,device=self.device)
        tilt_sum=torch.zeros_like(scores);margin_min=torch.ones_like(scores)
        broken_seen=torch.zeros_like(alive);success_seen=torch.zeros_like(alive)
        wait_seen=torch.zeros_like(alive);max_gap=torch.zeros_like(scores)
        current_broken=torch.zeros_like(alive);current_success=torch.zeros_like(alive)
        current_waiting=torch.zeros_like(alive);current_gap=torch.zeros_like(scores)
        peak_tilt_seen=torch.zeros_like(scores);break_load_seen=torch.zeros_like(scores)
        current_peak_tilt=torch.zeros_like(scores);current_break_load=torch.zeros_like(scores)
        executed_steps=0
        wrong_seen=torch.zeros_like(alive);safety_seen=torch.zeros_like(alive);torque_seen=torch.zeros_like(alive)
        current_wrong=torch.zeros_like(alive);current_safety=torch.zeros_like(alive);current_torque=torch.zeros_like(alive)
        manager=core.reward_manager
        components=torch.zeros_like(manager._step_reward)
        component_sum=torch.zeros_like(components)
        current_tilt=torch.zeros_like(scores);current_margin=torch.ones_like(scores)
        original_compute=manager.compute
        def capture_rewards(*args,**kwargs):
            result=original_compute(*args,**kwargs)
            # Capture before automatic reset clears terminal reward terms.
            components.copy_(manager._step_reward)
            current_tilt.copy_(target_tilt_load(core))
            st=state(core)
            current_margin.copy_(st.other_margin())
            current_broken.copy_(~st.connected()[:,core.cfg.target_connector])
            from .tasks.reconfiguration.disconnect.state import success
            current_success.copy_(success(core))
            current_waiting.copy_(st.waiting)
            current_wrong.copy_(st.wrong_break)
            current_safety.copy_(st.safety_margin_violation)
            current_torque.copy_(core.action_manager.get_term('joint_torque').torque_failure)
            current_gap.copy_(st.geometry()['distance'])
            current_peak_tilt.copy_(st.peak_target_tilt_load)
            current_break_load.copy_(st.break_utilization[:,core.cfg.target_connector]*core.cfg.break_force)
            return result
        manager.compute=capture_rewards
        try:
            for _ in range(steps):
                executed_steps+=1
                selected=actions
                if self.separation_phase:
                    active=obs['actor'][:,24+core.cfg.target_connector:25+core.cfg.target_connector]>.5
                    selected=torch.where(active,self.policy.loading_mean,actions)
                obs,reward,done,_=self.env.step(torch.where(alive[:,None],selected,0.))
                scores+=torch.where(alive,reward,0.)
                component_sum+=torch.where(alive[:,None],components,0.)
                failed |= alive & core.reset_terminated
                tilt_sum+=torch.where(alive,current_tilt,0.)
                margin_min=torch.where(alive,torch.minimum(margin_min,current_margin),margin_min)
                peak_tilt_seen=torch.maximum(peak_tilt_seen,torch.where(alive,current_peak_tilt,0.))
                break_load_seen=torch.maximum(break_load_seen,torch.where(alive,current_break_load,0.))
                wrong_seen |= alive & current_wrong
                safety_seen |= alive & current_safety
                torque_seen |= alive & current_torque
                broken_seen |= alive & current_broken
                success_seen |= alive & current_success
                wait_seen |= alive & current_waiting
                max_gap=torch.maximum(max_gap,torch.where(alive,current_gap,0.))
                alive &= ~done.bool()
                if self.separation_phase and not alive.any():break
        finally:
            manager.compute=original_compute
        scores/=steps
        report={'score':float(scores.mean()),'failure_rate':float(failed.float().mean()),
                'target_tilt_load_N':float((tilt_sum/steps).mean()),
                'other_min_margin':float(margin_min.min()),'steps':steps,'executed_steps':executed_steps,
                'reward_terms':{name:float(value) for name,value in zip(manager.active_terms,component_sum.mean(0)/steps)}}
        if self.separation_phase:
            report.update(success_rate=float(success_seen.float().mean()),
                target_broken_rate=float(broken_seen.float().mean()),
                wait_started_rate=float(wait_seen.float().mean()),max_target_gap_m=float(max_gap.mean()),
                target_break_load_N=float(break_load_seen.mean()),target_peak_tilt_load_N=float(peak_tilt_seen.mean()),
                wrong_break_rate=float(wrong_seen.float().mean()),safety_violation_rate=float(safety_seen.float().mean()),
                torque_violation_rate=float(torque_seen.float().mean()))
        return scores.clone(),report

    def sample_gradient(self):
        n=self.env.unwrapped.num_envs
        if n<2 or n%2:
            raise ValueError('Antithetic Gaussian sampling needs an even number of worlds >=2')
        for _ in range(self.settings.exploration_retries):
            z=torch.randn((n//2,6),generator=self.generator,device=self.device)
            sample_std=self.policy.std.clone()
            noise=torch.cat((z,-z),0)*sample_std
            scores,report=self.evaluate(self.policy.mean+noise,self.settings.sample_steps)
            if report['failure_rate']==0 or float(self.policy.std.min())<=self.settings.minimum_exploration_std:
                break
            self.policy.std.mul_(.5).clamp_(min=self.settings.minimum_exploration_std)
        return score_gradient(noise,scores,sample_std),report

    def validate(self):
        _,report=self.evaluate(self.policy.mean,self.settings.validation_steps)
        if report['failure_rate']==0:
            if self.best_report is None or (report.get('success_rate',0.),report['score'])>(self.best_report.get('success_rate',0.),self.best_report['score']):
                self.best_mean=self.policy.mean.clone();self.best_report=report
                if self.log_dir:self.save(self.log_dir/'best.pt')
        elif self.best_mean is not None:
            self.policy.mean.copy_(self.best_mean)
            self.center_report=None
        return report

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        if not self.env.unwrapped.cfg.auto_reset:
            raise ValueError('Training requires auto_reset=true')
        # Every gradient batch resets explicitly; randomized episode counters
        # and temporal return estimation have no role in this static objective.
        if self.log_dir:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer=SummaryWriter(str(self.log_dir))
        try:
            self.initialize_mean()
            if self.revalidate_on_learn:
                self.best_mean=None;self.best_report=None
                self.revalidate_on_learn=False
            if self.best_report is None:self.validate()
            print('STATIC_GAUSSIAN_INITIAL '+json.dumps({'mean_action':self.policy.mean.tolist(),'loading_action':self.policy.loading_mean.tolist(),
                'initialization':self.initialization_report,'validation':self.best_report}),flush=True)
            for _ in range(num_learning_iterations):
                start=time.perf_counter()
                gradient,sampled=self.sample_gradient()
                sample_time=time.perf_counter()-start
                candidate_start=time.perf_counter()
                if self.center_report is None:
                    _,self.center_report=self.evaluate(self.policy.mean,self.settings.sample_steps)
                old=self.policy.mean.clone();accepted=False;used_step=0.
                norm=float(gradient.norm())
                if math.isfinite(norm) and norm>1e-12:
                    direction=gradient/norm
                    for attempt in range(self.settings.backtrack_attempts):
                        step=self.settings.mean_step_size*(.5**attempt)
                        proposal=(old+step*direction).clamp(-1.,1.)
                        _,candidate=self.evaluate(proposal,self.settings.sample_steps)
                        better=((candidate.get('success_rate',0.),candidate['score'])
                                >(self.center_report.get('success_rate',0.),self.center_report['score']+1e-8))
                        if candidate['failure_rate']==0 and better:
                            self.policy.mean.copy_(proposal)
                            self.center_report=candidate;accepted=True;used_step=step
                            break
                if not accepted:
                    # Smaller smoothing radius improves the local score estimate
                    # near a safety boundary; the mean step size stays independent.
                    self.policy.std.mul_(.5).clamp_(min=self.settings.minimum_exploration_std)
                candidate_time=time.perf_counter()-candidate_start
                self.completed_updates+=1;self.current_learning_iteration=self.completed_updates-1
                validation=None;validation_time=0.
                if self.completed_updates%self.settings.validation_interval==0:
                    validation_start=time.perf_counter()
                    validation=self.validate()
                    validation_time=time.perf_counter()-validation_start
                report={'update':self.completed_updates,'accepted':accepted,'action_step_norm':used_step,
                        'gradient_norm':norm,'mean_action':self.policy.mean.tolist(),
                        'sample_failure_rate':sampled['failure_rate'],'exploration_std':float(self.policy.std.mean()),**(self.center_report or {}),
                        'iteration_time_s':time.perf_counter()-start,'sample_time_s':sample_time,
                        'candidate_time_s':candidate_time,'validation_time_s':validation_time}
                if validation is not None:report['validation']=validation
                print('STATIC_GAUSSIAN '+json.dumps(report),flush=True)
                if self.writer:
                    for key in ('score','target_tilt_load_N','other_min_margin','gradient_norm','action_step_norm',
                                'sample_failure_rate','iteration_time_s','sample_time_s','candidate_time_s','validation_time_s','success_rate','target_broken_rate','wait_started_rate','max_target_gap_m','target_break_load_N','target_peak_tilt_load_N','wrong_break_rate','safety_violation_rate','torque_violation_rate'):
                        if key in report:self.writer.add_scalar('StaticGaussian/'+key,report[key],self.completed_updates)
                    for key,value in report.get('reward_terms',{}).items():
                        self.writer.add_scalar('StaticReward/'+key,value,self.completed_updates)
                    if validation:
                        for key,value in validation.items():
                            if isinstance(value,(int,float)):self.writer.add_scalar('Validation/'+key,value,self.completed_updates)
                if self.log_dir and self.completed_updates%self.cfg['save_interval']==0:
                    self.save(self.log_dir/f'model_{self.current_learning_iteration}.pt')
            self.validate()
            if self.log_dir:self.save(self.log_dir/f'model_{self.current_learning_iteration}.pt')
        finally:
            if self.writer:self.writer.close();self.writer=None

    def save(self,path,infos=None):
        torch.save({'actor_state_dict':self.policy.state_dict(),'iter':self.current_learning_iteration,
            'completed_updates':self.completed_updates,'generator_state':self.generator.get_state(),
            'best_mean':self.best_mean,'best_report':self.best_report,'center_report':self.center_report,
            'settings':vars(self.settings),'initialization_report':self.initialization_report,'infos':{**(infos or {}),'disconnect_control_mode':self.mode,'target_connector':getattr(self.env.unwrapped.cfg,'target_connector',5)}},str(path))

    def load(self,path,load_cfg=None,strict=True,map_location=None):
        saved=torch.load(path,map_location=map_location or self.device,weights_only=False)
        if saved.get('infos',{}).get('disconnect_control_mode')!=self.mode:
            raise ValueError('旧PPO/静态阶段或其他控制模式模型不能直接作为当前阶段高斯策略加载；旧文件保留。')
        if saved.get('infos',{}).get('target_connector',5)!=getattr(self.env.unwrapped.cfg,'target_connector',5):
            raise ValueError('Checkpoint target connector differs from the current task')
        sd=saved['actor_state_dict']
        if self.mode==MODE and 'loading_mean' not in sd:sd={**sd,'loading_mean':torch.zeros_like(sd['mean'])}
        self.policy.load_state_dict(sd,strict=strict)
        self.initialized=True
        self.initialization_report=saved.get('initialization_report',{})
        if load_cfg is None or load_cfg.get('iteration',False):
            self.completed_updates=saved['completed_updates'];self.current_learning_iteration=saved['iter']
            self.revalidate_on_learn=True
            self.generator.set_state(saved['generator_state'].cpu())
            self.best_mean=None if saved['best_mean'] is None else saved['best_mean'].to(self.device)
            self.best_report=saved['best_report'];self.center_report=saved['center_report']
            # New explicit exploration settings take effect on resume.
            if saved['settings']['exploration_std']!=self.settings.exploration_std:
                self.policy.std.fill_(self.settings.exploration_std)
            self.policy.std.clamp_(min=self.settings.minimum_exploration_std)
            self.center_report=None
        return saved.get('infos',{})
