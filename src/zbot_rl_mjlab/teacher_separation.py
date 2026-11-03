"""Frozen loading teacher and RSL-RL PPO student, with phase-masked policy loss."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from mjlab.rl import MjlabOnPolicyRunner,RslRlOnPolicyRunnerCfg

MODE='frozen_loading_teacher_masked_separation_ppo_v1'


@dataclass
class TeacherSeparationRunnerCfg(RslRlOnPolicyRunnerCfg):
    teacher_checkpoint: str='outputs/reconfiguration/disconnect_torque/static_gaussian/penalty02/verification_model.pt'


class TeacherStudentPolicy(torch.nn.Module):
    def __init__(self,actor,teacher,target):
        super().__init__();self.actor=actor;self.target=target
        self.register_buffer('teacher',teacher.clone())

    def forward(self,obs,stochastic_output=False):
        student=self.actor(obs,stochastic_output=stochastic_output)
        loading=obs['actor'][:,24+self.target:25+self.target]>.5
        action=torch.where(loading,self.teacher,student)
        # The native Action also enforces this latch at every physics substep.
        if 'critic' in obs and obs['critic'].shape[-1]==33:
            action=torch.where(obs['critic'][:,31:32]>.5,0.,action)
        return action

    def reset(self,*args,**kwargs):return self.actor.reset(*args,**kwargs)


class TeacherMaskedPPO(PPO):
    def __init__(self,*args,min_learning_rate=1e-5,max_learning_rate=1e-2,**kwargs):
        super().__init__(*args,**kwargs)
        self.min_learning_rate=min_learning_rate;self.max_learning_rate=max_learning_rate
        if self.is_multi_gpu or self.rnd or self.symmetry or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('Phase-masked separation PPO currently supports single-GPU feedforward policies without extensions')
        self.policy_masks=torch.zeros(self.storage.num_transitions_per_env,self.storage.num_envs,
                                      device=self.device,dtype=torch.bool)
        self.teacher_action=torch.zeros(6,device=self.device)
        self.core=None

    @staticmethod
    def construct_algorithm(obs,env,cfg,device):
        algorithm=PPO.construct_algorithm(obs,env,cfg,device)
        algorithm.core=env.unwrapped
        return algorithm

    def act(self,obs):
        from .tasks.reconfiguration.disconnect.state import state
        student=super().act(obs)
        s=state(self.core)
        loading=s.connected()[:,self.core.cfg.target_connector]
        self.pending_mask=(~loading & ~s.waiting).clone()
        selected=torch.where(loading[:,None],self.teacher_action,student)
        return torch.where(s.waiting[:,None],0.,selected)

    def process_env_step(self,obs,rewards,dones,extras):
        # The mask belongs to the PRE-action state, never the reset/next state.
        self.policy_masks[self.storage.step].copy_(self.pending_mask)
        super().process_env_step(obs,rewards,dones,extras)

    def compute_returns(self,obs):
        # Preserve all transitions so wait-period rewards and safety failures
        # propagate through native GAE to the preceding student actions.
        super().compute_returns(obs)
        raw=self.storage.returns-self.storage.values
        mask=self.policy_masks
        advantages=torch.zeros_like(raw)
        if mask.any():
            active=raw[mask]
            if not self.normalize_advantage_per_mini_batch:
                active=(active-active.mean())/(active.std(unbiased=False)+1e-8)
            advantages[mask]=active
        self.storage.advantages=advantages

    def update(self):
        """Native PPO objective/Adam/GAE, selecting actor samples by phase.

        Critic learns ALL states; surrogate, entropy, KL and actor observation
        normalization use ONLY states in which the student controlled torque.
        No shared-environment dependency is patched.
        """
        st=self.storage
        obs=st.observations.flatten(0,1)
        actions=st.actions.flatten(0,1)
        old_logp=st.actions_log_prob.flatten()
        old_params=tuple(p.flatten(0,1) for p in st.distribution_params)
        advantages=st.advantages.flatten()
        returns=st.returns.flatten();old_values=st.values.flatten()
        mask=self.policy_masks.flatten()
        count=len(mask);batch_size=max(1,count//self.num_mini_batches)
        sums={'value':0.,'surrogate':0.,'entropy':0.,'kl':0.};batches=0;actor_batches=0
        for _ in range(self.num_learning_epochs):
            permutation=torch.randperm(count,device=self.device)
            for start in range(0,count,batch_size):
                idx=permutation[start:start+batch_size]
                value=self.critic(obs[idx]).flatten()
                if self.use_clipped_value_loss:
                    clipped=old_values[idx]+(value-old_values[idx]).clamp(-self.clip_param,self.clip_param)
                    value_loss=torch.maximum((value-returns[idx]).square(),(clipped-returns[idx]).square()).mean()
                else:value_loss=(value-returns[idx]).square().mean()
                selected=idx[mask[idx]]
                surrogate=value.new_zeros(());entropy=value.new_zeros(());kl_mean=value.new_zeros(())
                if selected.numel():
                    self.actor(obs[selected],stochastic_output=True)
                    logp=self.actor.get_output_log_prob(actions[selected])
                    entropy=self.actor.output_entropy.mean()
                    adv=advantages[selected]
                    if self.normalize_advantage_per_mini_batch:
                        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                    ratio=(logp-old_logp[selected]).exp()
                    surrogate=torch.maximum(-adv*ratio,-adv*ratio.clamp(1.-self.clip_param,1.+self.clip_param)).mean()
                    with torch.no_grad():
                        kl_mean=self.actor.get_kl_divergence(tuple(p[selected] for p in old_params),self.actor.output_distribution_params).mean()
                        if self.desired_kl is not None and self.schedule=='adaptive':
                            if kl_mean>self.desired_kl*2.:self.learning_rate=max(self.min_learning_rate,self.learning_rate/1.5)
                            elif 0.<kl_mean<self.desired_kl/2.:self.learning_rate=min(self.max_learning_rate,self.learning_rate*1.5)
                            for group in self.optimizer.param_groups:group['lr']=self.learning_rate
                    actor_batches+=1
                loss=surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
                self.optimizer.step()
                for key,item in [('value',value_loss),('surrogate',surrogate),('entropy',entropy),('kl',kl_mean)]:sums[key]+=float(item.detach())
                batches+=1
        if mask.any():self.actor.update_normalization(obs[mask])
        self.critic.update_normalization(obs)
        result={k:v/max(batches,1) for k,v in sums.items()}
        result.update(student_fraction=float(mask.float().mean()),actor_batches=float(actor_batches))
        st.clear();self.policy_masks.zero_()
        return result

    def save(self):
        saved=super().save();saved['loading_teacher_action']=self.teacher_action.clone()
        return saved

    def load(self,saved,load_cfg,strict):
        restored=super().load(saved,load_cfg,strict)
        self.teacher_action.copy_(saved['loading_teacher_action'].to(self.device))
        return restored


class TeacherSeparationRunner(MjlabOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        if not env.unwrapped.cfg.enable_weld_failure:
            raise ValueError('Teacher/student separation requires enable_weld_failure=true')
        super().__init__(env,train_cfg,log_dir,device)
        self.teacher_initialized=False;self.teacher_report={}
        # Start the student safely, without inheriting an incompatible action head.
        with torch.no_grad():
            final=[layer for layer in self.alg.actor.mlp.modules() if isinstance(layer,torch.nn.Linear)][-1]
            final.weight.zero_();final.bias.zero_()

    def initialize_teacher(self):
        if self.teacher_initialized:return
        path=Path(self.cfg['teacher_checkpoint'])
        saved=torch.load(path,map_location=self.device,weights_only=False)
        if saved.get('infos',{}).get('disconnect_control_mode')!='static_gaussian_tilt_mean_v1':
            raise ValueError('Loading teacher must be the verified static Gaussian policy')
        teacher=saved['actor_state_dict']['mean'].to(self.device)
        if teacher.shape!=(6,) or not torch.isfinite(teacher).all() or (teacher.abs()>1).any():
            raise ValueError('Loading teacher must contain six finite normalized actions in [-1,1]')
        self.alg.teacher_action.copy_(teacher)
        self.teacher_report={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        self.teacher_initialized=True

    def learn(self,*args,**kwargs):
        self.initialize_teacher()
        return super().learn(*args,**kwargs)

    def get_inference_policy(self,device=None):
        self.initialize_teacher()
        actor=self.alg.get_policy().to(device or self.device).eval()
        return TeacherStudentPolicy(actor,self.alg.teacher_action,self.env.unwrapped.cfg.target_connector).to(device or self.device).eval()

    def save(self,path,infos=None):
        self.initialize_teacher()
        super().save(path,{**(infos or {}),'disconnect_control_mode':MODE,
                         'loading_teacher':self.teacher_report,'target_connector':self.env.unwrapped.cfg.target_connector})

    def load(self,path,*args,**kwargs):
        saved=torch.load(path,map_location=self.device,weights_only=False)
        infos=saved.get('infos',{})
        if infos.get('disconnect_control_mode')!=MODE or infos.get('target_connector',5)!=self.env.unwrapped.cfg.target_connector:
            raise ValueError('旧阶段/其他目标模型不能作为教师加载+PPO分离模型续训或回放，旧文件保留。')
        result=super().load(path,*args,**kwargs)
        self.teacher_report=infos.get('loading_teacher',{});self.teacher_initialized=True
        return result
