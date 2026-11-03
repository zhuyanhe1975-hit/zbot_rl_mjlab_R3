"""Pure joint-plan teacher imitation; deployed students execute without a teacher."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from rsl_rl.utils import resolve_class, resolve_obs_groups
from tensordict import TensorDict

from .tasks.reconfiguration.planning import DEFAULT_PLAN, load_plan


@dataclass
class PlanImitationCfg:
    class_name: str = 'zbot_rl_mjlab.planning_guidance:PlanImitation'
    plan_file: str = str(DEFAULT_PLAN)
    plan_branch: int = 0
    plan_duration_s: float = 4.
    demo_hold_s: float = 2.
    demo_envs: int = 16
    demo_action_noise: float = .03
    batch_size: int = 512
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.
    validation_interval: int = 250
    validation_envs: int = 4
    validation_duration_s: float = 9.
    validation_required_passes: int = 2


@dataclass
class PlanGuidedRunnerCfg(RslRlOnPolicyRunnerCfg):
    algorithm: PlanImitationCfg = field(default_factory=PlanImitationCfg)


class JointPlanTeacher:
    """State-feedback path follower using the observed integrated joint target.

    One branch per student avoids opposing labels at the identical straight
    start. No planner phase or teacher information enters the policy input.
    """
    def __init__(self, document, branch, duration, dt, max_speed, device):
        if not 0 <= branch < len(document['solutions']):
            raise ValueError('plan_branch is not present in the plan')
        if min(duration, dt, max_speed) <= 0:
            raise ValueError('Plan duration, control step and speed must be positive')
        self.points = torch.tensor(document['solutions'][branch]['waypoints'], device=device, dtype=torch.float32)
        self.delta = self.points[1:]-self.points[:-1]
        self.length = self.delta.norm(dim=-1)
        if (self.length <= 1e-8).any():
            raise ValueError('Plan contains a zero-length segment')
        self.cumulative = torch.cat((self.length.new_zeros(1), self.length.cumsum(0)))
        self.arc_step = self.cumulative[-1]*dt/duration
        self.dt, self.max_speed = dt, max_speed

    def __call__(self, observations):
        target = observations['actor'][:, 12:18]
        fraction = (((target[:, None]-self.points[:-1])*self.delta).sum(-1)
                    / self.length.square()).clamp(0., 1.)
        projection = self.points[:-1]+fraction[..., None]*self.delta
        index = (projection-target[:, None]).square().sum(-1).argmin(-1)
        row = torch.arange(len(target), device=target.device)
        arc = self.cumulative[index]+fraction[row, index]*self.length[index]
        advance = torch.minimum(self.arc_step, .2*(self.cumulative[-1]-arc).clamp_min(0.))
        next_arc = arc+advance
        segment = (torch.searchsorted(self.cumulative, next_arc, right=True)-1).clamp(0, len(self.length)-1)
        f = ((next_arc-self.cumulative[segment])/self.length[segment]).clamp(0., 1.)
        desired = self.points[segment]+f[:, None]*self.delta[segment]
        return ((desired-target)/(self.max_speed*self.dt)).clamp(-1., 1.)


def collect_demonstrations(cfg, algorithm, device):
    """Execute the plan in real dynamics; reject unsuccessful demonstrations."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .tasks.reconfiguration import plan_guided
    core_cfg = copy.deepcopy(cfg)
    core_cfg.scene.num_envs = algorithm.demo_envs
    core_cfg.terminations.pop('pose_success', None)
    core_cfg.episode_length_s = max(core_cfg.episode_length_s, algorithm.plan_duration_s+algorithm.demo_hold_s+1.)
    samples, labels = [], []
    with torch.inference_mode():
        core = ManagerBasedRlEnv(core_cfg, device=device)
        env = RslRlVecEnvWrapper(core, clip_actions=1.)
        try:
            observations, _ = env.reset()
            teacher = algorithm.make_teacher(core)
            failed = torch.zeros(core.num_envs, dtype=torch.bool, device=core.device)
            count = math.ceil((algorithm.plan_duration_s+algorithm.demo_hold_s)/core.step_dt)
            for _ in range(count):
                action = teacher(observations)
                samples.append(observations['actor'].clone())
                labels.append(action.clone())
                executed = (action+torch.randn_like(action)*algorithm.demo_action_noise).clamp(-1., 1.)
                observations, _, done, _ = env.step(executed)
                failed |= done.bool()
            pose = plan_guided.progress(core)
            achieved = (pose.count >= math.ceil(core.cfg.pose_hold_s/core.step_dt-1e-8)) & ~failed
            g = plan_guided.mdp.state(core).geometry()
            report = {'teacher_success_rate': achieved.float().mean().item(),
                      'final_max_distance_m': g['distance'].max().item(),
                      'final_max_normal_angle_rad': plan_guided.normal_angle(core).max().item(),
                      'unsafe_or_timeout_worlds': failed.sum().item(),
                      'worlds': core.num_envs, 'steps': count}
            if not achieved.all():
                raise RuntimeError('Planned demonstrations did not all reach stable pose: '+json.dumps(report))
        finally:
            env.close()
    # Clone outside InferenceMode so supervised backprop can save these inputs.
    return torch.cat(samples).clone(), torch.cat(labels).clone(), report


def evaluate_student(actor, cfg, device, worlds=4, duration=9.):
    """Teacher-free rollout from the straight reset; no plan is queried."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .tasks.reconfiguration import plan_guided
    evaluation_cfg = copy.deepcopy(cfg)
    evaluation_cfg.scene.num_envs = worlds
    evaluation_cfg.terminations.pop('pose_success',None)
    evaluation_cfg.episode_length_s = max(evaluation_cfg.episode_length_s,duration+1.)
    actor.eval()
    with torch.inference_mode():
        core = ManagerBasedRlEnv(evaluation_cfg,device=device)
        env = RslRlVecEnvWrapper(core,clip_actions=1.)
        try:
            observations,_ = env.reset()
            failed = torch.zeros(worlds,dtype=torch.bool,device=core.device)
            for _ in range(math.ceil(duration/core.step_dt)):
                observations,_,done,_ = env.step(actor(observations))
                failed |= done.bool()
            pose = plan_guided.progress(core)
            success = (pose.count >= math.ceil(core.cfg.pose_hold_s/core.step_dt-1e-8)) & ~failed
            g = plan_guided.mdp.state(core).geometry()
            result = {'success_rate':success.float().mean().item(),
                      'ever_success_rate':(pose.achieved & ~failed).float().mean().item(),
                      'min_final_hold_s':pose.count.min().item()*core.step_dt,
                      'mean_final_distance_m':g['distance'].mean().item(),
                      'max_final_distance_m':g['distance'].max().item(),
                      'max_final_normal_angle_rad':plan_guided.normal_angle(core).max().item(),
                      'unsafe_or_timeout_worlds':failed.sum().item(),
                      'worlds':worlds,'duration_s':duration}
        finally:
            env.close()
    actor.train()
    return result


class PlanImitation:
    """Actor-only supervised training. No PPO, critic, return or rollout storage."""
    def __init__(self,actor,device='cpu',**settings):
        self.actor = actor
        self.device = device
        defaults = PlanImitationCfg()
        for name in PlanImitationCfg.__dataclass_fields__:
            if name != 'class_name':
                setattr(self,name,settings.get(name,getattr(defaults,name)))
        for name in ('batch_size','demo_envs','validation_interval','validation_envs','validation_required_passes'):
            value = getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('plan_duration_s','demo_hold_s','learning_rate','max_grad_norm','validation_duration_s'):
            value = getattr(self,name)
            if not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.demo_action_noise) or self.demo_action_noise<0:
            raise ValueError('demo_action_noise must be nonnegative and finite')
        self.optimizer = torch.optim.Adam(actor.parameters(),lr=self.learning_rate)
        self.demo_observations = self.demo_actions = None
        self.demo_report = {}
        self.plan_sha256 = None
        self.completed_updates = 0
        self.validation_passes = 0
        self.best_score = (-1.,-float('inf'))
        self.last_validation = {}

    @staticmethod
    def construct_algorithm(obs,env,cfg,device):
        if cfg.get('multi_gpu'):
            raise ValueError('Plan imitation currently supports one GPU')
        actor_class,actor_cfg = resolve_class(cfg['actor'])
        cfg['obs_groups'] = resolve_obs_groups(obs,cfg['obs_groups'],['actor'])
        actor = actor_class(obs,cfg['obs_groups'],'actor',env.num_actions,**actor_cfg).to(device)
        if actor.is_recurrent or actor.distribution is not None:
            raise ValueError('Plan imitation uses a deterministic feedforward actor')
        _,settings = resolve_class(cfg['algorithm'])
        return PlanImitation(actor,device=device,**settings)

    def make_teacher(self,core):
        document = load_plan(self.plan_file)
        digest = hashlib.sha256(Path(self.plan_file).read_bytes()).hexdigest()
        if self.plan_sha256 is not None and digest!=self.plan_sha256:
            raise ValueError('Plan changed since the demonstration checkpoint')
        self.plan_sha256 = digest
        action = core.action_manager.get_term('joint_pos')
        teacher = JointPlanTeacher(document,self.plan_branch,self.plan_duration_s,
                                   core.step_dt,action.rate_cfg.max_target_speed,self.device)
        limits = action._target_limits()
        if ((teacher.points<limits[...,0].amax(0)) | (teacher.points>limits[...,1].amin(0))).any():
            raise ValueError('Plan exceeds configured joint target limits')
        return teacher

    def set_demonstrations(self,observations,actions,report):
        self.demo_observations = observations.to(self.device).detach().clone()
        self.demo_actions = actions.to(self.device).detach().clone()
        self.demo_report = report
        batch = TensorDict({'actor':self.demo_observations},batch_size=[len(observations)])
        self.actor.update_normalization(batch)

    def update(self):
        if self.demo_observations is None:
            raise RuntimeError('No validated demonstrations')
        ids = torch.randint(len(self.demo_observations),(self.batch_size,),device=self.device)
        batch = TensorDict({'actor':self.demo_observations[ids]},batch_size=[len(ids)])
        prediction = self.actor(batch)
        loss = (prediction-self.demo_actions[ids]).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
        self.optimizer.step()
        self.completed_updates += 1
        return loss.item()

    def train_mode(self):
        self.actor.train()

    def eval_mode(self):
        self.actor.eval()

    def get_policy(self):
        return self.actor

    def save(self):
        return {'actor_state_dict':self.actor.state_dict(),
                'optimizer_state_dict':self.optimizer.state_dict(),
                'plan_imitation':{'version':2,'completed_updates':self.completed_updates,
                    'plan_sha256':self.plan_sha256,'report':self.demo_report,
                    'settings':{name:getattr(self,name) for name in PlanImitationCfg.__dataclass_fields__ if name!='class_name'},
                    'observations':self.demo_observations,'actions':self.demo_actions,
                    'validation_passes':self.validation_passes,'best_score':self.best_score,
                    'last_validation':self.last_validation}}

    def load(self,saved,load_cfg=None,strict=True):
        config = {'actor':True,'optimizer':True,'iteration':True} if load_cfg is None else load_cfg
        if config.get('actor'):
            self.actor.load_state_dict(saved['actor_state_dict'],strict=strict)
        if config.get('optimizer'):
            self.optimizer.load_state_dict(saved['optimizer_state_dict'])
        resume = config.get('iteration',False)
        if resume:
            state = saved['plan_imitation']
            self.completed_updates = state['completed_updates']; self.plan_sha256 = state['plan_sha256']
            self.demo_report = state['report']; self.validation_passes = state['validation_passes']
            self.best_score = tuple(state['best_score']); self.last_validation = state['last_validation']
            if state.get('version',1)<2:
                self.validation_passes = 0
                self.best_score = (-1.,-float('inf'))
            for name,value in state['settings'].items():
                setattr(self,name,value)
            if state['observations'] is not None:
                self.demo_observations = state['observations'].to(self.device)
                self.demo_actions = state['actions'].to(self.device)
        return resume


class PlanGuidedRunner(MjlabOnPolicyRunner):
    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        algorithm = self.alg
        if algorithm.validation_passes>=algorithm.validation_required_passes:
            print('PLAN_IMITATION '+json.dumps({'event':'already_taught','updates':algorithm.completed_updates}))
            return
        if algorithm.demo_observations is None:
            observations,actions,report = collect_demonstrations(self.env.unwrapped.cfg,algorithm,self.device)
            algorithm.set_demonstrations(observations,actions,report)
            print('PLAN_IMITATION '+json.dumps({'event':'demonstrations',**report}))
        algorithm.train_mode()
        self.logger.init_logging_writer()
        for _ in range(num_learning_iterations):
            loss = algorithm.update()
            self.current_learning_iteration = algorithm.completed_updates-1
            if self.logger.writer is not None:
                self.logger.writer.add_scalar('Loss/teacher_mse',loss,algorithm.completed_updates)
            if algorithm.completed_updates%100==0:
                print('PLAN_IMITATION '+json.dumps({'event':'progress','updates':algorithm.completed_updates,'mse':loss}))
            if algorithm.completed_updates%algorithm.validation_interval==0:
                result = evaluate_student(algorithm.actor,self.env.unwrapped.cfg,self.device,
                                          algorithm.validation_envs,algorithm.validation_duration_s)
                algorithm.last_validation = result
                passed = result['success_rate']==1. and result['unsafe_or_timeout_worlds']==0
                algorithm.validation_passes = algorithm.validation_passes+1 if passed else 0
                score = (result['success_rate'],-result['mean_final_distance_m'])
                best = score>algorithm.best_score
                if best:
                    algorithm.best_score = score
                print('PLAN_IMITATION '+json.dumps({'event':'student_validation','updates':algorithm.completed_updates,**result}))
                if self.logger.writer is not None:
                    for name,value in result.items():
                        self.logger.writer.add_scalar('Student/'+name,value,algorithm.completed_updates)
                if self.logger.log_dir is not None and best:
                    self.save(str(Path(self.logger.log_dir)/'best.pt'))
            if self.logger.log_dir is not None and algorithm.completed_updates%self.cfg['save_interval']==0:
                self.save(str(Path(self.logger.log_dir)/f'model_{self.current_learning_iteration}.pt'))
            if algorithm.validation_passes>=algorithm.validation_required_passes:
                print('PLAN_IMITATION '+json.dumps({'event':'teacher_taught_student','updates':algorithm.completed_updates}))
                break
        if self.logger.log_dir is not None:
            self.save(str(Path(self.logger.log_dir)/f'model_{self.current_learning_iteration}.pt'))
        self.logger.stop_logging_writer()
