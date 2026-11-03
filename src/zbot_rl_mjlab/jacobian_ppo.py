"""PPO loading residual, followed by deterministic distance-Jacobian native PD."""
from dataclasses import dataclass

import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlOnPolicyRunnerCfg
from rsl_rl.algorithms import PPO
from mjlab.rl.config import RslRlPpoAlgorithmCfg
from .teacher_separation import TeacherMaskedPPO

MODE='jacobian_loading_deterministic_distance_pd_v3'


@dataclass
class ResidualPpoCfg(RslRlPpoAlgorithmCfg):
    min_learning_rate: float=1e-7
    max_learning_rate: float=1e-4


@dataclass
class JacobianPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    pass


class ResidualMaskedPPO(TeacherMaskedPPO):
    def act(self,obs):
        from .tasks.reconfiguration.disconnect.state import state
        residual=PPO.act(self,obs)
        waiting=state(self.core).waiting
        # Only loading uses the policy; deterministic separation/waiting are masked.
        self.pending_mask=(~waiting & state(self.core).connected()[:,self.core.cfg.target_connector]).clone()
        return torch.where(self.pending_mask[:,None],residual,0.)

    def save(self):return PPO.save(self)

    def load(self,saved,load_cfg,strict):return PPO.load(self,saved,load_cfg,strict)


class ResidualPolicy(torch.nn.Module):
    def __init__(self,actor,target_connector=5):
        super().__init__();self.actor=actor;self.target_connector=target_connector

    def forward(self,obs,stochastic_output=False):
        action=self.actor(obs,stochastic_output=stochastic_output)
        dim=obs['actor'].shape[-1]
        if 'critic' in obs and obs['critic'].shape[-1]==dim+2:
            action=torch.where(obs['critic'][:,dim:dim+1]>.5,0.,action)
        action=torch.where(obs['actor'][:,24+self.target_connector:25+self.target_connector]<.5,0.,action)
        return action

    def reset(self,*args,**kwargs):return self.actor.reset(*args,**kwargs)


class JacobianPpoRunner(MjlabOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        if not env.unwrapped.cfg.jacobian_feedforward or not env.unwrapped.cfg.enable_weld_failure or not env.unwrapped.cfg.separation_velocity_pd or not env.unwrapped.cfg.separation_distance_jacobian:
            raise ValueError('This mode requires Jacobian loading, passive weld failure and deterministic distance PD')
        if not env.unwrapped.action_manager.get_term('joint_torque').native_pd:
            raise ValueError('Distance separation requires MJLab BuiltinPositionActuatorCfg')
        super().__init__(env,train_cfg,log_dir,device)
        with torch.no_grad():
            final=[layer for layer in self.alg.actor.mlp.modules() if isinstance(layer,torch.nn.Linear)][-1]
            final.weight.zero_();final.bias.zero_()
        print('JACOBIAN_FEEDFORWARD',env.unwrapped.action_manager.get_term('joint_torque').jacobian_report)

    def get_inference_policy(self,device=None):
        return ResidualPolicy(self.alg.get_policy().to(device or self.device).eval(),self.env.unwrapped.cfg.target_connector).to(device or self.device).eval()

    def save(self,path,infos=None):
        report=self.env.unwrapped.action_manager.get_term('joint_torque').jacobian_report
        super().save(path,{**(infos or {}),'disconnect_control_mode':MODE,
                          'jacobian_feedforward':report,'target_connector':self.env.unwrapped.cfg.target_connector})

    def load(self,path,*args,**kwargs):
        saved=torch.load(path,map_location=self.device,weights_only=False);infos=saved.get('infos',{})
        if infos.get('disconnect_control_mode')!=MODE or infos.get('target_connector',5)!=self.env.unwrapped.cfg.target_connector:
            raise ValueError('旧教师/高斯/其他目标模型不能作为统一雅可比残差PPO模型加载；旧文件保留。')
        return super().load(path,*args,**kwargs)
