"""Batched native-PD action and GPU Jacobian feedforward for mixed PPO."""
from dataclasses import dataclass
import torch

from .disconnect.env_cfg import TorqueAction,TorqueActionCfg
from .disconnect.state import state
from .batched_state import BatchedState
from .batched_controller import BatchedController,BatchedGuide
from .batched_loading import BatchedLoading
from .batched_collision import install_model_contact_filter
from .state_machine import Phase


@dataclass(kw_only=True)
class BatchedMotionActionCfg(TorqueActionCfg):
    def build(self,env):return BatchedMotionAction(self,env)


class BatchedMotionAction(TorqueAction):
    def __init__(self,cfg,env):
        feedforward=env.cfg.jacobian_feedforward;distance=env.cfg.separation_distance_jacobian
        env.cfg.jacobian_feedforward=False;env.cfg.separation_distance_jacobian=False
        try:super().__init__(cfg,env)
        finally:env.cfg.jacobian_feedforward=feedforward;env.cfg.separation_distance_jacobian=distance
        install_model_contact_filter(env)
        env.sim.expand_model_fields(('eq_solref',))
        env._disconnect_state=BatchedState(env)
        self.jacobian_torque=torch.zeros_like(self._raw)
        self.jacobian_qref=self.position_target.clone()
        self.motion_action=torch.zeros_like(self._raw)
        self.nominal_velocity=torch.zeros_like(self._raw)
        self.guide_position=torch.zeros_like(self._raw)
        self.progress_before=torch.zeros(env.num_envs,device=env.device)
        self.loading=BatchedLoading(env)
        self.guide=BatchedGuide(env)
        self.distance_controller=BatchedController(env,self.modules)
        self.jacobian_report=self.loading.report

    def update_jacobian_gain(self,s):pass

    def jacobian_command(self,positions=None,active=None):
        s=state(self._env)
        positions=torch.cat([m.data.joint_pos for m in self.modules],-1) if positions is None else positions
        active=s.connected() if active is None else active
        valid=(self.distance_controller.phase==Phase.LOAD)&active.all(-1)&~s.waiting&~s.safety_margin_violation&~s.wrong_break&~self.torque_failure&((positions-self.jacobian_qref).abs().amax(-1)<.05)
        return self.jacobian_torque*valid[:,None]

    def process_actions(self,actions):
        from .mixed_ppo import progress,moving_mask,failed
        e=self._env;s=state(e);c=self.distance_controller
        self.progress_before.copy_(progress(e))
        self.motion_action.copy_(torch.where(moving_mask(e)[:,None],torch.nan_to_num(actions).clamp(-1.,1.),0.))
        loading=(c.phase==Phase.LOAD)&s.connected().all(-1)&~failed(e)
        self.loading.update(s,loading)
        self.jacobian_torque.copy_(self.loading.torque)
        q=torch.cat([m.data.joint_pos for m in self.modules],-1)
        self.jacobian_qref.copy_(torch.where(loading[:,None],q,self.jacobian_qref))
        super().process_actions(torch.zeros_like(actions))

    def apply_actions(self):
        s=state(self._env)
        s.waiting.copy_(self.distance_controller.phase==Phase.FAILED)
        super().apply_actions()

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        super().reset(env_ids)
        self.motion_action[ids]=0.;self.nominal_velocity[ids]=0.;self.progress_before[ids]=0.
        self.guide_position[ids]=self.position_target[ids]
        self.loading.reset(ids);self.jacobian_torque[ids]=0.;self.jacobian_qref[ids]=self.position_target[ids]
