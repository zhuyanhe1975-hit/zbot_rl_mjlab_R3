"""Phase-masked RSL PPO for asynchronous batched mixed-clock reconfiguration."""
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO

from .teacher_separation import TeacherMaskedPPO
from .tasks.reconfiguration.mixed_ppo import MODE, moving_mask


class MotionMaskedPPO(TeacherMaskedPPO):
    def act(self, obs):
        residual = PPO.act(self, obs)
        self.pending_mask = moving_mask(self.core).clone()
        return torch.where(self.pending_mask[:, None], residual, 0.)

    def save(self):
        return PPO.save(self)

    def load(self, saved, load_cfg, strict):
        return PPO.load(self, saved, load_cfg, strict)


class MixedReconfigurationRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        if not getattr(env.unwrapped.cfg, 'mixed_reconfiguration_ppo', False):
            raise ValueError('Mixed reconfiguration PPO requires its batched full-FSM environment')
        super().__init__(env, train_cfg, log_dir, device)
        with torch.no_grad():
            head = [m for m in self.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            head.weight.zero_(); head.bias.zero_()

    @classmethod
    def prepare_playback_env_cfg(cls, env_cfg, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if (saved.get('infos', {}).get('reconfiguration_control_mode') != MODE
                or saved.get('infos', {}).get('observation_recipe') != 'full_state_guide_velocity_v1'):
            raise ValueError('Checkpoint does not belong to the mixed full-FSM PPO experiment')
        if not getattr(env_cfg, 'mixed_reconfiguration_ppo', False):
            raise ValueError('Checkpoint requires the mixed full-FSM environment')

    def save(self, path, infos=None):
        super().save(path, {**(infos or {}), 'reconfiguration_control_mode': MODE,
                          'observation_recipe': 'full_state_guide_velocity_v1',
                          'target_clocking_deg': (270, 270, 180, 90, 90),
                          'connection_order_zero_based': (0, 1, 3, 4),
                          'kinematic_guide': self.env.unwrapped.action_manager.get_term('joint_torque').guide.report,
                          'gpu_loading': self.env.unwrapped.action_manager.get_term('joint_torque').loading.report,
                          'physical_open_face_contacts': True,
                          'collision_filter': 'weld_active_constraint_bit_v1'})

    def load(self, path, *args, **kwargs):
        self.prepare_playback_env_cfg(self.env.unwrapped.cfg, path)
        return super().load(path, *args, **kwargs)
