"""Closed 180-degree robot to mixed-angle assembly, with PPO motion residuals."""
from dataclasses import fields
from .base import Experiment
from ..mixed_reconfiguration_ppo import MixedReconfigurationRunner
from ..tasks.reconfiguration.mixed_ppo import motion_ppo_env_cfg
from ..jacobian_ppo import ResidualPpoCfg


class MixedReconfigurationPPO(Experiment):
    task_id = 'Mjlab-Zbot-6dof-Reconfigure-Mixed-270-270-180-90-90-PPO'
    experiment_name = 'zbot_reconfigure_mixed_270_270_180_90_90_ppo'
    runner_cls = MixedReconfigurationRunner
    train_envs = 4096

    def create_env(self, play=False):
        return motion_ppo_env_cfg(play)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.mixed_reconfiguration_ppo:MixedReconfigurationRunner'
        cfg.num_steps_per_env = 128
        cfg.algorithm = ResidualPpoCfg(**{f.name: getattr(cfg.algorithm, f.name) for f in fields(cfg.algorithm)})
        cfg.algorithm.class_name = 'zbot_rl_mjlab.mixed_reconfiguration_ppo:MotionMaskedPPO'
        cfg.algorithm.min_learning_rate = 1e-5
        cfg.algorithm.max_learning_rate = 1e-3
        cfg.algorithm.gamma = .995
        cfg.actor.distribution_cfg['init_std'] = .25
        cfg.clip_actions = 1.
        return cfg
