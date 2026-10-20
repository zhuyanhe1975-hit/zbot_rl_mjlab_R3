"""Independent privileged site-frame snake teacher and qualified snapshots."""

import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.modules import BetaDistribution

from .omni_teacher import OmniTeacherRunner, OmniTeacherRunnerCfg, omni_teacher_ppo_cfg
from .tasks.snake.site_omni import SiteSnakeCommandCfg, SiteSnakeEnvCfg
from .tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg


def site_snake_ppo_cfg() -> OmniTeacherRunnerCfg:
    cfg = omni_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.site_snake_teacher:SiteSnakeTeacherRunner'
    # Successful snake iter1499:256/128 ELU, normalized Beta,1e-3/.002.
    cfg.algorithm.learning_rate = 1e-3
    cfg.algorithm.entropy_coef = .002
    cfg.console_mode = 'full'
    cfg.validation_interval_iterations = 0
    cfg.robust_validation_interval = 0
    return cfg


class SiteSnakeTeacherRunner(OmniTeacherRunner):
    @staticmethod
    def _metadata_schema() -> dict[str, Any]:
        return {
            'role': 'privileged_site_snake_omni_teacher', 'schema_version': 3,
            'actor_obs_dim': 56, 'critic_obs_dim': 56, 'action_dim': 6,
            'snake_omni_version': 3, 'objective': 'site_velocity_tracking',
            'tracking_point': 'foot_0_sole', 'reward_recipe': 'zbot_snake_velocity_limits_tilt_v5',
            'motion_site': 'foot_0_sole',
            'command_axes': ('site_minus_z_gravity_aligned', 'left', 'world_plus_z'),
            'distribution': 'BetaDistribution', 'action_range': (-1., 1.),
            'action_control': 'integrated_joint_target_velocity',
            'qualification_protocol': 'site_snake_omni_v3',
        }

    @classmethod
    def _validate_checkpoint(cls, saved: dict) -> dict:
        metadata = (saved.get('infos') or {}).get('privileged_omni_teacher')
        if not isinstance(metadata, dict) or any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value
            for key, value in cls._metadata_schema().items()
        ):
            raise ValueError('checkpoint is not a compatible site-frame snake omni teacher')
        limits = metadata.get('velocity_limits')
        if (not isinstance(limits, tuple) or len(limits) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in limits)
                or metadata.get('joint_target_limit') != math.pi / 3
                or metadata.get('max_target_speed') != 2 * math.pi):
            raise ValueError('site snake checkpoint control limits are invalid')
        return metadata

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        metadata = SiteSnakeTeacherRunner._validate_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
        if not getattr(env_cfg, 'site_snake_omni', False):
            raise ValueError('site snake checkpoint requires its independent environment')
        command = env_cfg.commands['transform']
        command.mode = 'snake'
        command.task_observation_encoding = 'scalar'
        command.velocity_limits = metadata['velocity_limits']

    def _schema(self) -> dict[str, Any]:
        schema = super()._schema()
        env_cfg = cast(SiteSnakeEnvCfg, self.env.unwrapped.cfg)
        if not getattr(env_cfg, 'site_snake_omni', False):
            raise ValueError('site snake runner requires its independent environment')
        if not env_cfg.omni_site_frame or env_cfg.motion_site != 'foot_0_sole':
            raise ValueError('site snake teacher requires the foot_0_sole coordinate frame')
        command = cast(SiteSnakeCommandCfg, env_cfg.commands['transform'])
        action = cast(SnakeCrawlJointTargetRateActionCfg, env_cfg.actions['joint_pos'])
        if (schema['actor_obs_dim'], schema['critic_obs_dim'], schema['action_dim']) != (56, 56, 6) or command.mode != 'snake':
            raise ValueError('site snake teacher requires 56-dimensional observations,6 actions,and fixed snake mode')
        if action.target_limit != math.pi / 3 or action.max_target_speed != 2 * math.pi:
            raise ValueError('site snake control contract requires pi/3 target limits and 2*pi target speed')
        return schema | self._metadata_schema() | {
            'velocity_limits': command.velocity_limits,
            'joint_target_limit': action.target_limit,
            'max_target_speed': action.max_target_speed,
        }

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        distribution = self.alg.actor.distribution
        if not isinstance(distribution, BetaDistribution) or tuple(distribution.action_range) != (-1., 1.):
            raise ValueError('site snake teacher requires the archived bounded Beta policy')
        self._schema()
        from .site_snake_evaluation import SiteSnakeEvaluator

        # The inherited progress wrapper reads these attributes at update time.
        # Replace its generic20-scenario evaluator with the snake-only protocol.
        for name, interval, randomized in (
            ('_evaluator', self._validation_interval, False),
            ('_robust_evaluator', self._robust_interval, True),
        ):
            previous = getattr(self, name)
            if previous is not None:
                previous.close()
            setattr(self, name, SiteSnakeEvaluator(device=device, env_cfg=env.unwrapped.cfg, randomized=randomized)
                    if self._training and interval else None)
        if self._training:
            from .training_console import (
                install_compact_logging,
                install_site_snake_rollout_logging,
            )

            if train_cfg.get('console_mode', 'compact') == 'full':
                install_compact_logging(self.logger, rewards_only=True)
            install_site_snake_rollout_logging(self.logger, env.unwrapped)
            directory = Path(cast(str, log_dir)) / 'qualification'
            update = self.alg.update

            def update_and_record_qualification():
                losses = dict(update())
                if 'snake_omni_passed' in losses and self._evaluator is not None:
                    report = dict(cast(SiteSnakeEvaluator, self._evaluator).last_report)
                    report['completed_updates'] = self.completed_ppo_iterations
                    directory.mkdir(parents=True, exist_ok=True)
                    qualified = bool(losses['snake_omni_passed'] and report.get('passed') is True
                                     and report.get('duration_eligible') is True
                                     and report.get('protocol') == self._metadata_schema()['qualification_protocol'])
                    report['passed'] = qualified
                    losses['snake_omni_passed'] = float(qualified)
                    if qualified:
                        checkpoint = directory / f'model_{self.completed_ppo_iterations}.pt'
                        report['qualified_checkpoint'] = str(checkpoint)
                        self.save(str(checkpoint), {'snake_site_qualification': report})
                    (directory / 'latest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
                return losses

            self.alg.update = update_and_record_qualification

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        self._validate_checkpoint(saved)
        for name, model in (('actor', self.alg.actor), ('critic', self.alg.critic)):
            if load_cfg is not None and not load_cfg.get(name, False):
                continue
            current = model.state_dict()
            incoming = saved.get(f'{name}_state_dict', {})
            if current.keys() != incoming.keys() or any(current[key].shape != incoming[key].shape for key in current):
                raise ValueError('site snake checkpoint model architecture mismatch')
        return super().load(path, load_cfg, strict, map_location)
