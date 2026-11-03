"""Checkpoint identity and playback for the mixed-clocked snake."""
from pathlib import Path

import torch

from .rz270_snake_teacher import Rz270SnakeTeacherRunner
from .site_snake_teacher import SiteSnakeTeacherRunner
from .tasks.snake.mixed_clocking import CLOCKING_DEG, mixed_clocking_spec


class MixedClockingSnakeTeacherRunner(Rz270SnakeTeacherRunner):
    @staticmethod
    def _metadata_schema():
        return Rz270SnakeTeacherRunner._metadata_schema() | {
            'role': 'privileged_site_snake_mixed_clocking_teacher',
            'module_clocking_deg': CLOCKING_DEG,
            'robot_assembly': 'zbot_six_270_270_180_90_90_v1',
        }

    @classmethod
    def prepare_playback_env_cfg(cls, env_cfg, checkpoint: str | Path):
        metadata = cls._validate_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
        if env_cfg.scene.entities['robot'].spec_fn is not mixed_clocking_spec:
            raise ValueError('Mixed-clocking checkpoint requires its matching robot assembly')
        command = env_cfg.commands['transform']
        command.mode = 'snake'
        command.task_observation_encoding = 'scalar'
        command.velocity_limits = metadata['velocity_limits']
        cls._restore_action_cfg(env_cfg, metadata)
        command.heading_range = metadata['heading_range']
        command.forward_speed_cap = metadata['forward_speed_cap']

    def _schema(self):
        if self.env.unwrapped.cfg.scene.entities['robot'].spec_fn is not mixed_clocking_spec:
            raise ValueError('Mixed-clocking runner requires its matching robot assembly')
        command = self.env.unwrapped.cfg.commands['transform']
        return SiteSnakeTeacherRunner._schema(self) | {
            'heading_range': tuple(command.heading_range),
            'forward_speed_cap': command.forward_speed_cap,
        }
