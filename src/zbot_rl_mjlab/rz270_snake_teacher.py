"""Keep checkpoints for the 270-degree assembly separate from old robots."""

from pathlib import Path

import torch

from .site_snake_teacher import SiteSnakeTeacherRunner
from .tasks.snake.rz270 import rz270_spec


class Rz270SnakeTeacherRunner(SiteSnakeTeacherRunner):
    @staticmethod
    def _metadata_schema():
        return SiteSnakeTeacherRunner._metadata_schema() | {
            'role': 'privileged_site_snake_rz270_teacher',
            'module_clocking_deg': 270,
            'robot_assembly': 'zbot_six_rz270_v1',
            'actor_obs_dim': 63, 'critic_obs_dim': 63,
            'objective': 'head_site_forward_speed_and_sampled_heading',
            'observation_recipe': 'site_head_frame_with_heading_sin_cos_v1',
            'reward_recipe': 'rz270_head_site_forward_heading_v1',
            'qualification_protocol': 'rz270_sampled_heading_v1',
        }

    @classmethod
    def _validate_checkpoint(cls, saved):
        import math

        metadata = super()._validate_checkpoint(saved)
        heading = metadata.get('heading_range')
        cap = metadata.get('forward_speed_cap')
        if (not isinstance(heading, tuple) or len(heading) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in heading)
                or not -math.pi <= heading[0] <= heading[1] <= math.pi
                or type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0):
            raise ValueError('invalid RZ270 heading command metadata')
        return metadata

    @classmethod
    def prepare_playback_env_cfg(cls, env_cfg, checkpoint: str | Path):
        metadata = cls._validate_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
        if env_cfg.scene.entities['robot'].spec_fn is not rz270_spec:
            raise ValueError('RZ270 checkpoint requires the RZ270 robot assembly')
        command = env_cfg.commands['transform']
        command.mode = 'snake'
        command.task_observation_encoding = 'scalar'
        command.velocity_limits = metadata['velocity_limits']
        cls._restore_action_cfg(env_cfg, metadata)
        command.heading_range = metadata['heading_range']
        command.forward_speed_cap = metadata['forward_speed_cap']

    def _schema(self):
        if self.env.unwrapped.cfg.scene.entities['robot'].spec_fn is not rz270_spec:
            raise ValueError('RZ270 runner requires the RZ270 robot assembly')
        command = self.env.unwrapped.cfg.commands['transform']
        return super()._schema() | {
            'heading_range': tuple(command.heading_range),
            'forward_speed_cap': command.forward_speed_cap,
        }
