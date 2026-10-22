"""Separate checkpoint contracts for forward-only diagnostic training."""
import torch

from .site_snake_teacher import SiteSnakeTeacherRunner


def _prepare_playback(env_cfg, checkpoint, runner_cls):
    metadata = runner_cls._validate_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False))
    if not getattr(env_cfg, 'site_snake_omni', False):
        raise ValueError('forward checkpoint requires a site snake environment')
    command = env_cfg.commands['transform']
    command.mode = 'snake'
    command.task_observation_encoding = 'scalar'
    command.velocity_limits = metadata['velocity_limits']
    SiteSnakeTeacherRunner._restore_action_cfg(env_cfg, metadata)


class ForwardSnakeTeacherRunner(SiteSnakeTeacherRunner):
    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint):
        _prepare_playback(env_cfg, checkpoint, ForwardSnakeTeacherRunner)

    @staticmethod
    def _metadata_schema():
        return SiteSnakeTeacherRunner._metadata_schema() | {
            'role': 'site_snake_forward_baseline',
            'reward_recipe': 'site_snake_forward_baseline_clipped_v4',
        }


class RelaxedForwardSnakeTeacherRunner(ForwardSnakeTeacherRunner):
    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint):
        _prepare_playback(env_cfg, checkpoint, RelaxedForwardSnakeTeacherRunner)

    @staticmethod
    def _metadata_schema():
        return ForwardSnakeTeacherRunner._metadata_schema() | {
            'role': 'site_snake_forward_relaxed',
            'reward_recipe': 'site_snake_forward_relaxed_clipped_v4',
        }
