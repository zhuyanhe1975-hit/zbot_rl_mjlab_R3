"""Restore the site-snake iter599 recipe from its bundled source snapshot.

Only the archived task uses this 56D Beta contract. New adaptive and forward
experiments retain their independent 61D Gaussian contract.
"""
import math
from dataclasses import dataclass
from typing import Any

import torch

from .site_snake_teacher import SiteSnakeTeacherRunner
from .tasks.snake.site_omni import (
    SiteSnakeCommand,
    SiteSnakeCommandCfg,
    command,
)
from .tasks.universal import mdp


class ArchivedSiteSnakeCommand(SiteSnakeCommand):
    uses_speed_course = True


@dataclass(kw_only=True)
class ArchivedSiteSnakeCommandCfg(SiteSnakeCommandCfg):
    resampling_time_range: tuple[float, float] = (5., 9.)
    velocity_limits: tuple[float, float, float] = (.2, .2, .5)
    motion_stop_probability: float = 1 / 8
    motion_x_only_probability: float | None = None

    def build(self, env):
        return ArchivedSiteSnakeCommand(self, env)


def linear_tracking(env, std_min=.02, std_scale=.5, std_max=.1):
    """Zbot XY kernel, capped at the archived walking width of .1 m/s.

    A fixed humanoid width would give almost full credit to a stationary snake.
    Zero-command and pure-turn trials still penalize unwanted site translation.
    """
    target = command(env)
    actual = mdp.tracking_velocity(env)
    std = (std_min + std_scale * target[:, :2].norm(dim=-1)).clamp(max=std_max)
    return torch.exp(-(actual[:, :2] - target[:, :2]).square().sum(-1) / std.square())


def _axis_linear_tracking(env, axis, std):
    if not math.isfinite(std) or std <= 0:
        raise ValueError('std must be finite and positive')
    target = command(env)
    actual = mdp.tracking_velocity(env)
    return torch.exp(-((actual[:, axis] - target[:, axis]) / std).square())


def forward_tracking(env, std=.1):
    """Independent vx tracking with a fixed speed-error bandwidth."""
    return _axis_linear_tracking(env, 0, std)


def lateral_tracking(env, std=.1):
    """Independent vy tracking; lateral errors do not reduce the vx term."""
    return _axis_linear_tracking(env, 1, std)


def angular_tracking(env, std=.25):
    """Use the archived Zbot omni yaw width, including during translation."""
    target = command(env)
    actual = mdp.tracking_velocity(env)
    return torch.exp(-((actual[:, 2] - target[:, 2]) / std).square())



class ArchivedSiteSnakeTeacherRunner(SiteSnakeTeacherRunner):
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

    def _schema(self):
        schema = super()._schema()
        schema.pop('snake_target_limit')
        if schema['joint_target_limit'] != math.pi / 3 or schema['max_target_speed'] != 2 * math.pi:
            raise ValueError('archived site snake requires pi/3 target limits and 2*pi target speed')
        return schema

    @classmethod
    def _validate_checkpoint(cls, saved):
        metadata = super()._validate_checkpoint(saved)
        if metadata['joint_target_limit'] != math.pi / 3 or metadata['max_target_speed'] != 2 * math.pi:
            raise ValueError('archived site snake control limits are invalid')
        return metadata

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint):
        metadata = ArchivedSiteSnakeTeacherRunner._validate_checkpoint(
            torch.load(checkpoint, map_location='cpu', weights_only=False))
        if not getattr(env_cfg, 'archive_site_checkpoint', False):
            raise ValueError('archived checkpoint requires its restored site environment')
        command_cfg = env_cfg.commands['transform']
        command_cfg.mode = 'snake'
        command_cfg.task_observation_encoding = 'scalar'
        command_cfg.velocity_limits = metadata['velocity_limits']
        SiteSnakeTeacherRunner._restore_action_cfg(env_cfg, metadata)
