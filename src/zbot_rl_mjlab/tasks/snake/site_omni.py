"""Standalone privileged snake teacher with commands relative to foot_0_sole.

G1-style exponential velocity tracking adapted to crawling. Both the
tracked point and the command frame are foot_0_sole; no world heading is locked.
"""
import math
from dataclasses import asdict, dataclass
from typing import cast

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from ..universal import mdp
from ..universal.actions import SnakeCrawlJointTargetRateActionCfg
from ..universal.env_cfg import universal_env_cfg
from ..universal.omni import motion_site_index, reset_omni
from ..universal.privileged import (
    PrivilegedOmniCommand,
    PrivilegedOmniCommandCfg,
    privileged_state,
)
from ..universal.teacher_rewards import ArchiveOmniTeacherEnvCfg
from . import mdp as snake_mdp


@dataclass(kw_only=True)
class SiteSnakeCommandCfg(PrivilegedOmniCommandCfg):
    resampling_time_range: tuple[float, float] = (5., 9.)
    velocity_limits: tuple[float, float, float] = (.2, .2, .5)
    motion_axis_min_fraction: float = .4
    mode: str = 'snake'

    def __post_init__(self):
        if self.mode != 'snake':
            raise ValueError("Site snake teacher requires mode='snake'")

    def build(self, env):
        return SiteSnakeCommand(self, env)


class SiteSnakeCommand(PrivilegedOmniCommand):
    def __init__(self, cfg: SiteSnakeCommandCfg, env):
        if cfg.mode != 'snake':
            raise ValueError("Site snake teacher requires mode='snake'")
        super().__init__(cfg, env)

    def set_target(self, target, env_ids=None):
        if target != 'snake':
            raise ValueError('Site snake teacher only accepts the snake target; standing is disabled')
        super().set_target(target, env_ids)


@dataclass
class SiteSnakeEnvCfg(ArchiveOmniTeacherEnvCfg):
    site_snake_omni: bool = True
    site_up_axis: tuple[float, float, float] = (1., 0., 0.)
    site_tilt_limit: float = math.pi / 3
    head_body: str = 'foot_0'
    head_up_axis: tuple[float, float, float] = (1., 0., 0.)
    head_tilt_limit: float = math.pi / 3
    max_contact_bodyweights: float = 8.


def command(env):
    return mdp.command_term(env).velocity_command


def linear_tracking(env, std_min=.02, std_scale=.5, std_max=.1):
    """Zbot XY kernel, capped at the archived walking width of .1 m/s.

    A fixed humanoid width would give almost full credit to a stationary snake.
    Zero-command and pure-turn trials still penalize unwanted site translation.
    """
    target = command(env)
    actual = mdp.tracking_velocity(env)
    std = (std_min + std_scale * target[:, :2].norm(dim=-1)).clamp(max=std_max)
    return torch.exp(-(actual[:, :2] - target[:, :2]).square().sum(-1) / std.square())


def angular_tracking(env, std=.25):
    """Use the archived Zbot omni yaw width, including during translation."""
    target = command(env)
    actual = mdp.tracking_velocity(env)
    return torch.exp(-((actual[:, 2] - target[:, 2]) / std).square())


def joint_limit_penalty(env, soft_limit_factor=.9, use_target=False):
    """G1-style summed angular excess, using the snake's commanded angle limit.

    Penalizing the integrated target too makes sustained saturation costly even
    when PD tracking keeps actual joints just short of the soft boundary.
    """
    if use_target:
        position = env.action_manager.get_term('joint_pos').commanded_target
    else:
        position = env.scene['robot'].data.joint_pos
    limit = env.cfg.actions['joint_pos'].snake_target_limit
    return (position.abs() - soft_limit_factor * limit).clamp_min(0.).sum(-1)


def _site_up_cosine(env):
    quat = env.scene['robot'].data.site_quat_w[:, motion_site_index(env)]
    gravity = quat.new_tensor(env.cfg.sim.mujoco.gravity)
    gravity = gravity / gravity.norm().clamp_min(1e-6)
    local_gravity = quat_apply_inverse(quat, gravity.expand(quat.shape[0], -1))
    up = quat.new_tensor(env.cfg.site_up_axis)
    return -(local_gravity * up).sum(-1).clamp(-1., 1.)


def site_tilt(env):
    """Tilt of the site's canonical up axis relative to gravity, independent of yaw."""
    return _site_up_cosine(env).acos()


def site_tilt_penalty(env):
    return 1 - _site_up_cosine(env)


def site_tilt_limit_reached(env):
    # A small tolerance prevents float rounding from resetting a pose at exactly 60 degrees.
    return site_tilt(env) > env.cfg.site_tilt_limit + 1e-6


def velocity_error(env, axis):
    return (mdp.tracking_velocity(env)[:, axis] - command(env)[:, axis]).abs()


def command_exposure(env, axis, sign):
    target = command(env)
    pure = (target.abs() > 1e-6).sum(-1) == 1
    return (pure & (target[:, axis] * sign > 1e-6)).float()


def site_height(env):
    return (env.scene['robot'].data.site_pos_w[:, motion_site_index(env), 2]
            - env.scene.env_origins[:, 2])


def site_snake_env_cfg(play=False) -> SiteSnakeEnvCfg:
    cfg = cast(SiteSnakeEnvCfg, universal_env_cfg(play=play, cfg_cls=SiteSnakeEnvCfg))
    cfg.omni_site_frame = True
    cfg.motion_site = 'foot_0_sole'
    cfg.workspace_radius = 15.
    cfg.episode_length_s = 30.
    cfg.auto_reset = not play
    action = SnakeCrawlJointTargetRateActionCfg(**asdict(cfg.actions['joint_pos']))
    action.target_limit = math.pi / 3
    action.snake_target_limit = math.pi / 3
    action.max_target_speed = 2 * math.pi
    action.biped_residual_scale = None
    action.transition_end_joint_radius = None
    cfg.actions['joint_pos'] = action
    cfg.commands = {'transform': SiteSnakeCommandCfg(
        training=not play, mode='snake', task_observation_encoding='scalar',
        debug_vis=play,
        play_random_velocity=play,
    )}
    cfg.events = {
        'reset_scene': EventTermCfg(func=common_mdp.reset_scene_to_default, mode='reset'),
        'reset_transition': EventTermCfg(func=reset_omni, mode='reset'),
    }
    cfg.observations = {'actor': ObservationGroupCfg(
        {'privileged': ObservationTermCfg(func=privileged_state)},
        concatenate_terms=True, enable_corruption=False, nan_policy='disabled',
    )}
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name='snake_self_contact',
        primary=ContactMatch(mode='subtree', pattern='foot_0', entity='robot'),
        secondary=ContactMatch(mode='subtree', pattern='foot_0', entity='robot'),
        fields=('found', 'force', 'dist'), reduce='maxforce', history_length=cfg.decimation,
    ))
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.rewards = {
        'track_linear_velocity': RewardTermCfg(
            func=linear_tracking, weight=4., params={'std_min': .02, 'std_scale': .5, 'std_max': .1}),
        'track_angular_velocity': RewardTermCfg(
            func=angular_tracking, weight=1., params={'std': .25}),
        'site_tilt': RewardTermCfg(func=site_tilt_penalty, weight=-1.0),
        'joint_pos_limits': RewardTermCfg(
            func=joint_limit_penalty, weight=-1., params={'soft_limit_factor': .9}),
        'joint_target_limits': RewardTermCfg(
            func=joint_limit_penalty, weight=-1., params={'soft_limit_factor': .9, 'use_target': True}),
    }
    cfg.terminations = {} if play else {
        'self_collision': TerminationTermCfg(func=snake_mdp.self_collision),
        'site_tilt': TerminationTermCfg(func=site_tilt_limit_reached),
    }
    cfg.metrics = {
        **{f'velocity_error_{name}': MetricsTermCfg(func=velocity_error, params={'axis': axis})
           for axis, name in enumerate(('vx', 'vy', 'wz'))},
        **{f'command_{name}_{direction}_fraction': MetricsTermCfg(
            func=command_exposure, params={'axis': axis, 'sign': sign})
           for axis, name in enumerate(('vx', 'vy', 'wz'))
           for sign, direction in ((1, 'positive'), (-1, 'negative'))},
        'site_height': MetricsTermCfg(func=site_height),
        'site_tilt_rad': MetricsTermCfg(func=site_tilt),
    }
    cfg.viewer.lookat = (0., 0., .05)
    return cfg
