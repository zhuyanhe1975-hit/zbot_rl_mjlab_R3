"""Standalone snake teacher with velocities expressed in the current head frame.

Signed scalar velocity tracking adapted to crawling. The
tracked point and gravity-aligned velocity frame both follow foot_0_sole.
"""
import math
from dataclasses import dataclass

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.utils.lab_api.math import quat_apply_inverse

from ..common import constant
from ..transformation.poses import SNAKE_JOINT_POS
from ..universal import mdp
from ..universal.observation_layout import (
    PRIVILEGED_OMNI_OBS_DIM,
    teacher_motion_fields,
)
from ..universal.omni import motion_site_index
from ..universal.privileged import (
    PrivilegedOmniCommand,
    PrivilegedOmniCommandCfg,
    privileged_state,
)
from ..universal.teacher_rewards import ArchiveOmniTeacherEnvCfg


@dataclass(kw_only=True)
class SiteSnakeCommandCfg(PrivilegedOmniCommandCfg):
    resampling_time_range: tuple[float, float] = (15., 15.)
    velocity_limits: tuple[float, float, float] = (0.2, 0.2, 2.0)
    motion_axis_min_fraction: float = .4
    motion_stop_probability: float = .01
    mode: str = 'snake'

    def __post_init__(self):
        if self.mode != 'snake':
            raise ValueError("Site snake teacher requires mode='snake'")

    def build(self, env):
        return SiteSnakeCommand(self, env)


class SiteSnakeCommand(PrivilegedOmniCommand):
    # Continuous command sampling does not use the inherited deployment stages.
    uses_speed_course = False

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
    archive_site_checkpoint: bool = False
    command_visual_height: float = 0.
    command_visual_actual_velocity: bool = False
    site_up_axis: tuple[float, float, float] = (1., 0., 0.)
    site_tilt_limit: float = math.pi / 3
    head_body: str = 'foot_0'
    head_up_axis: tuple[float, float, float] = (1., 0., 0.)
    head_tilt_limit: float = math.pi / 3
    max_contact_bodyweights: float = 8.


def command(env):
    return mdp.command_term(env).velocity_command


def site_snake_state(env):
    """61D motion-frame state: explicit velocity tracking error and local gravity."""
    state = privileged_state(env)
    if state.shape[-1] != PRIVILEGED_OMNI_OBS_DIM:
        raise ValueError('site snake observations require scalar task encoding')
    target, actual, _ = teacher_motion_fields(state)
    tracking_error = _normalized_velocity_error(env, target, actual)
    values = (state[..., :46], state[..., 47:],
              tracking_error, site_projected_gravity(env))
    return torch.nan_to_num(torch.cat(values, -1), nan=0., posinf=0., neginf=0.)


def _normalized_velocity_error(env, target, actual):
    """Signed command-minus-site velocity error, scaled by each axis's command limit."""
    limits = constant(target, mdp.command_term(env).cfg.velocity_limits)
    return (target - actual) / limits


def relative_velocity_scores(target, actual, limits):
    """Linear relative tracking score: stationary=-1, half-speed=0, matched=1."""
    moving = target.abs() > 1e-6
    scale = torch.where(moving, target.abs(), limits)
    error = (target - actual).abs() / scale
    return torch.where(moving, 1. - 2. * error, -actual.abs() / limits).clamp(-2., 1.)


def _velocity_tracking_reward(env):
    target = command(env)
    actual = mdp.tracking_velocity(env)
    limits = constant(target, mdp.command_term(env).cfg.velocity_limits)
    return relative_velocity_scores(target, actual, limits)


def linear_tracking(env):
    """Sum forward/lateral relative tracking scores in the current head frame."""
    return _velocity_tracking_reward(env)[:, :2].sum(-1)


def forward_tracking(env):
    return _velocity_tracking_reward(env)[:, 0]


def lateral_tracking(env):
    return _velocity_tracking_reward(env)[:, 1]


def angular_tracking(env):
    """Signed site yaw-rate tracking without a yaw-angle target."""
    return _velocity_tracking_reward(env)[:, 2]


def failure_reset_cost(env):
    """One fixed event cost for a failed episode, independent of the control step size."""
    return common_mdp.is_terminated(env) / env.step_dt


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


def joint_pose_deviation_cost(env):
    """Mean squared joint deviation from the straight snake reset pose."""
    position = env.scene['robot'].data.joint_pos
    return (position - constant(position, SNAKE_JOINT_POS)).square().mean(-1)


def site_projected_gravity(env):
    """Unit gravity expressed in the actual snake-head site frame, independent of world yaw."""
    quat = env.scene['robot'].data.site_quat_w[:, motion_site_index(env)]
    gravity = quat.new_tensor(env.cfg.sim.mujoco.gravity)
    gravity = gravity / gravity.norm().clamp_min(1e-6)
    return quat_apply_inverse(quat, gravity.expand(quat.shape[0], -1))


def site_tilt_angular_velocity_penalty(env):
    """Squared site angular velocity perpendicular to gravity; yaw is excluded."""
    angular = env.scene['robot'].data.site_ang_vel_w[:, motion_site_index(env)]
    gravity = constant(angular, env.cfg.sim.mujoco.gravity)
    gravity = gravity / gravity.norm().clamp_min(1e-6)
    tilt_rate = angular - (angular * gravity).sum(-1, keepdim=True) * gravity
    return tilt_rate.square().sum(-1)


def site_yaw_velocity_penalty(env):
    """Squared actual motion-site wz, independent of the yaw-rate command."""
    return env.scene['robot'].data.site_ang_vel_w[:, motion_site_index(env), 2].square()


def _site_up_cosine(env):
    local_gravity = site_projected_gravity(env)
    up = constant(local_gravity, env.cfg.site_up_axis)
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
