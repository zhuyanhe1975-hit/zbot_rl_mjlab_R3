"""Endpoint-relative observations and rewards for passive loop closure."""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .magnet import MagneticDock


def state(env):
    if not hasattr(env, '_magnetic_dock'):
        env._magnetic_dock = MagneticDock(env)
    return env._magnetic_dock


def reset_docking(env, env_ids):
    state(env).reset(env_ids)


def observations(env):
    s = state(env)
    g, delta = s.observation_geometry()
    data = s.robot.data
    action = env.action_manager.get_term('joint_pos')
    gravity = delta.new_tensor((0., 0., -1.)).expand_as(delta)
    scalar = lambda x: x.float().unsqueeze(-1)
    return torch.cat((data.joint_pos, data.joint_vel, action.commanded_target, action.raw_action,
        delta, g['relative_quat'], quat_apply_inverse(g['q0'], g['dv']),
        quat_apply_inverse(g['q0'], g['dw']), quat_apply_inverse(g['q0'], gravity),
        quat_apply_inverse(g['q1'], gravity), scalar(s.enabled().any(-1)),
        scalar((s.capture_time / s.cfg.capture_hold_s).clamp(0., 1.)), scalar((s.hold_time / s.cfg.success_hold_s).clamp(0., 1.)),
        scalar(s.force / s.cfg.break_force), scalar(s.torque / s.cfg.break_torque),
        torch.nn.functional.one_hot(g['quarter'], 4).float(),
        quat_apply_inverse(g['q0'], data.site_lin_vel_w[:, s.sites[0]]),
        quat_apply_inverse(g['q0'], data.site_ang_vel_w[:, s.sites[0]]),
        scalar(s.cooldown / s.cfg.recapture_cooldown_s)), dim=-1)


def distance_reward(env):
    return _distance_score(state(env).geometry())


def _distance_score(g):
    return torch.exp(-g['distance'] / .2)


def alignment_reward(env):
    """Facing score in [0, 1]; head +Z is inward, tail +Z is outward."""
    return _normal_score(state(env).geometry())


def _normal_score(g):
    local_z = g['delta'].new_tensor((0., 0., 1.)).expand_as(g['delta'])
    z_head = quat_apply(g['q0'], local_z)
    z_tail = quat_apply(g['q1'], local_z)
    direction = g['delta'] / g['distance'][:, None].clamp_min(1e-6)
    parallel = (z_head * z_tail).sum(-1)
    facing = torch.minimum(
        parallel,
        torch.minimum(-(z_head * direction).sum(-1), -(z_tail * direction).sum(-1)),
    )
    # At coincident origins the line of sight is undefined; compare the axes only.
    score = torch.where(g['distance'] < 1e-6, parallel, facing)
    return (1. + score.clamp(-1., 1.)) * .5


def distance_closing_speed(env):
    """Head/tail origin relative velocity projected toward coincidence, m/s."""
    g = state(env).geometry()
    direction = g['delta'] / g['distance'][:, None].clamp_min(1e-6)
    closing = -(g['dv'] * direction).sum(-1)
    return torch.where(g['distance'] >= 1e-6, closing, torch.zeros_like(closing))


def alignment_closing_rate(env):
    """Decrease rate of the registered relative axis-angle error, rad/s."""
    g = state(env).geometry()
    relative = g['relative_quat']
    relative = torch.where(relative[:, :1] < 0., -relative, relative)
    vector = relative[:, 1:]
    length = vector.norm(dim=-1)
    axis = vector / length[:, None].clamp_min(1e-6)
    relative_omega = quat_apply_inverse(g['q0'], g['dw'])
    closing = -(axis * relative_omega).sum(-1)
    # At identity the axis is undefined; at pi the shortest-angle branch has a
    # cusp. Neutral rates avoid assigning an arbitrary signed reward there.
    regular = (length > 1e-6) & (relative[:, 0].abs() > 1e-6)
    return torch.where(regular, closing, torch.zeros_like(closing))


def distance_velocity_reward(env):
    return distance_closing_speed(env)


def alignment_velocity_reward(env):
    return alignment_closing_rate(env)


def approach_velocity_cost(env):
    g = state(env).geometry()
    return torch.exp(-g['distance'] / .1) * g['speed'].square()


def connected_reward(env):
    return state(env).enabled().any(-1).float()


def success_reward(env):
    return state(env).just_succeeded.float() / env.step_dt


def success(env):
    return state(env).succeeded


def unsafe(env):
    data = state(env).robot.data
    pos = data.root_link_pos_w - env.scene.env_origins
    return (~torch.isfinite(data.joint_pos).all(-1) | ~torch.isfinite(data.joint_vel).all(-1)
            | ~torch.isfinite(pos).all(-1) | (pos[:, :2].norm(dim=-1) > env.cfg.workspace_radius)
            | (pos[:, 2] < -.1) | (pos[:, 2] > 2.))


def joint_velocity_cost(env):
    return state(env).robot.data.joint_vel.square().mean(-1)


def limit_proximity_cost(env, margin: float = .3):
    """Persistent nonnegative cost near the effective commanded joint bounds.

    Per joint, use the worse of actual angle and integrated target so tracking
    error is not charged twice. The squared hinge has zero slope on entering
    the buffer, and actual overshoot is not capped. Raw actions are velocities.
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('limit_proximity margin must be positive and finite')
    action = env.action_manager.get_term('joint_pos')
    limits = action._target_limits()
    actual = action._entity.data.joint_pos[:, action._target_ids]
    target = action.commanded_target
    low, high = limits[..., 0], limits[..., 1]
    actual_clearance = torch.minimum(actual - low, high - actual)
    target_clearance = torch.minimum(target - low, high - target)
    clearance = torch.minimum(actual_clearance, target_clearance)
    return ((margin - clearance).clamp_min(0.) / margin).square().mean(-1)


def end_distance(env): return state(env).geometry()['distance']
def end_angle(env): return state(env).geometry()['angle']
def relative_speed(env): return state(env).geometry()['speed']
def connection_count(env): return state(env).connections
def break_count(env): return state(env).breaks
def weld_force(env): return state(env).force
def weld_torque(env): return state(env).torque
