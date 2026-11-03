"""Endpoint-relative observations and rewards for passive loop closure."""
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

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
    return torch.exp(-end_distance(env) / .2)


def alignment_reward(env):
    g = state(env).geometry()
    return torch.exp(-g['distance'] / .1 - (g['angle'] / .3).square())


def approach_velocity_cost(env):
    g = state(env).geometry()
    return torch.exp(-g['distance'] / .1) * g['speed'].square()


def connected_reward(env):
    dock = state(env)
    quarter = dock.geometry()['quarter']
    return (dock.enabled() & torch.nn.functional.one_hot(quarter, 4).bool()).any(-1).float()


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


def body_height_cost(env):
    height = state(env).robot.data.body_com_pos_w[..., 2] - env.scene.env_origins[:, None, 2]
    return (height - .15).clamp_min(0).square().mean(-1)


def end_distance(env): return state(env).geometry()['distance']
def end_angle(env): return state(env).geometry()['angle']
def relative_speed(env): return state(env).geometry()['speed']
def connection_count(env): return state(env).connections
def break_count(env): return state(env).breaks
def weld_force(env): return state(env).force
def weld_torque(env): return state(env).torque
