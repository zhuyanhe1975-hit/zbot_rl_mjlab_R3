"""Planning-guided variant: geometric pose success independent of weld activation."""
import math
from dataclasses import dataclass, fields

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_apply

from .basic import mdp
from .basic.env_cfg import ReconfigurationEnvCfg, head_tail_env_cfg


@dataclass
class PlanGuidedEnvCfg(ReconfigurationEnvCfg):
    pose_distance_tolerance: float = .004
    pose_normal_tolerance: float = math.radians(8.)
    pose_linear_speed_tolerance: float = .1
    pose_angular_speed_tolerance: float = 1.
    pose_hold_s: float = .5


class _PoseProgress:
    def __init__(self, env):
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.achieved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.step = None

    def reset(self, ids):
        ids = slice(None) if ids is None else ids
        self.count[ids] = 0
        self.achieved[ids] = False

    def update(self, env):
        if self.step != env.common_step_counter:
            g = mdp.state(env).geometry()
            angle = normal_angle(env, g)
            z = g['delta'].new_tensor((0., 0., 1.)).expand_as(g['delta'])
            axis = quat_apply(g['q0'], z)
            perpendicular_omega = g['dw'] - (g['dw']*axis).sum(-1, keepdim=True)*axis
            valid = ((g['distance'] <= env.cfg.pose_distance_tolerance)
                     & (angle <= env.cfg.pose_normal_tolerance)
                     & (g['speed'] <= env.cfg.pose_linear_speed_tolerance)
                     & (perpendicular_omega.norm(dim=-1) <= env.cfg.pose_angular_speed_tolerance))
            self.count.copy_(torch.where(valid, self.count+1, 0))
            settled = self.count >= math.ceil(env.cfg.pose_hold_s/env.step_dt-1e-8)
            self.achieved |= settled
            self.step = env.common_step_counter
        return self


def progress(env):
    if not hasattr(env, '_planned_pose_progress'):
        env._planned_pose_progress = _PoseProgress(env)
    return env._planned_pose_progress.update(env)


def normal_angle(env, geometry=None):
    g = mdp.state(env).geometry() if geometry is None else geometry
    z = g['delta'].new_tensor((0., 0., 1.)).expand_as(g['delta'])
    dot = (quat_apply(g['q0'], z)*quat_apply(g['q1'], z)).sum(-1)
    return dot.clamp(-1., 1.).acos()


def reset_guided(env, env_ids):
    mdp.reset_docking(env, env_ids)
    if hasattr(env, '_planned_pose_progress'):
        env._planned_pose_progress.reset(env_ids)


def pose_success(env):
    return progress(env).achieved


def pose_hold_time(env):
    return progress(env).count.float()*env.step_dt


def plan_guided_env_cfg(play=False):
    base = head_tail_env_cfg(play)
    cfg = PlanGuidedEnvCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    cfg.events['reset_docking'] = EventTermCfg(func=reset_guided, mode='reset')
    cfg.terminations.pop('success', None)
    if not play:
        cfg.terminations['pose_success'] = TerminationTermCfg(func=pose_success)
    cfg.rewards = {}  # Supervised imitation does not optimize environment rewards.
    cfg.metrics.update(pose_success_fraction=MetricsTermCfg(func=pose_success),
                       pose_normal_angle_rad=MetricsTermCfg(func=normal_angle),
                       pose_hold_s=MetricsTermCfg(func=pose_hold_time))
    return cfg
