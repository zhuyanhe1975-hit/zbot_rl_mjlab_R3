"""PPO-only transformation quality costs; defaults are simulation tuning targets.

Policy costs exclude forced straightening, where the executor overrides actions.
Impact protection includes that substage. Teachers and DAgger do not use this module.
"""

import math
from typing import cast

import torch

from ..transformation import commanded
from ..transformation import mdp as transform
from ..transformation import snake_to_biped as standing
from . import mdp, teacher_rewards
from .guided_transitions import GuidedOmniCommand
from .task_fsm import TaskState


def _positive(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')


def _active(env, *, policy=True):
    term = cast(GuidedOmniCommand, mdp.command_term(env))
    mask = term.transition_mask & (
        (term.current_task == TaskState.ASCENDING) | (term.current_task == TaskState.DESCENDING)
    )
    if policy:
        mask = mask & ~cast(torch.Tensor, term.straightening_mask)
    return mask


def _bounded_square(value):
    # Clamp before squaring to avoid overflow for large finite observations.
    return torch.nan_to_num(value, nan=2., posinf=2., neginf=-2.).clamp(-2., 2.).square()


def student_motion_progress_score(actual, command, std):
    """Reward requested-axis progress without credit for standing still.

    Equal mean/worst-axis credit prevents a successful axis from hiding a
    missed mixed-command axis. Unrequested drift has a bounded Gaussian cost;
    stop commands instead reward simultaneous stillness on all three axes.
    """
    widths = actual.new_tensor(std)
    if widths.shape != (3,) or not torch.isfinite(widths).all() or not (widths > 0).all():
        raise ValueError('std must contain three finite positive values')
    if actual.shape != command.shape or actual.ndim != 2 or actual.shape[-1] != 3:
        raise ValueError('actual and command must have matching [N, 3] shapes')
    finite = torch.isfinite(actual).all(-1) & torch.isfinite(command).all(-1)
    actual = torch.nan_to_num(actual, nan=0., posinf=0., neginf=0.)
    command = torch.nan_to_num(command, nan=0., posinf=0., neginf=0.)
    active = command != 0
    magnitude = torch.where(active, command.abs(), 1.)
    progress = (1 - (actual - command).abs() / magnitude).clamp(-1, 1)
    mean = torch.where(active, progress, 0.).sum(-1) / active.sum(-1).clamp_min(1)
    worst = torch.where(active, progress, 1.).amin(-1)
    drift = torch.where(active, 0., actual / widths).square().sum(-1)
    moving = (.5 * (mean + worst) - (1 - torch.exp(-drift))).clamp(-1, 1)
    stopped = torch.exp(-(actual / widths).square().sum(-1))
    score = 4 * torch.where(active.any(-1), moving, stopped)
    return torch.where(finite, score, -4.)


def student_velocity_progress(env, std=(.1, .1, .25), snake_std=(.05, .05, .15)):
    """Opt-in PPO motion score; transformation rewards remain separate."""
    term = cast(GuidedOmniCommand, mdp.command_term(env))
    actual = mdp.tracking_velocity(env)
    biped = student_motion_progress_score(actual, term.velocity_command, std)
    snake = student_motion_progress_score(actual, term.velocity_command, snake_std)
    motion = ((term.current_task == TaskState.BIPED_MOTION)
              | (term.current_task == TaskState.SNAKE_MOTION)) & ~term.transition_mask
    return torch.where(motion, torch.where(term.standing_mode, biped, snake), 0.)


def _settling(env, near_ground_height=.12, near_goal_error=.35):
    _positive(near_ground_height=near_ground_height, near_goal_error=near_goal_error)
    term = mdp.command_term(env)
    q = env.scene['robot'].data.joint_pos
    height = standing.base_height(env)
    goal = torch.where(term.standing_mode[:, None], q.new_tensor(env.cfg.standing_joint_pos), 0.)
    close = (q - goal).abs().amax(-1) <= near_goal_error
    at_height = torch.where(term.standing_mode, height >= env.cfg.ascent_reward_stop_height,
                            height <= near_ground_height)
    return _active(env) & at_height & close, goal


def transition_joint_overspeed(env, ascent_limit=3., descent_limit=2.5):
    """Mean squared actual speed excess in units of the phase's speed target."""
    _positive(ascent_limit=ascent_limit, descent_limit=descent_limit)
    velocity = env.scene['robot'].data.joint_vel
    limit = torch.where(mdp.command_term(env).standing_mode, ascent_limit, descent_limit)
    cost = _bounded_square((velocity.abs() / limit[:, None] - 1).clamp_min(0)).mean(-1)
    return torch.where(_active(env), cost, 0.)


def transition_settle_target(env, angle_scale=.15, near_ground_height=.12, near_goal_error=.35):
    _positive(angle_scale=angle_scale)
    mask, goal = _settling(env, near_ground_height, near_goal_error)
    target = env.action_manager.get_term('joint_pos').commanded_target
    cost = _bounded_square((target - goal) / angle_scale).mean(-1)
    return torch.where(mask, cost, 0.)


def transition_settle_speed(env, speed_scale=.5, near_ground_height=.12, near_goal_error=.35):
    _positive(speed_scale=speed_scale)
    mask, _ = _settling(env, near_ground_height, near_goal_error)
    cost = _bounded_square(env.scene['robot'].data.joint_vel / speed_scale).mean(-1)
    return torch.where(mask, cost, 0.)


def transition_settle_action(env, near_ground_height=.12, near_goal_error=.35):
    """Penalize a persistent rate command even when its step-to-step delta is zero."""
    mask, _ = _settling(env, near_ground_height, near_goal_error)
    cost = _bounded_square(env.action_manager.get_term('joint_pos').raw_action).mean(-1)
    return torch.where(mask, cost, 0.)


def student_transition_ascent(env, target_speed=.10, pose_error_range=(.2, .8)):
    """Keep the archived peak latch and readiness; reduce credit above target speed.

    The bounded excess cost scales the positive archived credit, so stopped or
    pose-ready phases do not gain a new constant reward or an unrelated penalty.
    """
    _positive(target_speed=target_speed)
    value = teacher_rewards.transition_ascent(env, target_speed, pose_error_range)
    excess = (standing.base_vertical_velocity(env) / target_speed - 1).clamp_min(0)
    value = value - value.clamp_min(0) * _bounded_square(excess)
    return torch.where(_active(env), value, 0.)


def transition_descent_speed(env, max_down_speed=.10, approach_height=.15):
    _positive(max_down_speed=max_down_speed, approach_height=approach_height)
    term = mdp.command_term(env)
    mask = _active(env) & ~term.standing_mode & (standing.base_height(env) <= approach_height)
    excess = (-standing.base_vertical_velocity(env) / max_down_speed - 1).clamp_min(0)
    return torch.where(mask, _bounded_square(excess), 0.)


def transition_ground_impact(env, max_body_weights=2.):
    """Squared excess body-weight peak, in either transformation direction."""
    _positive(max_body_weights=max_body_weights)
    if max_body_weights <= 1:
        raise ValueError('max_body_weights must be greater than one')
    gravity = math.sqrt(sum(component ** 2 for component in env.cfg.sim.mujoco.gravity))
    _positive(gravity=gravity)
    weight = transform.body_masses(env).sum(-1) * gravity
    cost = _bounded_square((commanded.ground_force_peak(env) / weight - max_body_weights).clamp_min(0))
    return torch.where(_active(env, policy=False), cost, 0.)
