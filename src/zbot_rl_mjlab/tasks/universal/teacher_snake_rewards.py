"""Archived snake exploration mechanics adapted to site-relative commands.

The archive's 15*v_forward, -5*|v_lateral| and .25+5*v_forward
schedule retain their units. Command direction replaces fixed world X; requested
speed caps forward credit, so omnidirectional control cannot reward overspeed.
Applying the exploration schedule to energy is a joint-task adaptation, rather
than an energy term present in the snake archive. No fixed heading is imposed.
"""

import torch

from ..transformation import mdp as transform
from . import mdp


def _translation(actual, command):
    speed = command[:, :2].norm(dim=-1)
    direction = command[:, :2] / speed[:, None].clamp_min(1e-6)
    advance = (actual[:, :2] * direction).sum(-1)
    perpendicular = actual[:, 1] * direction[:, 0] - actual[:, 0] * direction[:, 1]
    active = speed > 1e-6
    scale = (.25 + 5 * advance.clamp_min(0.)).clamp(max=1.)
    return speed, advance, perpendicular, active, scale


def snake_motion_score(actual, command, limits=(.25, .20, .8)):
    """Signed COM progress with archive units and an independent yaw objective."""
    actual = torch.nan_to_num(actual, nan=0., posinf=0., neginf=0.)
    command = torch.nan_to_num(command, nan=0., posinf=0., neginf=0.)
    speed, advance, perpendicular, active, scale = _translation(actual, command)
    translation = 15 * torch.minimum(advance, speed.clamp(max=.2))
    translation -= 5 * perpendicular.abs() * scale
    translation = torch.where(active, translation, 0.)
    yaw_actual = torch.cat((torch.zeros_like(actual[:, :2]), actual[:, 2:3]), -1)
    yaw_command = torch.cat((torch.zeros_like(command[:, :2]), command[:, 2:3]), -1)
    yaw = 4 * mdp.motion_tracking_score(yaw_actual, yaw_command, limits)
    return translation + torch.where(command[:, 2].abs() > 1e-6, yaw, 0.)


def snake_score(env):
    term = mdp.command_term(env)
    score = snake_motion_score(mdp.tracking_velocity(env), term.velocity_command, term.cfg.velocity_limits)
    return torch.where(~term.standing_mode & ~term.transition_mask, score, 0.)


def exploration_scale(env):
    """Reduce snake energy cost before useful motion, leaving other phases intact."""
    term = mdp.command_term(env)
    actual = torch.nan_to_num(mdp.tracking_velocity(env), nan=0., posinf=0., neginf=0.)
    command = torch.nan_to_num(term.velocity_command, nan=0., posinf=0., neginf=0.)
    _, _, _, active, scale = _translation(actual, command)
    stable_snake = ~term.standing_mode & ~term.transition_mask
    # Pure yaw has no commanded translation: retain the archive's exploratory floor.
    moving = command.norm(dim=-1) > 1e-6
    exploratory = torch.where(active, scale, .25)
    return torch.where(stable_snake & moving, exploratory, 1.)


def _snake_motion(env, value):
    term = mdp.command_term(env)
    return torch.where(~term.standing_mode & ~term.transition_mask, value, 0.)


def snake_vertical_speed(env):
    """Archived absolute COM vertical speed, in m/s, only during snake motion."""
    return _snake_motion(env, transform.com_state(env)[:, 1].abs())


def snake_joint_deviation(env):
    """Archived mean squared deviation from the zero-angle snake reference.

    The snake archive's default_joint_pos was all zeros. The shared universal
    scene instead has standing defaults, so using those would change the goal.
    """
    return _snake_motion(env, env.scene['robot'].data.joint_pos.square().mean(-1))


def snake_target_saturation(env):
    """Archived fraction within .001 rad of a snake target limit, dimensionless."""
    action = env.action_manager.get_term('joint_pos')
    target = action.commanded_target
    limit = env.cfg.actions['joint_pos'].snake_target_limit
    value = (target.abs() >= limit - 1e-3).float().mean(-1)
    return _snake_motion(env, value)
