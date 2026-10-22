"""Student-only reward corrections; frozen teachers retain their own recipes."""

import math

import torch

from ..walking import mdp as walking
from . import mdp
from .teacher_rewards import teacher_velocity_tracking


def student_touchdown_flat(env, sigma=.05, sensor_name='feet_ground_contact'):
    term = mdp.command_term(env)
    cost = walking.touchdown_flat_penalty(env, sigma=sigma, sensor_name=sensor_name)
    return torch.where(term.standing_mode & ~term.transition_mask, cost, 0.)


def student_snake_motion_score(actual, command, std=(.05, .05, .15)):
    """Track all three signed axes, including unrequested drift and overspeed.

    Equal per-axis contributions prevent yaw from dominating translation. The
    joint kernel reserves half the score for satisfying all axes together.
    Maximum credit is four, matching the walking tracking reward's maximum.
    """
    if len(std) != 3 or any(not math.isfinite(value) or value <= 0 for value in std):
        raise ValueError('snake tracking std must contain three positive finite values')
    finite = torch.isfinite(actual).all(-1) & torch.isfinite(command).all(-1)
    error = (torch.nan_to_num(actual) - torch.nan_to_num(command)) / actual.new_tensor(std)
    per_axis = torch.exp(-error.square())
    score = 2 * (per_axis.mean(-1) + per_axis.prod(-1))
    return torch.where(finite, score, 0.)


def student_velocity_tracking(env, std=(.1, .1, .25), axis_weights=(2., 1., 1.),
                             snake_std=(.05, .05, .15)):
    term = mdp.command_term(env)
    baseline = teacher_velocity_tracking(env, std=std, axis_weights=axis_weights)
    snake = student_snake_motion_score(mdp.tracking_velocity(env), term.velocity_command, std=snake_std)
    return torch.where(~term.standing_mode & ~term.transition_mask, snake, baseline)
