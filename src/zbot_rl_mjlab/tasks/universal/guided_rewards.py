"""Archive regularizers by skill, without changing existing teacher recipes."""

import torch

from ..walking import mdp as walking
from . import mdp
from .teacher_rewards import teacher_action_rate


def guided_joint_energy(env, energy_scale=10.):
    term = mdp.command_term(env)
    # Only the walking archive included this energy objective. Learning the
    # archived crawl/transform must not first overcome an additional power tax.
    walking_task = term.standing_mode & ~term.transition_mask
    return torch.where(walking_task, walking.joint_energy_penalty(env, energy_scale=energy_scale), 0.)


def guided_action_rate(env):
    term = mdp.command_term(env)
    return torch.where(term.transition_mask, 0., teacher_action_rate(env))


def guided_standstill(env):
    # Tiny nonzero movement commands must not receive a competing stop reward.
    stop = mdp.command_term(env).requested_velocity.norm(dim=-1) <= 1e-6
    return torch.where(stop, mdp.standstill_reward(env), 0.)
