"""Penalize premature end-body pressure, not ground support or valid mating."""
import math

import torch

from . import mdp


def end_contact_force(env):
    data = env.scene['end_to_end_contact'].data
    # maxforce sensor uses the contact frame: component 0 is normal pressure.
    history = data.force_history
    force = history if history is not None else data.force
    return force[..., 0].abs().reshape(env.num_envs, -1).amax(-1)


def premature_contact_cost(env, free_force=2., force_scale=10.):
    if not math.isfinite(free_force) or not math.isfinite(force_scale) or free_force < 0 or force_scale <= 0:
        raise ValueError('free_force must be nonnegative and force_scale positive')
    dock = mdp.state(env)
    g = dock.geometry()
    pose_ready = ((g['distance'] <= dock.cfg.capture_distance) & (g['angle'] <= dock.cfg.capture_angle)
                  & (g['speed'] <= dock.cfg.capture_speed) & (g['angular_speed'] <= dock.cfg.capture_angular_speed))
    matched = (dock.enabled() & torch.nn.functional.one_hot(g['quarter'], 4).bool()).any(-1)
    pressure = (end_contact_force(env) - free_force).clamp_min(0.)
    return torch.log1p(pressure / force_scale) * (~pose_ready & ~matched)
