"""Six-module snake with 270-degree fixed inter-module clocking."""

import math

import mujoco
import numpy as np

from ...robot import get_spec

# Six actuated modules have five fixed connections. The base is module a4.
CONNECTION_BODIES = ('a2', 'a3', 'base', 'a5', 'a6')


def rz270_spec():
    """Change Ry(-45) Rz(180) to Ry(-45) Rz(270), keeping motor zeros."""
    spec = get_spec()
    extra_rz = np.array((math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4)))
    for body in spec.bodies:
        if body.name in CONNECTION_BODIES:
            rotated = np.empty(4)
            mujoco.mju_mulQuat(rotated, body.quat, extra_rz)
            body.quat = rotated
    return spec


def reset_rz270_snake(env, env_ids):
    """Start flat on the ground, with the foot_0 head facing world +X."""
    import torch

    from ..universal import mdp

    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not len(ids):
        return
    robot = env.scene['robot']
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = env.scene.env_origins[ids] + root.new_tensor((0., 0., .0555))
    # Head forward is site -Z. Ry(-90) maps it to world +X.
    root[:, 3:7] = root.new_tensor((2**-.5, 0., -(2**-.5), 0.))
    root[:, 7:] = 0.
    joints = torch.zeros((len(ids), 6), device=env.device, dtype=root.dtype)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)
    mdp.command_term(env).heading_initialized[ids] = False
