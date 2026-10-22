"""Mode-specific physical observations for the current clean privileged teacher."""

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .guided_transitions import guided_privileged_state
from .omni import guided_reference_state, guided_snake_mask, motion_site_index
from .teacher_rewards import ArchiveOmniTeacherEnvCfg


@dataclass
class GuidedTeacherEnvCfg(ArchiveOmniTeacherEnvCfg):
    guided_local_reference: bool = True
    native_biped_actions: bool = False
    mirrored_transform_actions: bool = False
    all_gaussian_actions: bool = False
    command_visual_actual_velocity: bool = False
    play_transition_sync: bool = False


def local_reference_privileged_state(env):
    """Keep raw56 layout; use the selected body's local IMU and reference point.

    Planar velocities use the gravity-aligned locomotion axes. Gravity and gyro
    use the actual reference body's XYZ axes, as with a mounted IMU.
    """
    state = guided_privileged_state(env)
    if getattr(env.cfg, 'play_transition_sync', False):
        transition = env.command_manager.get_term('transform').transition_mask[:, None]
        # Transform training starts with zero motion commands. Preserve the
        # pending user request internally, without feeding motion cues to this head.
        state[:, 40:43] = torch.where(transition, 0., state[:, 40:43])
        state[:, 53:56] = torch.where(transition, 0., state[:, 53:56])
    if not getattr(env.cfg, 'guided_local_reference', False):
        return state
    quaternion, linear, angular = guided_reference_state(env)
    gravity = torch.zeros_like(angular)
    gravity[:, 2] = -1.
    state[:, 27] = linear[:, 2]
    state[:, 28:31] = quat_apply_inverse(quaternion, gravity)
    state[:, 31:34] = quat_apply_inverse(quaternion, angular)
    robot = env.scene['robot']
    base = robot.find_bodies('base')[0][0]
    base_height = robot.data.body_link_pos_w[:, base, 2]
    site_height = robot.data.site_pos_w[:, motion_site_index(env), 2]
    state[:, 34] = torch.where(guided_snake_mask(env), site_height, base_height) - env.scene.env_origins[:, 2]
    return torch.nan_to_num(state, nan=0., posinf=0., neginf=0.)
