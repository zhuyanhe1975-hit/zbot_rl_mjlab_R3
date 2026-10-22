"""Sensor-limited student frames, independently assembled from the teacher.

The 56-slot layout permits the existing task encoder and routed actor to be
reused. Omitted privileged fields are zero. Gravity and gyro model one base
IMU in every morphology, including snake locomotion: no foot/site orientation
or velocity is supplied to the student. The simulator's base attitude models
an IMU attitude estimator, rather than an external position/heading sensor.

The inherited command FSM still uses privileged completion/recovery checks;
this experiment distils the policy, not a deployable state estimator or FSM.
"""

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from . import mdp
from .guided_observations import GuidedTeacherEnvCfg
from .observation_layout import PRIVILEGED_OMNI_OBS_DIM
from .task_fsm import TaskState


@dataclass
class GuidedStudentEnvCfg(GuidedTeacherEnvCfg):
    student_history_length: int = 5
    student_observation_noise: bool = True


def student_sensor_state(env):
    """One frame of encoders, base IMU and software commands, before history.

    Slots: biased encoder q 0:6, qdot 6:12, target 12:18, previous action 18:24,
    base gravity 28:31, base gyro 31:34, morphology/task 38:40,
    effective command 40:43, gait 43:46 and pending command 53:56.
    Noise standard deviations are .005 rad, .05 rad/s, .01 gravity units
    and .02 rad/s respectively. Software fields are never corrupted.
    """
    robot = env.scene['robot']
    data = robot.data
    base = robot.find_bodies('base')[0][0]
    command = env.command_manager.get_term('transform')
    action = env.action_manager.get_term('joint_pos')
    encoder = data.joint_pos_biased
    state = encoder.new_zeros((encoder.shape[0], PRIVILEGED_OMNI_OBS_DIM))
    state[:, :6] = encoder
    state[:, 6:12] = data.joint_vel
    state[:, 12:18] = action.commanded_target
    state[:, 18:24] = action.raw_action
    quaternion = data.body_link_quat_w[:, base]
    angular = data.body_link_ang_vel_w[:, base]
    gravity = torch.zeros_like(angular)
    gravity[:, 2] = -1.
    state[:, 28:31] = quat_apply_inverse(quaternion, gravity)
    state[:, 31:34] = quat_apply_inverse(quaternion, angular)
    if env.cfg.student_observation_noise:
        for start, stop, std in ((0, 6, .005), (6, 12, .05), (28, 31, .01), (31, 34, .02)):
            state[:, start:stop] += torch.randn_like(state[:, start:stop]) * std
    state[:, 38] = command.current_morphology
    recovering = command.straightening_mask & (command.current_task == TaskState.ASCENDING)
    state[:, 39] = torch.where(recovering, 3.25, command.current_task.to(state.dtype))
    state[:, 40:43] = command.velocity_command
    state[:, 43:46] = mdp.gait_observation(env)
    state[:, 53:56] = command.requested_velocity
    if env.cfg.play_transition_sync:
        transition = command.transition_mask[:, None]
        state[:, 40:43] = torch.where(transition, 0., state[:, 40:43])
        state[:, 53:56] = torch.where(transition, 0., state[:, 53:56])
    return torch.nan_to_num(state, nan=0., posinf=0., neginf=0.)
