"""Teacher-only reuse of successful archives; motion objectives remain site-relative.

Transformation weights come from bidirectional iter599 final_effective_weights;
energy and foot quality units come from deploy_walking iter1499. Phase gates
are joint-task adaptations, not claims that this combined task was validated.
"""
from dataclasses import dataclass
from typing import cast

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import TerrainHeightSensorCfg

from ..transformation import commanded
from ..transformation import snake_to_biped as standing
from ..walking import mdp as walking
from ..walking.env_cfg import env_cfg as walking_env_cfg
from . import mdp
from .env_cfg import UniversalEnvCfg
from .teacher_snake_rewards import (
    exploration_scale,
    snake_joint_deviation,
    snake_motion_score,
    snake_target_saturation,
    snake_vertical_speed,
)
from .teacher_transition import reset_transition_failure_memory, transition_failure


@dataclass
class ArchiveOmniTeacherEnvCfg(UniversalEnvCfg):
    joint_standing_error_scale: float = .5
    standing_pose_gravity_scale: float = 1.
    sole_alignment_height_std: float = .03
    sole_alignment_angle_std_deg: float = 10.
    ascent_reward_stop_height: float = .23
    require_sole_support_for_positive_rewards: bool = False
    curriculum_start_stage: int = 2
    support_angle_limit_deg: float = 25.
    support_height_limit: float = .01


def _transition(env, value):
    return torch.where(mdp.command_term(env).transition_mask, value, 0.)


def transition_posture(env):
    return _transition(env, mdp.posture_reward(env))


def transition_progress(env):
    return _transition(env, mdp.transition_progress_reward(env))


def transition_failure_cost(env):
    return _transition(env, standing.reset_failure_cost(env))


def transition_pose(env):
    return _transition(env, commanded.pose_reward(env))


def transition_height(env):
    return _transition(env, commanded.height_reward(env))


def transition_sole(env, angle_power=1.):
    return _transition(env, commanded.sole_reward(env, angle_power=angle_power))


def transition_finish_speed(env):
    return _transition(env, commanded.stand_finish_joint_speed_cost(env))


def transition_descent_impact(env, max_body_weights=2.):
    return _transition(env, commanded.descent_ground_impact_cost(env, max_body_weights=max_body_weights))


@dataclass
class _AscentMemory:
    peak_height: torch.Tensor
    start_step: torch.Tensor
    standing_mode: torch.Tensor


def _ascent_memory(env):
    if not hasattr(env, '_teacher_ascent_memory'):
        term = mdp.command_term(env)
        env._teacher_ascent_memory = _AscentMemory(
            torch.full((env.num_envs,), -torch.inf, device=env.device),
            torch.full_like(term.start_step, -1),
            term.standing_mode.clone(),
        )
    return env._teacher_ascent_memory


def reset_ascent_memory(env, env_ids):
    """Reset only teacher memory; never reset physics or legacy task state."""
    memory = _ascent_memory(env)
    ids = slice(None) if env_ids is None else env_ids
    with torch.inference_mode():
        memory.peak_height[ids] = -torch.inf
        memory.start_step[ids] = -1


def transition_ascent(env, target_speed=.15, pose_error_range=(.2, .8)):
    """Archived commanded ascent formula with a teacher-owned peak-height latch.

    Universal commands do not run the archived FallRetryState reset events.
    Separate memory therefore resets on both episode reset and command change,
    preserving the archive's one-ascent latch without stale cross-command state.
    """
    threshold = env.cfg.ascent_reward_stop_height
    if not 0 < threshold <= env.cfg.standing_base_height:
        raise ValueError('ascent_reward_stop_height must be positive and no higher than the standing goal')
    term = mdp.command_term(env)
    memory = _ascent_memory(env)
    changed = (term.start_step != memory.start_step) | (term.standing_mode != memory.standing_mode)
    fresh = changed | ~term.transition_mask
    height = standing.base_height(env)
    with torch.inference_mode():
        memory.peak_height.copy_(torch.where(fresh, height, torch.maximum(memory.peak_height, height)))
        memory.start_step.copy_(term.start_step)
        memory.standing_mode.copy_(term.standing_mode)
    value = standing.base_ascent_velocity_reward(env, target_speed=target_speed, pose_error_range=None)
    value = torch.where(memory.peak_height >= threshold, 0., value)
    if pose_error_range is not None:
        zero_error, full_error = pose_error_range
        if not (0 <= zero_error < full_error < float('inf')):
            raise ValueError('pose_error_range must satisfy 0 <= zero < full, both finite')
        error = (env.scene['robot'].data.joint_pos - commanded.goal_joints(env)).abs().amax(-1)
        fraction = ((error - zero_error) / (full_error - zero_error)).clamp(0, 1)
        pose_ready = 1 - fraction.square() * (3 - 2 * fraction)
        span = env.cfg.standing_base_height - standing.SNAKE_ROOT_POS[2]
        if span <= 0:
            raise ValueError('standing_base_height must exceed snake height')
        lift_ready = ((height - standing.SNAKE_ROOT_POS[2]) / span).clamp(0, 1)
        upright_ready = (1 - .5 * standing.base_gravity_alignment_cost(env)).clamp(0, 1)
        readiness = torch.nan_to_num(pose_ready * lift_ready * upright_ready, nan=0.)
        value = value * (1 - readiness)
    return _transition(env, torch.where(term.standing_mode, value, 0.))


def _biped_motion(env, value):
    term = mdp.command_term(env)
    return torch.where(term.standing_mode & ~term.transition_mask, value, 0.)


def biped_support_foot_slip(env, sensor_name='feet_ground_contact', site_names=('foot_0_sole', 'foot_1_sole')):
    return _biped_motion(env, walking.support_foot_slip(env, sensor_name=sensor_name, site_names=site_names))


def biped_foot_nonparallel(env, sensor_name='feet_ground_contact', std=.1):
    return _biped_motion(env, walking.foot_nonparallel_ground_penalty(env, sensor_name=sensor_name, std=std))


def teacher_velocity_tracking(env, std=(.1, .1, .25), axis_weights=(2., 1., 1.)):
    """Archived biped reward widths in the requested site frame; snake keeps signed progress."""
    term = mdp.command_term(env)
    actual = mdp.tracking_velocity(env)
    widths = actual.new_tensor(std)
    weights = actual.new_tensor(axis_weights)
    biped = (torch.exp(-((actual - term.velocity_command) / widths).square()) * weights).sum(-1)
    snake = snake_motion_score(actual, term.velocity_command, term.cfg.velocity_limits)
    return torch.where(term.transition_mask, 0., torch.where(term.standing_mode, biped, snake))


def biped_upright(env):
    return _biped_motion(env, walking.upright(env))


def biped_joint_limits(env):
    return _biped_motion(env, walking.joint_pos_limits(env))


def biped_joint_acceleration(env):
    return _biped_motion(env, walking.joint_acc_l2(env))


def biped_soft_landing(env, sensor_name='feet_ground_contact'):
    return _biped_motion(env, walking.soft_landing(env, sensor_name=sensor_name))


def biped_swing_clearance(env, height_sensor_name='foot_height_scan', target=.05, std=.04):
    """Archived clearance kernel, sharing the actor's clock and modern sole-site heights.

    Stop commands disable stepping assistance; snake and transformation never receive it.
    """
    phase_sin = mdp.gait_observation(env)[:, 0]
    swing = torch.stack((phase_sin < 0, phase_sin > 0), -1).float()
    heights = env.scene[height_sensor_name].data.heights
    value = (swing * torch.exp(-((heights - target) / std).square())).mean(-1)
    moving = mdp.command_term(env).requested_velocity.norm(dim=-1) > .01
    return _biped_motion(env, torch.where(moving, value, 0.))


def teacher_action_rate(env):
    term = mdp.command_term(env)
    biped = term.standing_mode & ~term.transition_mask
    return walking.action_rate_l2(env) * torch.where(biped, 5., 1.)


def teacher_joint_energy(env, energy_scale=10.):
    return walking.joint_energy_penalty(env, energy_scale=energy_scale) * exploration_scale(env)


def biped_fallen(env, height=.22, tilt=.75):
    """End failed walking attempts promptly without rejecting deliberate low-mode transitions."""
    term = mdp.command_term(env)
    failed = term.standing_mode & term.standing_achieved & walking.fallen(env, height=height, tilt=tilt)
    if not term.cfg.training:
        failed &= ~term.manual_control
    return failed


def configure_teacher_rewards(cfg):
    """Install archival units once, retaining the current joint-task objectives."""
    for name in ('energy', 'contact_impact', 'failure'):
        cfg.rewards.pop(name, None)
    cfg.rewards.update({
        'mode_posture': RewardTermCfg(func=transition_posture, weight=2.),
        'transition_progress': RewardTermCfg(func=transition_progress, weight=2.),
        'velocity_tracking': RewardTermCfg(func=teacher_velocity_tracking, weight=1., params={'std': (.1, .1, .25), 'axis_weights': (2., 1., 1.)}),
        'biped_upright': RewardTermCfg(func=biped_upright, weight=.2),
        'joint_limits': RewardTermCfg(func=biped_joint_limits, weight=-.1),
        'joint_acceleration': RewardTermCfg(func=biped_joint_acceleration, weight=-.0001),
        'soft_landing': RewardTermCfg(func=biped_soft_landing, weight=-.02, params={'sensor_name': 'feet_ground_contact'}),
        'swing_clearance': RewardTermCfg(func=biped_swing_clearance, weight=.5, params={'height_sensor_name': 'foot_height_scan', 'target': .05, 'std': .04}),
        'action_rate': RewardTermCfg(func=teacher_action_rate, weight=-.01),
        'joint_standing': RewardTermCfg(func=transition_pose, weight=12.),
        'base_height_progress': RewardTermCfg(func=transition_height, weight=1.),
        'base_ascent_velocity': RewardTermCfg(func=transition_ascent, weight=8., params={'target_speed': .15, 'pose_error_range': (.2, .8)}),
        'sole_ground_alignment': RewardTermCfg(func=transition_sole, weight=1., params={'angle_power': 1.}),
        'reset_failure': RewardTermCfg(func=transition_failure_cost, weight=-10.),
        'stand_finish_joint_speed': RewardTermCfg(func=transition_finish_speed, weight=-.02),
        'descent_ground_impact': RewardTermCfg(func=transition_descent_impact, weight=-2., params={'max_body_weights': 2.}),
        'joint_energy': RewardTermCfg(func=teacher_joint_energy, weight=-.5, params={'energy_scale': 10.}),
        'snake_vertical_speed': RewardTermCfg(func=snake_vertical_speed, weight=-.2),
        'snake_joint_deviation': RewardTermCfg(func=snake_joint_deviation, weight=-.02),
        'snake_target_saturation': RewardTermCfg(func=snake_target_saturation, weight=-.1),
        'support_foot_slip': RewardTermCfg(func=biped_support_foot_slip, weight=-5., params={'sensor_name': 'feet_ground_contact', 'site_names': ('foot_0_sole', 'foot_1_sole')}),
        'foot_nonparallel_ground': RewardTermCfg(func=biped_foot_nonparallel, weight=-5., params={'sensor_name': 'feet_ground_contact', 'std': .1}),
    })
    cfg.events['reset_archive_ascent_memory'] = EventTermCfg(func=reset_ascent_memory, mode='reset')
    cfg.terminations['biped_fallen'] = TerminationTermCfg(func=biped_fallen, params={'height': .22, 'tilt': .75})
    cfg.terminations['transition_failure'] = TerminationTermCfg(func=transition_failure)
    cfg.events['reset_transition_failure_memory'] = EventTermCfg(func=reset_transition_failure_memory, mode='reset')
    for sensor in cfg.scene.sensors:
        if sensor.name == 'feet_ground_contact':
            sensor.track_air_time = True
            sensor.history_length = cfg.decimation
    if not any(sensor.name == 'foot_height_scan' for sensor in cfg.scene.sensors):
        height = cast(TerrainHeightSensorCfg, next(sensor for sensor in walking_env_cfg().scene.sensors if sensor.name == 'foot_height_scan'))
        height.debug_vis = False
        cfg.scene.sensors = (*cfg.scene.sensors, height)
    return cfg
