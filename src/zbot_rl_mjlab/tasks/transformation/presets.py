"""变形任务的可复用动作、课程、扰动和观测选配。"""

import math
from dataclasses import replace

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ...robot import JOINT_NAMES
from ..common import add_base_disturbance
from .actions import JointTargetRateActionCfg, mirrored_last_action
from .poses import ARC_JOINT_POS, ARC_ROOT_POS, ARC_ROOT_QUAT
from .robust import (
    RampedBaseImpulse,
    abandoned_exploration,
    early_base_ground_contact,
    fallen_after_standing,
    initial_ascent_velocity_reward,
    reset_fall_retry,
    standing_success_metric,
    update_fixed_standing_state,
)
from .snake_to_biped import (
    base_height,
    base_height_progress_reward,
    base_vertical_velocity,
    joint_standing_error_metric,
    joint_standing_reward,
    reset_failure_cost,
    sole_ground_alignment_reward,
)
from .teacher_observations import joint_speed_peak, teacher_observation_groups


def configure_beta(cfg, play=False):
    cfg.curriculum_prepare_pose = True
    cfg.first_stage_joint_weight = 0.6
    cfg.first_stage_prepared_joint_weight = 3.0
    cfg.standing_stage_joint_weight = 6.0
    cfg.curriculum_joint_error_threshold = 1.0
    if not getattr(cfg, "compact_standing_rewards", False):
        cfg.rewards["joint_symmetry"].weight = -0.2
        cfg.rewards["joint_standing"].weight = 6.0 if play else 0.6
    return cfg


def configure_arc(cfg, play=False):
    cfg.initial_joint_pos = ARC_JOINT_POS
    robot = cfg.scene.entities["robot"]
    robot.init_state = replace(
        robot.init_state,
        pos=ARC_ROOT_POS,
        rot=ARC_ROOT_QUAT,
        joint_pos=dict(zip(JOINT_NAMES, ARC_JOINT_POS, strict=True)),
    )
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointTargetRateActionCfg):
        raise TypeError("arc transformation requires rate-controlled joint targets")
    bounds = action.joint_target_bounds
    for name, target in zip(JOINT_NAMES, cfg.standing_joint_pos, strict=True):
        low, high = bounds.get(name, (-math.pi, math.pi))
        bounds[name] = (max(0.0, low), min(math.pi, high)) if target > 0 else (max(-math.pi, low), min(0.0, high))
    return cfg


def configure_mirrored_standing(cfg, play=False):
    """共享镜像动作、滚转重置和紧凑站立行为，供教师与双向任务选配。"""
    # 确认跌倒后，从随机滚转的蛇形重置姿态重新尝试。
    cfg.terminations = {
        name: cfg.terminations[name] for name in ("update_state", "time_out")
    }
    cfg.terminations["update_state"] = TerminationTermCfg(func=update_fixed_standing_state)
    cfg.terminations["fallen_after_standing"] = TerminationTermCfg(func=fallen_after_standing)
    cfg.terminations["early_base_ground_contact"] = TerminationTermCfg(func=early_base_ground_contact)
    cfg.terminations["abandoned_exploration"] = TerminationTermCfg(func=abandoned_exploration)
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern="base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce",
    ))
    cfg.actions["joint_pos"].max_target_speed = 2 * math.pi
    for group in cfg.observations.values():
        group.terms["last_action"].func = mirrored_last_action
    cfg.events["reset_transition"].func = reset_fall_retry
    cfg.curriculum_start_stage = 2
    cfg.curriculum_prepare_pose = False
    cfg.require_sole_support_for_positive_rewards = False
    cfg.curriculum = {}
    cfg.standing_stage_ramp_steps = 0
    cfg.standing_stage_height_weight = 1.0
    cfg.standing_stage_velocity_weight = 8.0
    cfg.feet_contact_weight = 0.0
    cfg.base_support_distance_weight = 0.0
    cfg.first_stage_joint_weight = 3.0
    cfg.standing_stage_joint_weight = 12.0
    cfg.rewards = {
        "joint_standing": RewardTermCfg(func=joint_standing_reward, weight=12.0),
        "base_height_progress": RewardTermCfg(func=base_height_progress_reward, weight=1.0),
        "base_ascent_velocity": RewardTermCfg(func=initial_ascent_velocity_reward, weight=8.0,
            params={"target_speed": 0.15, "pose_error_range": (0.2, 0.8)}),
        "sole_ground_alignment": RewardTermCfg(func=sole_ground_alignment_reward, weight=1.0,
            params={"angle_power": 1.0}),
        "reset_failure": RewardTermCfg(func=reset_failure_cost, weight=-10.0),
    }
    cfg.sole_alignment_angle_std_deg = 10.0
    add_base_disturbance(
        cfg,
        force_range=(0.0, 0.0),
        torque_range=(0.0, 0.0),
        duration_s=(0.08, 0.20),
        cooldown_s=(0.8, 2.0),
        disturbed_fraction=0.0,
    )
    cfg.events["base_disturbance"].func = RampedBaseImpulse
    cfg.metrics = {
        "joint_error": MetricsTermCfg(func=joint_standing_error_metric),
        "base_height": MetricsTermCfg(func=base_height),
        "base_vertical_velocity": MetricsTermCfg(func=base_vertical_velocity),
        "success": MetricsTermCfg(func=standing_success_metric, reduce="last"),
        "joint_speed_peak": MetricsTermCfg(func=joint_speed_peak, reduce="max"),
    }
    cfg.scene.sensors = tuple(replace(sensor, fields=("found",)) if isinstance(sensor, ContactSensorCfg) else sensor for sensor in cfg.scene.sensors)
    if play:
        cfg.seed = -1  # 每次启动随机化；仍可通过 --env.seed 复现训练。
        cfg.disturbance_initial_scale = 1.0
    return cfg


def configure_teacher(cfg, play=False):
    cfg.observations = teacher_observation_groups()
    cfg.actions["joint_pos"].max_target_speed = 2 * math.pi
    return cfg
