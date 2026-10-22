"""两个变形方向的实验配方与训练设置。"""


import math
from dataclasses import replace
from typing import cast

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..rl import RobustSnakeToBipedRunner, SnakeToBipedHotRewardRunner
from ..robot import JOINT_NAMES
from ..tasks.common import add_base_disturbance
from ..tasks.transformation import mdp, robust, teacher_observations
from ..tasks.transformation import snake_to_biped as standing_task
from ..tasks.transformation.actions import (
    JointTargetRateActionCfg,
    MirroredJointTargetRateActionCfg,
    mirrored_last_action,
)
from ..tasks.transformation.env_cfg import (
    TASK_ID,
    BipedToSnakeEnvCfg,
    transformation_scene_cfg,
)
from ..tasks.transformation.poses import (
    ARC_JOINT_POS,
    ARC_ROOT_POS,
    ARC_ROOT_QUAT,
    SNAKE_ROOT_POS,
    SNAKE_ROOT_QUAT,
)
from ..tasks.transformation.reverse_observations import (
    reserved_hold_observation,
    support_observation,
)
from ..tasks.transformation.robust import ROBUST_TASK_ID, RobustSnakeToBipedEnvCfg
from ..tasks.transformation.snake_to_biped import (
    ARC_TASK_ID,
    BETA_TASK_ID,
    SnakeToBipedEnvCfg,
)
from ..tasks.transformation.snake_to_biped import (
    TASK_ID as REVERSE_TASK_ID,
)
from ..tasks.transformation.teacher_observations import TEACHER_TASK_ID
from .base import Experiment


def transformation_env_cfg(play=False, cfg_cls=BipedToSnakeEnvCfg):
    """装配下降方向，默认创建下降专用配置。"""
    cfg = transformation_scene_cfg(play=play, cfg_cls=cfg_cls)
    cfg.rewards = {
        "joint_zero": RewardTermCfg(func=mdp.joint_zero_reward, weight=3.0),
        "com_descent_velocity": RewardTermCfg(
            func=mdp.com_descent_velocity_reward,
            weight=1.0,
            params={"target_speed": 0.03},
        ),
        "joint_symmetry": RewardTermCfg(func=mdp.joint_symmetry_cost, weight=-0.5),
    }
    cfg.metrics = {
        "joint_error": MetricsTermCfg(func=mdp.joint_error_metric),
        "com_height": MetricsTermCfg(func=mdp.com_height),
        "success": MetricsTermCfg(func=mdp.success, reduce="last"),
    }
    return cfg


def snake_to_biped_env_cfg(play=False, cfg_cls=SnakeToBipedEnvCfg, action_cfg_cls=JointTargetRateActionCfg, compact=False):
    """直接创建最终起立配置类型，再装配方向相关的行为。"""
    cfg = cast(SnakeToBipedEnvCfg, transformation_scene_cfg(
        play=play, cfg_cls=cfg_cls, action_cfg_cls=action_cfg_cls,
    ))
    cfg.metrics = {
        "joint_error": MetricsTermCfg(func=standing_task.joint_standing_error_metric),
        "com_height": MetricsTermCfg(func=mdp.com_height),
        "success": MetricsTermCfg(func=standing_task.success, reduce="last"),
    }
    # 回放评估完整站立目标；训练从第一阶段开始。
    cfg.curriculum_start_stage = 2 if play else 1
    for group in cfg.observations.values():
        group.terms["hold_time"] = ObservationTermCfg(func=reserved_hold_observation)
        group.terms["support_state"] = ObservationTermCfg(func=support_observation)
    standing = cfg.curriculum_start_stage == 2
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
        ),
    )
    robot = cfg.scene.entities["robot"]
    robot.init_state = replace(
        robot.init_state,
        pos=SNAKE_ROOT_POS,
        rot=SNAKE_ROOT_QUAT,
        joint_pos=dict(zip(JOINT_NAMES, cfg.initial_joint_pos, strict=True)),
        joint_vel={".*": 0.0},
    )
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointTargetRateActionCfg):
        raise TypeError("reverse transformation requires rate-controlled joint targets")
    action.joint_target_bounds = {
        JOINT_NAMES[i]: (cfg.standing_joint_pos[i] - cfg.end_joint_allowed_error,
                         cfg.standing_joint_pos[i] + cfg.end_joint_allowed_error)
        for i in (0, 5)
    }
    if compact:
        return cfg
    cfg.rewards = {
        "end_joint_flip": RewardTermCfg(
            func=standing_task.end_joint_flip_cost,
            weight=cfg.end_joint_flip_weight,
        ),
        "axial_roll_rate": RewardTermCfg(
            func=standing_task.axial_roll_rate_cost,
            weight=0.0 if standing else cfg.first_stage_roll_rate_weight,
        ),
        "initial_joint_pose": RewardTermCfg(
            func=standing_task.initial_joint_pose_cost,
            weight=0.0 if standing else cfg.first_stage_initial_pose_weight,
        ),
        "joint_standing": RewardTermCfg(
            func=standing_task.joint_standing_reward,
            weight=cfg.standing_stage_joint_weight if standing else cfg.first_stage_joint_weight,
        ),
        "base_height_progress": RewardTermCfg(
            func=standing_task.base_height_progress_reward,
            weight=cfg.standing_stage_height_weight if standing else 0.0,
        ),
        "base_ascent_velocity": RewardTermCfg(
            func=standing_task.base_ascent_velocity_reward,
            weight=cfg.standing_stage_velocity_weight if standing else 0.0,
            params={"target_speed": 0.1},
        ),
        "base_support_distance": RewardTermCfg(
            func=standing_task.base_support_distance_cost,
            weight=cfg.base_support_distance_weight,
        ),
        "joint_symmetry": RewardTermCfg(
            func=mdp.joint_symmetry_cost,
            weight=-1.0
        ),
        "sole_ground_alignment": RewardTermCfg(
            func=standing_task.sole_ground_alignment_reward,
            weight=2.0,
        ),
        "both_feet_contact": RewardTermCfg(func=standing_task.both_feet_contact_metric, weight=cfg.feet_contact_weight),
        "sole_height": RewardTermCfg(func=standing_task.sole_height_cost, weight=-1.0),
        "base_z_roll": RewardTermCfg(func=standing_task.base_z_roll_cost, weight=-1.0),
        "base_gravity_alignment": RewardTermCfg(func=standing_task.base_gravity_alignment_cost, weight=cfg.base_gravity_alignment_weight),
        "reset_failure": RewardTermCfg(func=standing_task.reset_failure_cost, weight=-10.0),
        "survival": RewardTermCfg(func=standing_task.survival_reward, weight=cfg.survival_weight),
    }
    cfg.terminations["update_state"] = TerminationTermCfg(func=standing_task.update_state)
    cfg.terminations["base_z_roll"] = TerminationTermCfg(func=standing_task.base_z_roll_exceeded)
    cfg.terminations["foot_support_lost"] = TerminationTermCfg(func=standing_task.foot_support_lost)
    cfg.events["reset_transition"].func = standing_task.reset_reverse_transition
    cfg.curriculum = {
        "alignment_to_standing": CurriculumTermCfg(func=standing_task.alignment_to_standing_curriculum)
    }
    cfg.metrics["joint_error"] = MetricsTermCfg(func=standing_task.joint_standing_error_metric)
    cfg.metrics["base_height"] = MetricsTermCfg(func=standing_task.base_height)
    cfg.metrics["base_vertical_velocity"] = MetricsTermCfg(func=standing_task.base_vertical_velocity)
    cfg.metrics["base_support_distance"] = MetricsTermCfg(func=standing_task.base_support_distance)
    cfg.metrics["sole_support_valid"] = MetricsTermCfg(func=standing_task.sole_support_valid_metric)
    cfg.metrics["both_feet_contact"] = MetricsTermCfg(func=standing_task.both_feet_contact_metric)
    cfg.metrics["success"] = MetricsTermCfg(func=standing_task.success, reduce="last")
    cfg.metrics["base_z_roll_deg"] = MetricsTermCfg(
        func=standing_task.base_z_roll_degrees, reduce="max"
    )
    for foot in (0, 1):
        cfg.metrics[f"sole_{foot}_angle_deg"] = MetricsTermCfg(
            func=standing_task.sole_angle_degrees, params={"foot_index": foot}, reduce="mean"
        )
    cfg.metrics["both_soles_within_10deg"] = MetricsTermCfg(
        func=standing_task.both_soles_within_10deg, reduce="mean"
    )
    return cfg


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
    cfg.terminations["update_state"] = TerminationTermCfg(func=robust.update_fixed_standing_state)
    cfg.terminations["fallen_after_standing"] = TerminationTermCfg(func=robust.fallen_after_standing)
    cfg.terminations["early_base_ground_contact"] = TerminationTermCfg(func=robust.early_base_ground_contact)
    cfg.terminations["abandoned_exploration"] = TerminationTermCfg(func=robust.abandoned_exploration)
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="base_ground_contact",
        primary=ContactMatch(mode="body", pattern="base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce",
    ))
    cfg.actions["joint_pos"].max_target_speed = 2 * math.pi
    for group in cfg.observations.values():
        group.terms["last_action"].func = mirrored_last_action
    cfg.events["reset_transition"].func = robust.reset_fall_retry
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
        "joint_standing": RewardTermCfg(func=standing_task.joint_standing_reward, weight=12.0),
        "base_height_progress": RewardTermCfg(func=standing_task.base_height_progress_reward, weight=1.0),
        "base_ascent_velocity": RewardTermCfg(func=robust.initial_ascent_velocity_reward, weight=8.0,
            params={"target_speed": 0.15, "pose_error_range": (0.2, 0.8)}),
        "sole_ground_alignment": RewardTermCfg(func=standing_task.sole_ground_alignment_reward, weight=1.0,
            params={"angle_power": 1.0}),
        "reset_failure": RewardTermCfg(func=standing_task.reset_failure_cost, weight=-10.0),
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
    cfg.events["base_disturbance"].func = robust.RampedBaseImpulse
    cfg.metrics = {
        "joint_error": MetricsTermCfg(func=standing_task.joint_standing_error_metric),
        "base_height": MetricsTermCfg(func=standing_task.base_height),
        "base_vertical_velocity": MetricsTermCfg(func=standing_task.base_vertical_velocity),
        "success": MetricsTermCfg(func=robust.standing_success_metric, reduce="last"),
        "joint_speed_peak": MetricsTermCfg(func=teacher_observations.joint_speed_peak, reduce="max"),
    }
    cfg.scene.sensors = tuple(replace(sensor, fields=("found",)) if isinstance(sensor, ContactSensorCfg) else sensor for sensor in cfg.scene.sensors)
    if play:
        cfg.seed = -1  # 每次启动随机化；仍可通过 --env.seed 复现训练。
        cfg.disturbance_initial_scale = 1.0
    return cfg


def configure_teacher(cfg, play=False):
    cfg.observations = teacher_observations.teacher_observation_groups()
    cfg.actions["joint_pos"].max_target_speed = 2 * math.pi
    return cfg


class TransformationExperiment(Experiment):
    """共享变形训练默认值，由配置类型和工厂选择具体方向。"""

    supports_walking_addons = False
    env_cfg_type = BipedToSnakeEnvCfg
    action_cfg_type = JointTargetRateActionCfg
    compact_standing = False

    def create_env(self, play=False):
        return transformation_env_cfg(play=play, cfg_cls=self.env_cfg_type)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "GaussianDistribution", "init_std": 0.5,
            "std_type": "scalar", "std_range": (0.1, 1.0),
        }
        cfg.algorithm.entropy_coef = 0.005
        return cfg


class BipedToSnake(TransformationExperiment):
    task_id = TASK_ID
    experiment_name = "zbot_transform_biped_to_snake"


class SnakeToBiped(TransformationExperiment):
    task_id = REVERSE_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped"
    env_cfg_type = SnakeToBipedEnvCfg
    runner_cls = SnakeToBipedHotRewardRunner

    def create_env(self, play=False):
        return snake_to_biped_env_cfg(
            play=play, cfg_cls=self.env_cfg_type,
            action_cfg_cls=self.action_cfg_type, compact=self.compact_standing,
        )


class SnakeToBipedBeta(SnakeToBiped):
    """显式有界 Beta 候选方案；保留原 Gaussian 任务。"""

    task_id = BETA_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_beta"
    train_envs = 8192
    max_iterations = 4000

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "BetaDistribution", "action_range": (-1.0, 1.0),
        }
        return cfg

    def edit(self, cfg, play):
        configure_beta(cfg, play)


class SnakeToBipedArc(SnakeToBipedBeta):
    """从与站立目标同分支的低位弧形姿态探索。"""

    task_id = ARC_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_arc"

    def edit(self, cfg, play):
        configure_arc(cfg, play)


class MirroredStandingExperiment(SnakeToBipedBeta):
    """单向鲁棒起立与双向教师共用的镜像配方。"""

    env_cfg_type = RobustSnakeToBipedEnvCfg
    action_cfg_type = MirroredJointTargetRateActionCfg
    compact_standing = True
    runner_cls = RobustSnakeToBipedRunner
    max_iterations = 2000

    def edit(self, cfg, play):
        configure_mirrored_standing(cfg, play)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.resume = False
        cfg.algorithm.learning_rate = 1e-4
        cfg.algorithm.entropy_coef = 0.001
        return cfg


class SnakeToBipedRobust(MirroredStandingExperiment):
    task_id = ROBUST_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_robust"


class TeacherTrainingExperiment(MirroredStandingExperiment):
    """教师共享物理单位缩放及训练参数，观测由具体任务安装。"""

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.actor.obs_normalization = False
        cfg.critic.obs_normalization = False
        cfg.algorithm.learning_rate = 3e-4
        cfg.algorithm.entropy_coef = 0.005
        return cfg


class SnakeToBipedTeacher(TeacherTrainingExperiment):
    """完整状态教师策略，供后续学生策略学习。"""

    task_id = TEACHER_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_teacher"

    def edit(self, cfg, play):
        configure_teacher(cfg, play)
