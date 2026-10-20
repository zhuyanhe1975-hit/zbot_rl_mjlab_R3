"""独立蛇形教师：沿蛇形前向推进及原有安全复位边界。"""

import math
from dataclasses import dataclass
from typing import Literal, cast

from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef

from ..transformation.actions import JointTargetRateActionCfg
from ..transformation.env_cfg import TransformationEnvCfg, transformation_scene_cfg
from . import mdp
from .commands import SnakeVelocityCommandCfg

TASK_ID = "Mjlab-Zbot-6dof-Snake-Omni-Teacher"
SNAKE_JOINT_LIMIT = math.pi / 3
SNAKE_MAX_TARGET_SPEED = 2 * math.pi


@dataclass
class SnakeEnvCfg(TransformationEnvCfg):
    allow_resume_joint_limit_change: bool = False
    observation_reference: Literal['world_x', 'command'] = 'world_x'
    motion_frame: Literal['body_gravity', 'legacy_head'] = 'body_gravity'
    head_gyro_frame: Literal['head', 'world'] = 'head'
    head_body: str = "foot_0"
    motion_site: str = "foot_0_sole"
    tail_site: str = "foot_1_sole"
    head_forward_axis: tuple[float, float, float] = (0., 0., -1.)
    head_up_axis: tuple[float, float, float] = (1., 0., 0.)
    head_tilt_limit: float = math.pi / 3
    head_yaw_limit: float = math.pi / 3
    body_yaw_limit: float = math.pi / 6
    body_yaw_integral_time_scale: float = 1.
    head_yaw_integral_time_scale: float = 1.
    forward_speed_cap: float = .2
    motion_penalty_min_scale: float = 0.25
    motion_penalty_speed_gain: float = 5.
    workspace_radius: float = 15.
    max_contact_bodyweights: float = 8.
    stall_min_speed: float = .01
    stall_progress_fraction: float = .20
    stall_grace_s: float = 1.
    stall_filter_s: float = .5


def snake_env_cfg(play=False) -> SnakeEnvCfg:
    # 复用全身摩擦、平地、伺服和求解器，替换任务行为，不继承变形奖励。
    cfg = cast(SnakeEnvCfg, transformation_scene_cfg(play=play, cfg_cls=SnakeEnvCfg, action_cfg_cls=JointTargetRateActionCfg))
    cfg.scene.num_envs = 1 if play else 4096
    cfg.episode_length_s = 30.
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0., 0., .0555)
    # XML立柱姿态绕foot_0局部Y轴-90度：局部Z朝世界-X，局部X朝世界+Z。
    robot.init_state.rot = (math.sqrt(.5), 0., -math.sqrt(.5), 0.)
    robot.init_state.joint_pos = {".*": 0.}
    action = cast(JointTargetRateActionCfg, cfg.actions["joint_pos"])
    # 只限制本任务的控制目标，不改XML或运行时模型的机械关节范围。
    action.target_limit = SNAKE_JOINT_LIMIT
    action.max_target_speed = SNAKE_MAX_TARGET_SPEED
    cfg.commands = {"motion": SnakeVelocityCommandCfg(
        training=not play, resampling_time_range=(5., 9.), debug_vis=True,
        translation_probability=.5, turn_probability=.2, stop_probability=.1,
        sample_velocity_commands=False, translation_min_speed=0.,
    )}
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="snake_ground_contact",
            primary=ContactMatch(mode="geom", pattern=".*_geom", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="maxforce", history_length=cfg.decimation,
        ),
        ContactSensorCfg(
            name="snake_self_contact",
            primary=ContactMatch(mode="subtree", pattern="foot_0", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="foot_0", entity="robot"),
            fields=("found", "force", "dist"), reduce="maxforce", history_length=cfg.decimation,
        ),
        BuiltinSensorCfg(name="head_lin_acc", sensor_type="framelinacc", obj=ObjRef(type="xbody", name="foot_0", entity="robot")),
        BuiltinSensorCfg(name="head_ang_acc", sensor_type="frameangacc", obj=ObjRef(type="xbody", name="foot_0", entity="robot")),
    )
    cfg.events = {
        "reset_scene": EventTermCfg(func=common_mdp.reset_scene_to_default, mode="reset"),
        "reset_snake": EventTermCfg(func=mdp.reset_snake, mode="reset"),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg({"snake": ObservationTermCfg(func=mdp.actor_state)}, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg({"snake": ObservationTermCfg(func=mdp.critic_state)}, concatenate_terms=True, enable_corruption=False),
    }
    cfg.rewards = {
        "alive": RewardTermCfg(func=common_mdp.is_alive, weight=.02),
        "head_forward_speed": RewardTermCfg(func=mdp.head_forward_speed, weight=15.),
        "action_rate": RewardTermCfg(func=common_mdp.action_rate_l2, weight=-.0),
        "head_lateral_speed_cost": RewardTermCfg(func=mdp.head_lateral_speed_cost, weight=-5.0),
        "head_vertical_speed_cost": RewardTermCfg(func=mdp.head_vertical_speed_cost, weight=-0.2),
        "joint_pose_deviation": RewardTermCfg(func=mdp.joint_pose_deviation_cost, weight=-.02),
        "joint_fold_cost": RewardTermCfg(func=mdp.joint_fold_cost, weight=-1.,
                                        params={"soft_angle": 2 * math.pi / 3, "full_angle": math.pi}),
        "target_saturation_fraction": RewardTermCfg(func=mdp.target_saturation_fraction, weight=-.1),
        "body_yaw_integral_cost": RewardTermCfg(func=mdp.body_yaw_integral_cost, weight=-1.0),
        "body_yaw_cost": RewardTermCfg(func=mdp.body_yaw_cost, weight=-.1),
        "body_yaw_reset_cost": RewardTermCfg(func=mdp.body_yaw_reset_cost, weight=-5.),
        "head_yaw_integral_cost": RewardTermCfg(func=mdp.head_yaw_integral_cost, weight=-.5),
        "head_yaw_cost": RewardTermCfg(func=mdp.head_yaw_cost, weight=-.5),
        "head_yaw_reset_cost": RewardTermCfg(func=mdp.head_yaw_reset_cost, weight=-5.),
        "head_tilt_reset_cost": RewardTermCfg(func=mdp.head_tilt_reset_cost, weight=-5.0),
        "head_angular_velocity_cost": RewardTermCfg(func=mdp.head_angular_velocity_cost, weight=-.2),
        "self_collision_reset_cost": RewardTermCfg(func=mdp.self_collision_reset_cost, weight=-5.0),
    }
    cfg.terminations = {
        "unsafe": TerminationTermCfg(func=mdp.unsafe),
        "head_tilt": TerminationTermCfg(func=mdp.head_tilt_limit_reached),
        "head_yaw": TerminationTermCfg(func=mdp.head_yaw_limit_reached),
        "body_yaw": TerminationTermCfg(func=mdp.body_yaw_limit_reached),
        "self_collision": TerminationTermCfg(func=mdp.self_collision),
        "time_out": TerminationTermCfg(func=mdp.play_time_out if play else common_mdp.time_out, time_out=True),
    }
    cfg.metrics = {
        **{f"world_velocity_{name}": MetricsTermCfg(
            func=mdp.world_velocity_component, params={"axis": axis})
           for axis, name in enumerate(("x", "y", "z"))},
        "target_saturation_fraction": MetricsTermCfg(func=mdp.target_saturation_fraction),
        "action_clip_fraction": MetricsTermCfg(func=mdp.action_clip_fraction),
        "joint_pose_rms_rad": MetricsTermCfg(func=mdp.joint_pose_rms),
        "joint_velocity_rms_rad_s": MetricsTermCfg(func=mdp.joint_velocity_rms),
        "motion_penalty_scale": MetricsTermCfg(func=mdp.motion_penalty_scale),
        "head_forward_velocity_m_s": MetricsTermCfg(func=mdp.head_forward_velocity),
        "head_angular_speed_rad_s": MetricsTermCfg(func=mdp.head_angular_speed),
        "body_yaw_integral_abs_rad_s": MetricsTermCfg(func=mdp.body_yaw_integral_abs),
        "body_yaw_deviation_rad": MetricsTermCfg(func=mdp.body_yaw_abs_deviation),
        "body_yaw_peak_rad": MetricsTermCfg(func=mdp.body_yaw_abs_deviation, reduce="max"),
        "head_yaw_integral_abs_rad_s": MetricsTermCfg(func=mdp.head_yaw_integral_abs),
        "head_yaw_deviation_rad": MetricsTermCfg(func=mdp.head_yaw_abs_deviation),
        "head_yaw_peak_rad": MetricsTermCfg(func=mdp.head_yaw_abs_deviation, reduce="max"),
        "head_tilt_rad": MetricsTermCfg(func=mdp.head_tilt),
        "head_tilt_peak_rad": MetricsTermCfg(func=mdp.head_tilt, reduce="max"),
        "ground_force_peak_bodyweights": MetricsTermCfg(func=mdp.normalized_ground_peak, reduce="max"),
    }
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.viewer.lookat = (0., 0., .05)
    return cfg
