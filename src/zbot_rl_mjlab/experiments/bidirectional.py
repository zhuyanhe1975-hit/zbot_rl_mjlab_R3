"""独立的蛇形／双足双向指令实验。"""


from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..tasks.transformation import commanded
from ..tasks.transformation.commanded import (
    TASK_ID,
    BidirectionalEnvCfg,
)
from .transformation import TeacherTrainingExperiment


def configure_bidirectional(cfg, play=False):
    """在共享镜像任务上安装双向指令、奖励、终止和指标。"""
    cfg.pretrained_checkpoint = None
    cfg.initial_roll_range_deg = (-180.0, 180.0)
    cfg.early_contact_drop_threshold = 0.01
    cfg.fall_retry_arm_height = 0.05
    cfg.episode_length_s = 16.0
    cfg.commands = {"transform": commanded.TransformCommandCfg(resampling_time_range=(1e9, 1e9))}
    cfg.observations = {
        "actor": ObservationGroupCfg(terms={"command_teacher_state": ObservationTermCfg(func=commanded.actor_state)}, enable_corruption=False),
        "critic": ObservationGroupCfg(terms={"command_teacher_critic_state": ObservationTermCfg(func=commanded.critic_state)}, enable_corruption=False),
    }
    cfg.events["reset_transition"].func = commanded.reset_command_transition
    for name, func in (("joint_standing", commanded.pose_reward), ("base_height_progress", commanded.height_reward), ("base_ascent_velocity", commanded.velocity_reward), ("sole_ground_alignment", commanded.sole_reward)):
        cfg.rewards[name].func = func
    cfg.rewards["stand_finish_joint_speed"] = RewardTermCfg(
        func=commanded.stand_finish_joint_speed_cost, weight=-0.02,
    )
    cfg.rewards["descent_ground_impact"] = RewardTermCfg(
        func=commanded.descent_ground_impact_cost, weight=-2.0,
        params={"max_body_weights": 2.0},
    )
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="transform_ground_contact",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",), reduce="netforce", history_length=cfg.decimation,
    ))
    for name, func in (("update_state", commanded.update_state), ("fallen_after_standing", commanded.fallen), ("early_base_ground_contact", commanded.early_contact), ("abandoned_exploration", commanded.abandoned_exploration)):
        cfg.terminations[name].func = func
    if play:
        cfg.terminations["time_out"].func = commanded.play_time_out
        for name in ("fallen_after_standing", "early_base_ground_contact", "abandoned_exploration"):
            term = cfg.terminations[name]
            term.params = {"condition": term.func}
            term.func = commanded.play_failure
    cfg.metrics["joint_error"].func = commanded.joint_error
    cfg.metrics["success"].func = commanded.success
    cfg.metrics["success"].reduce = "max"
    cfg.metrics["transform_command"] = MetricsTermCfg(func=commanded.command_metric, reduce="last")
    cfg.metrics["stand_env_fraction"] = MetricsTermCfg(func=commanded.stand_env_fraction)
    cfg.metrics["stand_finish_joint_speed"] = MetricsTermCfg(func=commanded.stand_finish_joint_speed_cost)
    cfg.metrics["ground_force_peak_n"] = MetricsTermCfg(func=commanded.ground_force_peak, reduce="max")
    return cfg


class SnakeBipedBidirectional(TeacherTrainingExperiment):
    """共用镜像训练基础，直接装配双向行为与指令观测。"""

    task_id = TASK_ID
    experiment_name = "zbot_transform_snake_biped_bidirectional"
    env_cfg_type = BidirectionalEnvCfg

    def edit(self, cfg, play):
        configure_bidirectional(cfg, play)
