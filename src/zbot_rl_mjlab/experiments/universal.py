"""通过多教师蒸馏预训练并联合优化的综合运动实验。"""


import math
from dataclasses import asdict, dataclass
from typing import cast

from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..distillation import MultiTeacherPpoCfg, MultiTeacherRunner, MultiTeacherRunnerCfg
from ..tasks.transformation import mdp as transform
from ..tasks.transformation import teacher_observations as teacher
from ..tasks.transformation.actions import JointTargetRateActionCfg
from ..tasks.transformation.env_cfg import transformation_scene_cfg
from ..tasks.universal import deployment as deployment_runtime
from ..tasks.universal import mdp
from ..tasks.universal.commands import UniversalCommandCfg
from ..tasks.universal.env_cfg import TASK_ID, UniversalEnvCfg
from .addons import (
    BaseDisturbance,
    DomainRandomization,
    GainRandomization,
    MassRandomization,
)
from .base import Experiment


def universal_env_cfg(play=False, *, cfg_cls: type[UniversalEnvCfg] = UniversalEnvCfg) -> UniversalEnvCfg:
    cfg = cast(UniversalEnvCfg, transformation_scene_cfg(play=play, cfg_cls=cfg_cls))
    cfg.scene.num_envs = 1 if play else 8192
    cfg.episode_length_s = 45.
    action = cast(JointTargetRateActionCfg, cfg.actions["joint_pos"])
    action.max_target_speed = 2 * math.pi
    action.target_limit = math.pi
    cfg.commands = {"transform": UniversalCommandCfg(resampling_time_range=(6., 12.), training=not play, mode="stand" if play else "source")}
    cfg.observations = {
        "actor": ObservationGroupCfg({"universal_state": ObservationTermCfg(func=mdp.actor_state)}, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg({"universal_critic_state": ObservationTermCfg(func=mdp.critic_state)}, concatenate_terms=True, enable_corruption=False),
    }
    cfg.scene.sensors = (
        ContactSensorCfg(name="base_ground_contact", primary=ContactMatch(mode="body", pattern="base", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce"),
        ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce"),
        ContactSensorCfg(name="transform_ground_contact", primary=ContactMatch(mode="body", pattern=".*", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("force",), reduce="netforce", history_length=cfg.decimation),
    )
    cfg.events["reset_transition"] = EventTermCfg(func=mdp.reset_universal, mode="reset")
    cfg.terminations = {
        "unsafe": TerminationTermCfg(func=mdp.unsafe),
        "demonstration_time_out": TerminationTermCfg(func=mdp.demonstration_time_out, time_out=True),
        "time_out": TerminationTermCfg(func=mdp.play_time_out if play else common_mdp.time_out, time_out=True),
    }
    cfg.rewards = {
        "velocity_tracking": RewardTermCfg(func=mdp.velocity_tracking, weight=4.),
        "mode_posture": RewardTermCfg(func=mdp.posture_reward, weight=2.),
        "transition_progress": RewardTermCfg(func=mdp.transition_progress_reward, weight=2.),
        "standstill": RewardTermCfg(func=mdp.standstill_reward, weight=1.),
        "energy": RewardTermCfg(func=mdp.energy_cost, weight=-.002),
        "action_rate": RewardTermCfg(func=mdp.action_rate_cost, weight=-.01),
        "contact_impact": RewardTermCfg(func=mdp.impact_cost, weight=-.2),
        "failure": RewardTermCfg(func=mdp.failure_cost, weight=-5.),
    }
    cfg.metrics = {
        **{
            f"phase_{name}_fraction": MetricsTermCfg(func=mdp.phase_fraction, params={"phase": phase})
            for phase, name in enumerate(("biped", "descending", "snake", "ascending"))
        },
        "biped_motion_normalized_velocity_error": MetricsTermCfg(func=mdp.mode_motion_error, params={"phase": 0}),
        "snake_motion_normalized_velocity_error": MetricsTermCfg(func=mdp.mode_motion_error, params={"phase": 2}),
    }
    return cfg


def universal_ppo_cfg() -> MultiTeacherRunnerCfg:
    """六独立速度动作的综合策略，价值网络从头学习。"""
    return MultiTeacherRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128), activation="elu", obs_normalization=False,
            distribution_cfg={"class_name": "BetaDistribution", "action_range": (-1.0, 1.0)},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=False),
        experiment_name="zbot_universal_locomotion",
        max_iterations=3000, num_steps_per_env=24, save_interval=50,
        logger="tensorboard",
    )


def configure_deployment(cfg, play=False):
    """直接定义单帧 37 维可测状态，训练与回放均使用五帧无噪声历史。"""
    cfg.straighten_before_stand = True
    cfg.commands['transform'] = deployment_runtime.DeploymentCommandCfg(**asdict(cfg.commands['transform']))
    cfg.rewards['straightening'] = RewardTermCfg(func=deployment_runtime.straightening_cost, weight=-2.)
    cfg.events['reset_transition'].func = deployment_runtime.BentSnakeReset
    cfg.events['reset_transition'].params = {
        'bend_fraction': .5, 'bend_limit': .2, 'roll_limit_deg': 40.,
    }
    functions = {
        "joint_pos": deployment_runtime.joint_position,
        "joint_vel": teacher.joint_velocity,
        "commanded_target": transform.commanded_target,
        "actions": transform.last_action,
        "projected_gravity": transform.projected_gravity,
        "base_ang_vel": teacher.base_angular_velocity,
        "command": deployment_runtime.command,
        "gait": mdp.gait_observation,
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms={name: ObservationTermCfg(func=deployment_runtime._sanitized(func)) for name, func in functions.items()},
        concatenate_terms=True,
        enable_corruption=False,
        history_length=5,
        flatten_history_dim=True,
        # 逐项无条件清理与 MJLab 的 sanitize 结果相同，但不在每项先 .any() 同步。
        nan_policy="disabled",
    )


class UniversalLocomotion(Experiment):
    task_id = TASK_ID
    experiment_name = "zbot_universal_locomotion"
    train_envs = 8192
    max_iterations = 3000
    supports_walking_addons = False
    runner_cls = MultiTeacherRunner

    def create_env(self, play=False):
        return universal_env_cfg(play=play)

    def _build_ppo(self):
        cfg = universal_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg


DEPLOY_TASK_ID = TASK_ID + "-Deploy"


@dataclass(frozen=True)
class StudentDisturbance(BaseDisturbance):
    """前期轻扰动示范，随后由同一受扰学生通过 PPO 学习恢复。"""

    pretraining_scale: float = .2
    ramp_iterations: int = 500

    def __post_init__(self):
        import math

        if not math.isfinite(self.pretraining_scale) or not 0 <= self.pretraining_scale <= 1:
            raise ValueError("预热外扰比例必须在 0 到 1 之间")
        if type(self.ramp_iterations) is not int or self.ramp_iterations <= 0:
            raise ValueError("外扰增长轮数必须为正整数")

    def apply(self, cfg, play):
        from ..tasks.universal.deployment import StudentBaseImpulse

        super().apply(cfg, play)
        event = cfg.events['base_disturbance']
        event.func = StudentBaseImpulse
        event.params.update(pretraining_scale=self.pretraining_scale, ramp_iterations=self.ramp_iterations)


class DeployUniversalLocomotion(UniversalLocomotion):
    """从头培养部署受限学生，同一策略从蒸馏进入鲁棒节能 PPO。"""

    task_id = DEPLOY_TASK_ID
    experiment_name = "zbot_universal_locomotion_deploy"
    addons = (
        MassRandomization(), GainRandomization(), StudentDisturbance(),
        DomainRandomization(geom_names=(".*",), encoder_bias_range=None),
    )

    def _build_ppo(self):
        cfg = super()._build_ppo()
        # 有限输入无法精确复制全部特权标签，以实际闭环能力决定放行。
        cast(MultiTeacherPpoCfg, cfg.algorithm).validation_use_mse = False
        cfg.validation_randomized = False
        cfg.robust_validation_interval = 100
        return cfg

    def edit(self, cfg, play):
        configure_deployment(cfg, play=play)


class SnakeLegDeploy(DeployUniversalLocomotion):
    """蛇爬、腿走和双向变形共享一个部署学生的独立实验。"""

    task_id = "Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy"
    experiment_name = "zbot_universal_snake_leg_deploy"

    def _build_ppo(self):
        from ..teachers import SNAKE_CHECKPOINT

        cfg = super()._build_ppo()
        cast(MultiTeacherPpoCfg, cfg.algorithm).teacher_checkpoints = {'snake': SNAKE_CHECKPOINT}
        return cfg

    def edit(self, cfg, play):
        cfg.snake_teacher_enabled = True
        cfg.commands['transform'].snake_teacher_enabled = True


class SnakeLegOmniDeploy(DeployUniversalLocomotion):
    """两种形态共享site全向指令；只用变形教师辅助，从首轮联合PPO。"""

    task_id = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy'
    experiment_name = 'zbot_universal_snake_leg_omni_deploy'

    def _build_ppo(self):
        cfg = super()._build_ppo()
        algorithm = cast(MultiTeacherPpoCfg, cfg.algorithm)
        algorithm.enabled_teachers = ('transform',)
        algorithm.teacher_warmup_enabled = False
        algorithm.validation_interval_iterations = 100
        cfg.robust_validation_interval = 500
        return cfg

    def edit(self, cfg, play):
        from ..tasks.universal.omni import OmniCommandCfg, reset_omni

        cfg.omni_site_frame = True
        cfg.motion_site = 'foot_0_sole'
        cfg.commands['transform'] = OmniCommandCfg(**asdict(cfg.commands['transform']))
        cfg.events['reset_transition'].func = reset_omni
        cfg.events['reset_transition'].params = {}
