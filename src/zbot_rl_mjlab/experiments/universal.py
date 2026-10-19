"""通过多教师蒸馏预训练并联合优化的综合运动实验。"""
from dataclasses import dataclass
from typing import cast

from ..distillation import MultiTeacherPpoCfg, MultiTeacherRunner
from ..tasks.universal.env_cfg import TASK_ID, universal_env_cfg
from .addons import (
    BaseDisturbance,
    DomainRandomization,
    GainRandomization,
    MassRandomization,
)
from .base import Experiment


class UniversalLocomotion(Experiment):
    task_id = TASK_ID
    experiment_name = "zbot_universal_locomotion"
    train_envs = 8192
    max_iterations = 3000
    supports_walking_addons = False
    runner_cls = MultiTeacherRunner

    def create_env(self, play=False):
        return universal_env_cfg(play=play)

    def build_ppo(self):
        from ..distillation import universal_ppo_cfg

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

    def build_ppo(self):
        cfg = super().build_ppo()
        # 有限输入无法精确复制全部特权标签，以实际闭环能力决定放行。
        cast(MultiTeacherPpoCfg, cfg.algorithm).validation_use_mse = False
        cfg.validation_randomized = False
        cfg.robust_validation_interval = 100
        return cfg

    def edit(self, cfg, play):
        from ..tasks.universal.deployment import configure_deployment

        configure_deployment(cfg, play=play)
