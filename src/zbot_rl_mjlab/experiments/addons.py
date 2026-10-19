"""提升实验鲁棒性或面向部署训练的可选扩展。

扩展是带参数的小对象。实验在 ``addons`` 属性中列出所需扩展；所有继承的奖励和
环境改动完成后，再应用扩展，因此同一扩展适用于任意实验：

    class DisturbedWalking(FlatWalking):
        addons = DISTURBED                       # 预设：扰动与随机化

    class MyOmni(OmniVelocityEnergy):
        addons = (BaseDisturbance(disturbed_fraction=1.0), ActionDelay(max_lag=3))

扩展沿继承链累积。若子类声明的扩展类型已在父类中存在，则以子类为准，使子类能够
单独调整父类的扰动配置而无需重复其他扩展。每个扩展都有固定 ``order``，因此结果
不依赖声明顺序。顺序影响的一处是：``MassRandomization`` 会从默认值重写刚体质心，
故它先于移动基座质心的 ``DomainRandomization`` 执行。

仅训练生效的效果（观测噪声、动作延迟）在回放配置中关闭，与策略观测的
``enable_corruption`` 开关保持一致。
"""

from dataclasses import dataclass, field, replace
from typing import ClassVar

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
    UniformNoiseCfg,
)

from ..tasks.walking.actions import DelayedJointPositionActionCfg
from .common import add_base_disturbance, add_domain_randomization

# 实机不能直接测量的观测项，仅向价值网络提供。
PRIVILEGED_TERMS = ("base_lin_vel", "yaw_error")

# 策略观测的逐步高斯噪声标准差。
NOISE_STD = {
    "base_ang_vel": 0.05,  # rad/s，陀螺仪
    "projected_gravity": 0.02,  # 单位向量分量，加速度计／滤波器
    "joint_pos": 0.01,  # rad，编码器
    "joint_vel": 0.15,  # rad/s，编码器差分
}
# 每个回合采样一次偏置，在 +/- 指定值内均匀分布。
NOISE_BIAS = {
    "base_ang_vel": 0.02,  # rad/s，陀螺仪零偏
    "projected_gravity": 0.05,  # 各轴约 3 度的 IMU 倾斜
}

MAX_ACTION_LAG = 2  # 控制步数（每步 20 ms）
GAIN_SCALE = (0.7, 1.3)  # 逐环境、逐回合的 PD 刚度与阻尼
MASS_ALPHA = (-0.07, 0.07)  # 质量和惯量按 exp(2 * alpha) 缩放，约 +/-15 %


@dataclass(frozen=True)
class Addon:
    """对实验环境配置的一项可选改动。"""

    order: ClassVar[int]
    """固定应用顺序中的位置（数值越小越先应用）。"""

    def apply(self, cfg, play):
        raise NotImplementedError


@dataclass(frozen=True)
class BlindActor(Addon):
    """不向策略网络提供实机无法测量的量；价值网络仍保留这些量。"""

    order: ClassVar[int] = 10
    hidden: tuple[str, ...] = PRIVILEGED_TERMS

    def apply(self, cfg, play):
        actor = cfg.observations["actor"]
        actor.terms = {
            name: term for name, term in actor.terms.items() if name not in self.hidden
        }


@dataclass(frozen=True)
class ObservationNoise(Addon):
    """策略观测的高斯噪声，并为部分观测项添加逐回合偏置。"""

    order: ClassVar[int] = 20
    std: dict[str, float] = field(default_factory=lambda: dict(NOISE_STD))
    bias: dict[str, float] = field(default_factory=lambda: dict(NOISE_BIAS))

    def apply(self, cfg, play):
        actor = cfg.observations["actor"]
        known = set(actor.terms) | set(cfg.observations["critic"].terms)
        unknown = (set(self.std) | set(self.bias)) - known
        if unknown:
            raise KeyError(f"no observation term named {sorted(unknown)}")
        if set(self.bias) - set(self.std):
            raise ValueError("bias is only added on top of terms that also have std")
        actor.terms = {
            name: self._noisy(term, name) if name in self.std else term
            for name, term in actor.terms.items()
        }

    def _noisy(self, term, name):
        noise = GaussianNoiseCfg(std=self.std[name])
        if name in self.bias:
            noise = NoiseModelWithAdditiveBiasCfg(
                noise_cfg=noise,
                bias_noise_cfg=UniformNoiseCfg(
                    n_min=-self.bias[name], n_max=self.bias[name]
                ),
            )
        return replace(term, noise=noise)


@dataclass(frozen=True)
class ActionDelay(Addon):
    """关节位置目标延迟 0..max_lag 个控制步后到达执行器。"""

    order: ClassVar[int] = 30
    max_lag: int = MAX_ACTION_LAG

    def apply(self, cfg, play):
        joint_pos = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = DelayedJointPositionActionCfg(
            entity_name=joint_pos.entity_name,
            actuator_names=joint_pos.actuator_names,
            scale=joint_pos.scale,
            offset=joint_pos.offset,
            use_default_offset=joint_pos.use_default_offset,
            clip=joint_pos.clip,
            max_lag=0 if play else self.max_lag,
        )


@dataclass(frozen=True)
class MassRandomization(Addon):
    """启动时同步缩放连杆质量与惯量，以保持物理一致性。"""

    order: ClassVar[int] = 40
    alpha_range: tuple[float, float] = MASS_ALPHA

    def apply(self, cfg, play):
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={"alpha_range": self.alpha_range},
        )


@dataclass(frozen=True)
class GainRandomization(Addon):
    """每次重置时，逐环境缩放 PD 刚度和阻尼。"""

    order: ClassVar[int] = 45
    scale: tuple[float, float] = GAIN_SCALE

    def apply(self, cfg, play):
        cfg.events["randomize_pd_gains"] = EventTermCfg(
            mode="reset",
            func=dr.pd_gains,
            params={
                "kp_range": self.scale,
                "kd_range": self.scale,
                "operation": "scale",
            },
        )


@dataclass(frozen=True)
class BaseDisturbance(Addon):
    """对部分环境的基座施加随机力和力矩冲量。"""

    order: ClassVar[int] = 50
    force_range: tuple[float, float] = (-5.0, 5.0)  # N
    torque_range: tuple[float, float] = (-0.5, 0.5)  # N*m
    duration_s: tuple[float, float] = (0.08, 0.20)
    cooldown_s: tuple[float, float] = (0.8, 2.0)
    disturbed_fraction: float = 0.25

    def apply(self, cfg, play):
        add_base_disturbance(
            cfg,
            force_range=self.force_range,
            torque_range=self.torque_range,
            duration_s=self.duration_s,
            cooldown_s=self.cooldown_s,
            disturbed_fraction=self.disturbed_fraction,
        )


@dataclass(frozen=True)
class DomainRandomization(Addon):
    """启动时随机化足部摩擦、编码器零偏与基座质心。"""

    order: ClassVar[int] = 60
    friction_range: tuple[float, float] = (0.3, 1.2)
    geom_names: tuple[str, ...] = ("a1_geom", "b6_geom")
    encoder_bias_range: tuple[float, float] | None = (-0.015, 0.015)  # rad
    base_com_ranges: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {
            0: (-0.025, 0.025),
            1: (-0.025, 0.025),
            2: (-0.03, 0.03),
        }
    )  # m，沿基座刚体 x、y、z 轴

    def apply(self, cfg, play):
        add_domain_randomization(
            cfg,
            friction_range=self.friction_range,
            geom_names=self.geom_names,
            encoder_bias_range=self.encoder_bias_range,
            base_com_ranges=self.base_com_ranges,
        )


# 预设。
DISTURBED = (BaseDisturbance(), DomainRandomization())
"""随机冲击与启动随机化，即 ``*-Disturbed-*`` 任务采用的配置。"""

DEPLOYMENT = (
    BlindActor(),
    ObservationNoise(),
    ActionDelay(),
    MassRandomization(),
    GainRandomization(),
    BaseDisturbance(),
    DomainRandomization(),
)
"""缩小仿真到实机差距的完整配置：受限且带噪声的策略观测、控制延迟和全面随机化。
策略网络输入维度减少，因此此任务无法从未使用该配置的同一实验检查点热启动。"""
