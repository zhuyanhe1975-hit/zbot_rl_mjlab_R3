import math
from functools import partial
from itertools import pairwise
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ASSET_PATH = Path(__file__).resolve().parents[2] / "assets/zbot_6/ZBOT6.xml"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
DEFAULT_JOINT_POS = (0.312, 0.837, -2.02, 2.02, -0.837, -0.312)

# RobStride RS00（灵足时代）一体式准直驱关节，参数对应输出轴。
# 来源：RS00 使用说明书（260713），手册章节 1.2–1.4（第 6–9 页）、4.1（第 31–32 页）、
# 4.3.1（第 41 页）。已建模与未建模内容见 docs/rs00-actuator.md。
RS00_PEAK_TORQUE = 14.0  # N*m，“峰值负载”；未应用于仿真器，见下方说明
RS00_RATED_TORQUE = 5.0  # N*m，持续力矩，采用 90x85 mm 散热片、在 100 rpm 下测量
RS00_NO_LOAD_SPEED = 315.0 * 2.0 * math.pi / 60.0  # rad/s，315 rpm +/-10 %，约 33.0
RS00_KP_RANGE = (0.0, 500.0)  # N*m/rad，MIT 模式指令编码（第 32 页）
RS00_KD_RANGE = (0.0, 5.0)  # N*m*s/rad，MIT 模式指令编码（第 32 页）

# 48 V 下的 T-N 曲线（手册第 7 页，从图读取，误差 +/-0.5 N*m）：(rpm, N*m)。
# 仿真器不应用此曲线（见 docs/rs00-actuator.md）；
# scripts/actuator_usage.py 用它对照策略请求的关节速度与力矩。
RS00_TN_CURVE = (
    (100.0, 14.0),
    (150.0, 12.0),
    (200.0, 9.5),
    (250.0, 6.5),
    (280.0, 4.0),
    (300.0, 2.0),
    (315.0, 0.5),
)
RS00_CORNER_SPEED = 100.0 * 2.0 * math.pi / 60.0  # rad/s，此速度以内保持峰值力矩

# 旋转过载限制（手册第 8 页，同一散热片）：负载可持续的最长时间。
# （秒，N*m）；5 N*m 可持续运行。
RS00_OVERLOAD = (
    (3.0, 14.0),
    (7.0, 13.0),
    (10.0, 12.0),
    (20.0, 10.0),
    (53.0, 8.0),
    (94.0, 7.0),
    (285.0, 6.0),
)

# 增益为部署时发送的 MIT 模式 Kp、Kd，均处于 MIT 允许范围。
# 手册未给出转子惯量，因此附加转动惯量以及减速器摩擦、
# 回程间隙均未经核实。
STIFFNESS = 15.0  # N*m/rad
DAMPING = 1.5  # N*m*s/rad
ARMATURE = 0.001
# 仿真器实际允许的力矩。迄今所有检查点均使用此值训练；
# 硬件峰值为 RS00_PEAK_TORQUE（14），仿真步态可能要求
# 实机电机无法提供的力矩。scripts/actuator_usage.py 用于测量超限程度。
SIM_EFFORT_LIMIT = 20.0


def available_torque(speed):
    """通过插值计算 T-N 曲线在当前关节速度（rad/s）下允许的力矩（N*m）。

    100 rpm 以内保持峰值力矩，之后沿数据手册给出的点插值，
    直到 315 rpm 空载转速处的 0.5 N*m；超过最后一点后为零。
    仅供校核参考，仿真器不应用此曲线。
    """
    rpm = abs(speed) * 60.0 / (2.0 * math.pi)
    if rpm <= RS00_TN_CURVE[0][0]:
        return RS00_PEAK_TORQUE
    for (r0, t0), (r1, t1) in pairwise(RS00_TN_CURVE):
        if rpm <= r1:
            return t0 + (t1 - t0) * (rpm - r0) / (r1 - r0)
    return 0.0


def get_spec(contact_dimension=3):
    spec = mujoco.MjSpec.from_file(str(ASSET_PATH))
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            # 对齐 G1 的碰撞设置：仅足部使用三维摩擦接触；
            # 内部／自碰撞几何体使用 condim=1。
            is_foot = geom.name.startswith(("foot_0_", "foot_1_"))
            geom.condim = contact_dimension if is_foot else 1
            if is_foot:
                geom.friction = (0.6, geom.friction[1], geom.friction[2])
    return spec


def robot_cfg():
    return EntityCfg(
        spec_fn=partial(get_spec),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # 保留原执行器配置，以便所有既有检查点仍可回放：
                # 隐式积分位置伺服、15 / 1.5 增益、20 N*m 限制。
                # 上方 RS00 常量未在此应用（实际峰值为 14 N*m）；
                # 曾尝试切换为符合数据手册的执行器，但破坏了既有模型回放，
                # 故在重新训练前不作此切换。
                BuiltinPositionActuatorCfg(
                    target_names_expr=JOINT_NAMES,
                    stiffness=STIFFNESS,
                    damping=DAMPING,
                    effort_limit=SIM_EFFORT_LIMIT,
                    armature=ARMATURE,
                ),
            )
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, -0.06, 0.0),
            joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS)),
            joint_vel={".*": 0.0},
        ),
    )
