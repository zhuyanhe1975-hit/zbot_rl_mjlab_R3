"""将指令步频行走独立为实验，便于调整奖励。

继承 ``FlatWalking``（观测、奖励及权重相同），但使用独立任务和实验名注册，
因此修改下表仅影响本任务，不影响 ``Mjlab-Zbot-6dof-Walking`` 的检查点、日志或配方。

全部奖励权重和参数均位于 ``REWARD_WEIGHTS``／``REWARD_PARAMS``。
当前值与 ``Walking`` 相同；``tests/test_walking_tuning.py`` 验证本任务初始配置除
步频范围和训练计划外均与 ``Walking`` 一致，因此改表后测试失败可能是主动调参的结果。

这里的步频由指令指定：策略可见相位（sin、cos）和频率，``frequency_tracking`` 奖励
左右支撑力与该相位一致。下方步频范围将基础配方的 0.5–1.5 Hz 替换为
``train.sh`` 命令行使用的 0.25–1.0 Hz。
"""

from .base import RewardEdit
from .walking import FlatWalking

WALKING_TUNING_TASK_ID = "Mjlab-Zbot-6dof-Walking-Tuning"

# 指令步频范围（Hz）与训练计划。
STEP_FREQUENCY_RANGE = (0.25, 1.0)
NUM_ENVS = 8192
MAX_ITERATIONS = 600

# 奖励权重：正数为奖励，负数为惩罚。
REWARD_WEIGHTS = {
    "frequency_tracking": 1.5,  # 左右支撑力跟随指令相位
    "swing_clearance": 0.5,  # 摆动足达到目标高度
    "forward_velocity": 2.0,  # 有界前进奖励，不指定目标速度
    "lateral_velocity": -0.2,  # 横向速度
    "upright": 0.2,  # 身体保持直立
    "yaw_drift": -0.2,  # 相对初始方向的航向漂移
    "support_foot_slip": -5.0,  # 承重足滑动或扭转
    "soft_landing": -0.02,  # 着地峰值力
    "foot_nonparallel_ground": -5.0,  # 承重足底不平行于地面
    "action_rate": -0.05,  # 相邻控制步的动作变化
    "joint_limits": -0.1,  # 关节位置超限
    "joint_acceleration": -0.0005,  # 关节加速度绝对值之和
}

# 奖励塑形参数，仅列出可调参数的项。此处取值与函数默认值相同，
# 因此显式列出不会改变行为；未列出的奖励项
# 沿用基础配方参数。
REWARD_PARAMS = {
    "forward_velocity": {"scale": 0.2},  # 奖励达到 63 % 时的速度（m/s）
    "swing_clearance": {"target": 0.05},  # 摆动足目标高度（m），宽度为 0.04 m
    "foot_nonparallel_ground": {"std": 0.1},  # 平足惩罚的倾角容差
}


class WalkingTuning(FlatWalking):
    task_id = WALKING_TUNING_TASK_ID
    experiment_name = "zbot_walking_tuning"
    train_envs = NUM_ENVS
    step_frequency = STEP_FREQUENCY_RANGE
    max_iterations = MAX_ITERATIONS

    def reward_edit(self):
        # 在 FlatWalking 自身改动之后应用，因此这些表最终生效。
        return RewardEdit(weights=REWARD_WEIGHTS, params=REWARD_PARAMS)

    def edit(self, cfg, play):
        missing = set(REWARD_WEIGHTS) - set(cfg.rewards)
        unknown = set(cfg.rewards) - set(REWARD_WEIGHTS)
        if missing or unknown:
            raise ValueError(
                f"REWARD_WEIGHTS must list exactly the inherited rewards; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
