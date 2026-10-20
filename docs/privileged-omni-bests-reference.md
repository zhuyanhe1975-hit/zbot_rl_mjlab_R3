# 全能教师的成功归档参考

适用任务：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher`。配置直接固化在代码中；运行时不读取归档YAML或旧策略。旧walking等网络不进入教师集合，学生训练继续暂缓。本次未启动完整训练。

## 来源与证据边界

| `bests/`归档 | 用途 | 已有结果的边界 |
| --- | --- | --- |
| `transform_snake_to_biped_teacher_2026-10-17_iter200` | 特权教师PPO基线、Beta动作 | 起立16/16成功；不是全部运动技能的验收 |
| `transform_snake_biped_bidirectional_2026-10-17_iter599` | 相同PPO设置、`final_effective_weights.json`中的变形最终权重 | 有双向切换记录；不能推导全向跟踪全部达标 |
| `snake_world_x_2026-10-19_iter1499` | 积分角速度动作、控制周期、蛇形±60°限制 | 世界X单向爬行验证；不能直接复用世界方向奖励 |
| `deploy_walking_2026-10-11_iter1499` | 归一化能耗、支撑足滑移、足底平行代价 | 行走任务公式与单位；不加载其策略网络 |
| `velocity_energy_disturbed_2026-10-10_iter499` | 足部摩擦、基座COM与外扰幅度 | 受扰行走经验；全能变形下需重新验收 |
| `omni_velocity_energy_2026-10-10_iter499` | 核对多轴速度跟踪与运动控制设置 | 原地偏航仍弱，不作为全向能力已经解决的证据 |

## 已落实的训练设置

PPO直接对齐两份成功变形归档：初始学习率`3e-4`，熵系数`0.005`，adaptive schedule，5个epoch，4个mini-batch，clip`0.2`，gamma`0.99`，GAE lambda`0.95`，目标KL`0.01`，梯度上限`1`，value loss系数`1`，clipped value loss，Adam；每环境24控制步，每50轮保存。显式设置这些字段，避免上游默认值改变配方。

v2保留59维特权观测、观测归一化，网络恢复所有主要成功归档共有的`256/128`、ELU。六关节Beta用于蛇爬/变形积分控制；稳定双足借鉴归档默认站姿加`0.25×动作`的位置控制，采用有界Beta并限速衔接。这不是与旧Gaussian无限尾分布完全相同的动作。物理步长`0.005s`、decimation`4`、KP`15`、KD`1.5`、armature`0.001`、力矩上限`20`沿用既有控制设置。蛇形稳定爬行目标限制±π/3；变形端关节采用归档站姿中心±1rad，其他关节保留原范围。v2网络和动作均改变，schema=2，须新run，不能完整续训v1。

变形最终有效权重复用双向归档：

| 奖励项 | 权重 |
| --- | ---: |
| joint_standing | 12 |
| base_height_progress | 1 |
| base_ascent_velocity | 8 |
| sole_ground_alignment | 1 |
| reset_failure | -10 |
| stand_finish_joint_speed | -0.02 |
| descent_ground_impact | -2 |

姿态误差尺度`0.5`、上升目标速度`0.15m/s`、就位衰减范围`0.2–0.8rad`、上升奖励停止高度`0.23m`、足底高度std`0.03m`、角度std`10°`也沿用归档。失败只按联合终止掩码计一次，除以控制dt补偿奖励管理器积分，保持单次失败代价`-10`。

能耗采用归档公式`clamp(sum(abs(tau*qdot))/10, max=1)`，权重`-0.5`，替代未归一化功率乘`-0.002`；稳定蛇运动额外按探索进展缩放能耗，这是联合任务适配。支撑足滑移权重`-5`；足底不平行权重`-5`、std`0.1`，仅在双足稳定运动阶段生效。

随机化采用足部`a1_geom/b6_geom`摩擦`0.3–1.2`、基座COM偏移±`(0.025,0.025,0.03)m`，不新增质量和PD增益随机化，不向真值关节观测添加编码器偏差。外扰沿用±`5N`、±`0.5Nm`、25%环境、持续`0.08–0.2s`、间隔`0.8–2s`。

## 明确属于联合任务适配的部分

以下设置需要本次全能训练验证，不属于归档直接证明有效的组合：

- 变形奖励按transition阶段门控，足底代价按稳定双足阶段门控；上升峰值记忆在重置和新指令时独立清理。
- 双足site坐标速度采用归档三轴高斯核，std为`.1/.1/.25`、轴权重`2/1/1`；蛇形采用归档有方向推进及低速探索调节，方向改为site下的指令方向。保留`mode_posture=2`、`transition_progress=2`和`standstill=1`。动作变化代价双足`-.05`、其他阶段`-.01`。不复制世界X固定航向或全局镜像三动作。
- 外扰由0开始，在500轮内渐增。变形归档无扰动，因此不从第一轮直接施加受扰行走的全部强度。
- 8192环境、6000轮和全向幅值课程保持本实验配置；归档轮数不代表复杂联合任务的收敛轮数。

v2必须使用新run进行对比；旧v1不能完整续训。观测顺序保持59维，网络容量和动作解释已经改变。完整续训仍会恢复优化器状态，仅适用于相同版本的配方。

## 可复核入口

- `src/zbot_rl_mjlab/omni_teacher.py`：显式PPO配置。
- `src/zbot_rl_mjlab/experiments/omni_teacher.py`：动作、随机化与奖励接线。
- `src/zbot_rl_mjlab/tasks/universal/teacher_rewards.py`：归档奖励公式复用与阶段适配。
- `tests/test_teacher_archive_recipe.py`、`tests/test_teacher_archive_rewards.py`：直接读取归档YAML/JSON，对比参数、最终权重、公式和单位，验证阶段门控与记忆重置。

配置对齐不等于全能技能已达标；仍以本实验两种构型、各方向和双向变形的独立闭环验收为准。

完整逐项核对及第300轮诊断见[全面配方审计](privileged-omni-recipe-audit.md)。

当前v3已修复指令缩幅与环境配额，见[停滞诊断与网络规模对照](omni-v2-stall-diagnosis.md)。上文中的v2课程描述不再作为当前教师采样设置。
