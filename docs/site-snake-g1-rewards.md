# 蛇全向任务：蛇头系相对速度线性奖励与 alive 奖励 v16

> 2026-10-21：原 `Snake-Site-Omni-Teacher` 已按iter599归档恢复56维Beta、指数奖励和±60°目标。本文后续61维Gaussian及相对得分迭代保留作实验记录，自适应/base/forward任务仍使用各自配方。当前回放入口见[README](../README.md)。

当前site归档兼容实验的卡片增量：静止采样概率为1%；训练回合15秒触发timeout截断并自动复位，交互回放保留手动复位；增加 `site_tilt_angular_velocity`，初始权重−0.05。该项惩罚site世界角速度垂直于重力的分量平方和（标准重力下为ωx²+ωy²），抑制快速翻滚/俯仰，不计入绕竖直轴的yaw转向；原 `site_tilt` 姿态惩罚保留。权重在[实验卡片](../src/zbot_rl_mjlab/experiments/cards/snake_site_omni_teacher.json)修改，训练和回放共用。

当前归档兼容实验的平移跟踪已拆为 `track_forward_velocity`（vx）和 `track_lateral_velocity`（vy），拆分时权重各1.0，最高贡献合计2.0。两项分别计算单轴指数误差并相加，现改为各自固定的 `std`（初设0.1 m/s），不随速度指令大小变化；奖励形式为 `exp(-((v_actual-v_command)/std)^2)`。误差惩罚不再由另一轴的实际速度共同决定，权重及带宽参数均从实验卡片修改。

新增 `wz_velocity`：对运动site绕世界竖直轴的实际角速度平方扣分，初始权重−0.05，即 `-0.05*wz_actual^2`。它不依赖wz指令，左右转动都会扣分；与 `track_angular_velocity` 跟踪项独立，权重在实验卡片设置。

训练采样新增 `train.env.commands.transform.motion_x_only_probability`，初设为0.5：全部请求中50%仅vx非零，前进/后退各约25%；静止仍为1%，剩余49%按原相对比例覆盖纯vy、纯wz和组合运动。该字段未设置或为null时沿用原采样分布；它与静止概率之和不得超过1。本次只配置训练覆盖，回放随机采样保持原分布。

适用任务：`Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher`。

速度跟踪点为 **foot_0_sole site**，水平前向取该 site 当前 −Z 的水平投影，左向为水平前向逆时针 90°。
前后、左右速度随蛇头当前朝向转动；yaw 速度取 site 绕世界竖直轴的角速度。
不设置初始或积分目标 heading，不积累 yaw 指令为朝向目标，也不使用 heading 偏差奖励或观测。
指令箭头、速度观测、奖励、误差指标均采用同一个实时蛇头参考系。蛇头水平轴退化时使用既有 site frame 的有效方向回退。

## 速度误差与奖励

各轴有符号误差为 `e_i=(c_i-v_i)/L_i`：

- `c_i`：当前平滑后的速度指令；`v_i`：site 当前真实速度在相应轴上的标量。
- `L_i`：command 配置的该轴速度上限 `velocity_limits`，以当前配置为准。
- 误差不裁剪。指令 +0.2、实际 −0.2、边界 0.2 时误差为 +2，并不保证处于 ±1。
- 零指令附近的 0.01 m/s 波动相对 0.2 m/s 边界产生 0.05 的误差；没有除以微小指令的问题。

非零指令的速度项按相对误差线性计分：`r_i=clamp(1-2*abs(c_i-v_i)/abs(c_i), -2, 1)`。
前向、侧向、yaw 分别独立调权；观测中的有符号误差 `e_i` 保持原样。

- 非零指令下，实际速度为指令的0% / 25% / 50% / 75% / 100%时，分别得 −1 / −0.5 / 0 / 0.5 / 1。
- 超速对称扣分：125% / 150% / 200% 分别得0.5 / 0 / −1；以同样幅值反向运动原本得−3，现裁剪为−2。每轴原始分项统一限幅到[-2,1]，停车漂移项也适用。
- 各轴分别计分，正负指令对称，改变指令幅值不改变上述比例关系。
- 当该轴指令绝对值不超过1e-6时按停车处理：静止为0，漂移得 `-abs(v_i)/L_i`，避免除零。微小非零指令超过此阈值后仍采用相对误差，对速度误差更敏感。
- 观测中的误差仍用固定速度上限归一化，与相对奖励分母不同。

| 分项 | 定义 |
|---|---|
| alive | 健康步为1，失败终止步为0；权重沿用配置 |
| track_forward_velocity | `1-2*abs(c_x-v_x)/abs(c_x)`（非零指令） |
| track_lateral_velocity | `1-2*abs(c_y-v_y)/abs(c_y)`（非零指令） |
| track_angular_velocity | `1-2*abs(c_wz-v_wz)/abs(c_wz)`（非零指令） |
| action_rate | `sum((a_t-a_(t-1))²)`，负权重惩罚 |
| joint_pose_deviation | `mean((q-q_snake_init)²)`，负权重惩罚 |
| site_tilt | `1-cos(θ)`，负权重惩罚 |
| joint_pos_limits / joint_target_limits | `sum(max(abs(q)-.9*L_joint,0))`，分别用于实际角和目标角 |
| failure_reset | 任意失败终止当步一次性扣5分，纯timeout不扣 |

前向/侧向/yaw 的权重、alive和其他惩罚均保留当前配置，日志逐项显示。
每步奖励由框架乘 `step_dt`，失败项仍以 `terminated/step_dt` 返回，得到每次固定惩罚。
这里的“零速必须扣分”针对非零指令的速度项；总奖励还包含alive及其他项，不保证总分一定为负。

实际每轴速度奖励还需乘对应权重和step_dt。最高原始分项为1；停车不提供正奖励。新分母是该轴指令幅值，尚需训练验证，不能单凭奖励规则认定已解决推进或探索问题。
检查点奖励标识更新为 `zbot_snake_head_frame_relative_clipped_v16`，旧v13/v14/v15检查点不会被当作相同配方加载。

## 指令采样

Site 蛇任务默认自动采样全零停车指令的概率为 **1%**，从此前的 12.5% 降低。
配置项为 `commands.transform.motion_stop_probability`，训练和随机 Play 共用；设为 0 可关闭自动停车采样，设为 1 则全部停车。
非停车样本仍按 3:4 的相对比例覆盖六种带符号单轴及混合指令，其他任务默认仍为 12.5%，保持既有采样行为。
手动 X 停车或 `stop()` 仍会锁存全零指令，不受后续自动重采样覆盖；reset 后平滑指令的初始零值不属于停车采样。

## 观测与动作

Actor/Critic 共享 **61 维**输入。复用原 56 维基础状态，删除 heading 误差槽，附加速度误差 3 维和蛇头局部重力 3 维。

| 切片（从 0 开始） | 内容 |
|---|---|
| `0:46` | 原有状态；task 在 39，生效速度指令在 `40:43` |
| `46:49` | 原有 COM 高度/竖直速度及归一化接触力 |
| `49:52` | site 当前 vx/vy/wz |
| `52:55` | 请求的 vx/vy/wz |
| `55:58` | 有符号误差 `(指令−实际)/各轴速度上限` |
| `58:61` | site 局部坐标系中的单位重力向量 |

原有通用任务的 56/59 维及归档 63 维诊断解码保留，本任务不再向策略提供 heading 角或 sin/cos。
六关节使用目标角速度积分动作，目标角限制 ±π/2，最大目标变化率 2π rad/s。
实际角和目标角在 ±81° 以外开始软限位扣分，±90° 目标硬截断。

## 终止、回放与检查点

训练因自碰撞、site 倾角超过 60°、或达到 30 秒 / 1500 控制步而重置。
恰好 60° 不重置，保留 1e-6 rad 数值容差；倾角相对 site 当前朝上轴和重力计算，不受世界 yaw 影响。
Play 仅在 site 倾角超过 60° 时自动重置；无时间限制。独立能力评测仍显式关闭自动重置以记录完整失败。

当前检查点契约：`schema_version=6`、`snake_omni_version=12`、Actor/Critic 输入 61 维，
`observation_recipe=site_head_frame_error_gravity_v2`、`reward_recipe=zbot_snake_head_frame_relative_clipped_v16`。
命令坐标系标记为 `site_minus_z_gravity_aligned / left / world_plus_z`。
检查点保存 `velocity_limits`，回放恢复相同误差尺度；旧固定 heading 的 63 维及其他奖励配方检查点不能直接作为本版兼容模型加载。

## 训练输出

保留逐项奖励和 rsl_rl 状态统计，包括 alive、速度误差、倾角、限位、失败 reset、平均回合长度、当前存活步数/秒数、reset/timeout 数量、采样速度、动作标准差和 ETA。
日志中的回合平均只统计已结束回合，当前存活统计与其分开；计数逐轮清零，满足 `resets=非timeout终止+timeouts`。
非零指令的速度项准确跟踪为1，50%速度为零，静止为负；alive 为健康步提供正分，其他惩罚单列。总分可能为正，不能只用总分判断是否跟踪成功。

## 验证

v12 历史验证：353 项相关测试、Ruff 和 Pyright 通过。CUDA eager/cuda_graph 两后端各完成 16 环境、144 步、3 次 PPO 更新，
以及保存加载后的 4 环境、16 步 Play 检查；61 维观测、奖励及网络参数均有限，特征与物理状态一致，局部重置隔离正常。
测试覆盖实时蛇头参考系、移除 heading 观测、固定边界误差、健康/alive、timeout 与失败扣分。
冒烟记录：`outputs/diagnostics/head_frame_alive_v12_smoke.json`。完整训练效果仍需新模型验证。

## 历史归档依据

以下范围和宽度用于此前版本的归档比较，不代表 v16 当前参数。


| 设置 | 新值 |
|---|---|
| 前后、侧向指令上限 | 每轴 ±0.2 m/s |
| yaw 指令上限 | ±0.5 rad/s |
| 纯平移非零指令幅值 | 0.08–0.2 m/s |
| 纯转向非零指令幅值 | 0.2–0.5 rad/s |
| 混合指令 | 各轴分别在上限内连续采样，可接近零 |
| 重采样间隔 | 5–9 秒 |
| 可选手动前进速度档位 | 0.08 / 0.10 / 0.12 / 0.15 / 0.20 m/s |
| 独立评测两档（平移，yaw） | (0.08 m/s, 0.2 rad/s)、(0.15 m/s, 0.5 rad/s) |

采样仍覆盖停止、六个有符号单轴及混合运动；不按 tracking EMA 缩小范围。
纯指令的最小幅值是对应上限的 40%；停止另行采样。手动指令也使用相同限幅。
前进档位沿用已有部署命令配置，当前特权训练采用连续采样，不按这些档位运行课程。

- [蛇模型说明](../bests/snake_world_x_2026-10-19_iter1499/README.md)、
  [独立回放](../bests/snake_world_x_2026-10-19_iter1499/verification.json)：
  原世界 X 推进奖励在 0.2 m/s 封顶，训练平均约 0.150 m/s，独立回放平均约 0.083 m/s。
  用它们选择 0.08 的纯指令下限和 0.2 的平移上限；这些旧数据是质心世界速度，
  不宣称新任务的 foot_0_sole site 已验证达到同样速度。
- [Zbot 全向行走配置](../bests/omni_velocity_energy_2026-10-10_iter499/env.yaml)：
  原指令范围 vx=[−0.3,0.5]、vy=[−0.2,0.2] m/s、yaw=[−0.5,0.5] rad/s；
  平移奖励 std=0.1 m/s，yaw std=0.25 rad/s。蛇任务不照搬更快的双足前进范围。
  侧向和 yaw 的范围据此作为新训练目标，不作为蛇已有能力结论。
- 归档蛇配置中的 ±0.25 m/s、±π/2 rad/s 当时对应 `sample_velocity_commands=false`，
  不能当作已验证的全向跟踪范围。已归档行走模型的纯 yaw 表现也较弱。
- 直接使用固定 0.1 m/s 的线速度核，低速指令下静止得分偏高；因此采用随指令增长、
  上限为 0.1 的宽度。0.08 m/s 指令时 std=0.06，0.2 m/s 时 std=0.1。
  指数里的 std 是误差容差，不是策略动作分布的标准差。

v5 历史验证：139 项相关测试、Ruff、Pyright 和 32 环境 CUDA 下 3 次 PPO 更新通过。
这些设置仍需完整训练验证，未宣称后退、侧移和转向已收敛。

前向和侧向项独立记录日志并独立调权；当前前向4、侧向1；当两项都设为4时，与原合并项等价。保留现有检查点契约。命令行示例（降低侧向代价）：

```bash
python train.py Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher \
  --env.rewards.track-forward-velocity.weight 4 \
  --env.rewards.track-lateral-velocity.weight 1
```

纯前进A/B对照及复现方式见 [实验说明](site-snake-forward-ablation.md)。

## 探索参数

蛇任务现采用本地 MJLab G1 velocity 的高斯探索设置：

- `GaussianDistribution`，`init_std=1.0`、`std_type=scalar`，各动作维度的标准差可学习。
- `entropy_coef=0.01`、`learning_rate=0.001`。
- Actor 输出六个动作均值，取代此前十二个 Beta 形状参数；其他网络层及观测保持原样。
- 高斯样本本身无界。PPO 保存原始采样及其高斯 log-prob，执行前通过 `clip_actions=1.0` 裁到 ±1；动作项仍有相同裁剪，随后进行原有目标角速度积分和 ±π/2 目标限幅。
- schema 6 / snake version 12 标明高斯分布及执行裁剪。旧 Beta 检查点不能直接恢复到这个策略，需启动新训练；前进 A/B 子任务也继承新的高斯设置。

熵正则系数提高不代表探索幅度按同一倍数提高，实际探索还取决于已学均值、标准差和裁剪饱和。新启动训练生效；本次未启动后台训练，也未验证完整训练收敛效果。

`action_rate` 使用框架 `common_mdp.action_rate_l2`，由 RewardManager 统一乘控制步长；恒定动作不扣此项。训练和Play均记录该项。命令行可用 `--env.rewards.action-rate.weight -0.01` 调权；纯前进子任务同步继承。

初始构型偏离项以蛇形重置关节角为参考，使用实际关节角，既不追踪动作目标，也不使用双足默认关节姿态；世界位置和yaw不参与。权重可用 `--env.rewards.joint-pose-deviation.weight -0.02` 独立调整，日志名称为 `joint_pose_deviation`。较大权重可能抑制推进所需弯曲，当前先使用小权重。

速度分项在乘权重和控制步长之前裁剪到[-2,1]；加权总奖励、alive、失败惩罚及其他成本不受这个范围约束。截断负分后，极端反向速度的额外代价被饱和，零净位移往返运动不再保证比静止更差，仍需通过净位移评测检查。
