# 蛇形前向推进

任务注册名仍为 `Mjlab-Zbot-6dof-Snake-Omni-Teacher`，日志目录为 `logs/rsl_rl/zbot_snake_omni_teacher/`。当前训练不跟踪速度指令，只学习沿世界 +X 推进。

## 世界X推进与复位扣分

取所有刚体质量加权的质心速度，并按头端 `foot_0_sole` 的水平坐标系投影。前向是 site 局部 -Z 在水平面的单位投影，左向与世界向上构成右手系，得到 `v=(vx,vy,vz)`。site Z接近竖直、水平前向退化时沿用最近有效方向。前进奖励使用质心世界X速度，不随蛇头朝向旋转；侧向成本使用世界Y速度，动态惩罚倍率使用世界X速度，不使用头尾连线；没有身体展开比或卷曲门控。关节初始构型为 `q0`。

```text
forward = min(v_world_x, 0.2)
S = min(1, 5*max(v_world_x, 0))
reward_rate = 0.02*alive + 15*forward - 5.0*S*abs(v_world_y) - 0.2*abs(vz) - 0.02*mean((q-q0)²) - 0.1*target_saturation_fraction - 0.2*S*sum(omega_head²) - 0.01*sum((action - prev_action)²) - 0.5*(yaw_error/(π/3))² - 0.1*(body_yaw_error/(π/6))² - 0.5*tanh(I_body/(π/6*1s))² - 0.5*tanh(I_head/(π/3*1s))²
step_reward = reward_rate * step_dt - 5*head_tilt_terminated - 5*self_collision_terminated - 5*(head_yaw_terminated or body_yaw_terminated)
```

| 配置项 | 原始值 | 权重 |
|---|---|---:|
| alive | 本步未发生失败终止为1，否则为0 | +0.02 |
| action_rate | 六维原始动作相邻控制步差值的平方和 | -0.01 |
| head_forward_speed | 整机质心世界X速度的线性奖励，0.2 m/s封顶 | +15 |
| head_lateral_speed_cost | 整机质心世界Y的 abs(v_world_y) × S | -5.0 |
| head_vertical_speed_cost | 蛇形上下 abs(vz) | -0.2 |
| joint_pose_deviation | 相对初始关节角的均方偏差 | -0.02 |
| target_saturation_fraction | 目标角距 ±60° 边界不超过 0.001 rad 的关节比例 | -0.1 |
| head_angular_velocity_cost | 蛇头三轴角速度平方和 × S，单位 (rad/s)² | -0.2 |
| self_collision_reset_cost | 自碰撞终止标记 / step_dt | -5.0 |
| head_yaw_integral_cost | 有符号头部yaw积分，除以60°·1秒后取tanh平方 | -0.5 |
| head_yaw_reset_cost | yaw越界终止标记 / step_dt | -5.0 |
| body_yaw_integral_cost | 有符号体轴yaw积分，除以30°·1秒后取tanh平方 | -0.5 |
| body_yaw_cost | 尾→头site连线相对初始yaw的最短偏差 / 30°，再平方 | -0.1 |
| body_yaw_reset_cost | 体轴越界且本步头部yaw未越界的标记 / step_dt | -5.0 |
| head_yaw_cost | 相对本回合初始朝向的最短 yaw 偏差 / 60°，再平方 | -0.5 |
| head_tilt_reset_cost | 头倾终止标记 / step_dt | -5.0 |

alive 使用 MJLab 原生 `is_alive`：`~termination_manager.terminated`。正常存活每秒 +0.02，由奖励管理器按 `step_dt` 缩放；yaw越界、头倾、自碰撞或unsafe的失败终止帧不给。正常超时属于truncation，不算失败，最后一步仍可获得该项。环境reset后恢复正常判据，不累计额外奖励。alive也会奖励安全静止，因此这里只作小额生存辅助，最大前进奖励率为 +3/秒。

奖励项保留 `head_` 前缀；前进项 `head_forward_speed` 使用整机质心世界X速度，侧向项使用世界Y速度，竖向项使用世界Z速度。数值均取整机质心速度。前进奖励采用 `min(v, 0.2)`：0.2 m/s以下按速度线性计分，达到后保持0.2，不再增加，也不衰减；静止为0，倒退保留线性负分，不做负向截断。上限可用 `forward_speed_cap` 调整。左右/上下使用绝对速度成本，正负方向对称，当前基础权重分别为 -5.0、-0.2。构型惩罚权重为 -0.02，q0为六关节零角。目标顶限惩罚按六关节比例持续计费：一个关节顶限的奖励率约为 -0.01667，六个均顶限为 -0.1；离开边界即停止该项扣分。

左右速度和头部三轴角速度惩罚使用动态倍率 `S=min(1,5*max(vx,0))`。这里vx是未截断的实际质心世界X速度；静止或倒退倍率0，0.1 m/s为0.5，0.2 m/s及以上为1。左右速度的有效权重从0增加到-5.0，角速度有效权重从0增加到-0.2。每个环境独立计算，不修改共享RewardTermCfg权重。

倍率下限由 `motion_penalty_min_scale=0` 配置，速度增益为 `motion_penalty_speed_gain=5`；倍率上限固定1。`head_angular_speed_rad_s` 仍是原始物理角速度模长，不含动态倍率。瞬时yaw、yaw积分、竖向速度、action_rate及各复位判据不随该倍率改变。

头倾达到60°时，终止帧额外一次性扣5。奖励函数读取本步终止管理器的head_tilt标记，并除以step_dt抵消奖励管理器的dt缩放，因此实际事件费用恒为-5；自碰撞终止同样一次性扣5；yaw越界终止同样一次性扣5；头部和体轴yaw同时越界只计一次航向费用；航向、头倾和自碰撞之间仍独立计费。复位后下一步不重复扣，超时和unsafe不会触发这两项事件费用。没有恢复连续头倾姿态成本，也没有额外的速度/yaw跟踪、能耗或接触冲击成本。使用整机质心而非蛇头速度，避免内部摆动冒充推进。

## 保留的初态与复位

- 初态从 XML 直立立柱绕 foot_0 局部 Y 轴旋转 -90°，四元数为 `(√0.5,0,-√0.5,0)`。foot_0.Z 指向世界 -X，foot_0.X 指向世界 +Z；初始高度 0.0555 m，关节零角。
- 训练保留全周随机初始 yaw，play 使用默认航向。每次 reset 都重建上述基础构型，并从实际写入的根四元数记录该环境本回合的初始水平 yaw。部分 reset 只更新选中的环境。
- 头部 yaw 相对该初始朝向的最短角度差超过 ±60°时终止复位（恰好60°允许，带1e-6 rad浮点容差）。用 atan2(sin(delta),cos(delta)) 处理 ±180°跨界；不把随机初始朝向当作偏差。训练和手动回放均执行该边界。30°偏差的成本率为 -0.125，60°为 -0.5；yaw越界终止另计一次性 -5，与头倾、自碰撞一致。
- 新增独立 `body_yaw` 约束：尾site `foot_1_sole → foot_0_sole` 的世界水平连线朝向，相对同一个本回合初始yaw计算最短角度差。不是相对当前头部yaw，也不替代头部yaw约束。超过±30°立即复位，恰好30°允许。头尾水平距离≤0.1 mm或非有限时朝向无定义，也触发该终止；这种退化状态使用π的有限误差计成本。
- 体轴偏差成本为 `-0.1*(body_yaw_error/(π/6))²`；体轴单独越界的一次性费用为-5。若头部yaw同时越界，由已有head_yaw_reset_cost收费，body_yaw_reset_cost为0，避免一次航向失败重复收费。此机制不门控前进奖励，不恢复此前删除的展开比机制。
- 新增整回合有符号积分 `I += body_yaw_error * step_dt`，单位rad·s。左右等时等幅偏差可相互抵消，持续偏向同一侧会累积。费用为 `-0.5*tanh(I/(body_yaw_limit*body_yaw_integral_time_scale))²`，归一化时间默认1秒；仅费用有界，积分本身无泄漏、不截断。这样不会因回合变长而产生无限惩罚。
- 独立增加头部yaw积分 `I_head += head_yaw_error * step_dt`，同样相对本回合初始yaw。费用为 `-0.5*tanh(I_head/(head_yaw_limit*head_yaw_integral_time_scale))²`，时间尺度默认1秒、头部阈值60°。左右偏差可抵消，持续单侧偏转会累积；与体轴积分分别存储，均在回合reset时清零。
- 两项积分按 `common_step_counter` 每个控制步最多更新一次，奖励、指标、Actor和Critic重复读取不重复累加。reset仅清零选中环境，reset当步的观测不重新积分；无效连线帧不注入任意方向的积分，由现有终止条件处理。
- 蛇头相对世界向上的总倾角达到 60°立即复位；头部向上基准为局部 +X。绕世界重力轴的转动不单独触发头倾复位。
- 非有限状态、离开 15 m 工作区、极端接触冲击、真实自碰撞及 30 秒回合时限的复位设计保留。

动作恢复为六关节目标角速度积分：

```text
q_target = clip(q_target + clip(action, -1, 1) * 2π * dt, -π/3, π/3)
```

`action=±1` 对应 ±360°/s 的目标角速度，`action=±0.5` 对应 ±180°/s；控制步长0.02秒时单位动作每步改变目标角7.2°。`action=0` 保持当前目标角，不会回零。积分目标限制在 ±60°，裁剪后的目标直接作为下一步积分起点，反向动作能立即离开边界。沿用原 PD、机械限位和 reset 逻辑。


增加与 G1 一致的 `action_rate` 成本：`sum((action_t - action_{t-1})²)`，权重 -0.01。它基于缩放前的原始六维动作，不除以 dt；奖励管理器再乘控制步长。恒定动作不持续扣分，快速往返目标比平滑周期动作扣分更多。历史由原生 ActionManager 维护，环境 reset 时清零。

目标角仍与物理关节范围取交集，策略使用原生Beta默认初始化。

## 观测、命令和兼容性

Actor 为 51 维，Critic 为 66 维。保留原 40 维顺序：`24:27` 为世界 +X 在 foot_0 局部坐标中的表示，`33:36` 为头部局部三轴角速度（rad/s）。`40:43` 为头部原点在 foot_0 局部坐标中的三轴线速度（m/s），由世界线速度经过逆四元数旋转得到；原世界竖直速度槽位也保留。线速度和角速度均直接来自仿真状态，教师观测不加噪声。推进奖励使用整机质心世界X速度。追加 `43:47` 为头部世界姿态四元数 WXYZ（归一化且w非负），`47:49` 为相对初始yaw偏差的 sin/cos，使初始朝向约束直接可观测。`49:50` 为体轴归一化有符号积分 `I_body/(30°·1秒)`，新增 `50:51` 为头部归一化有符号积分 `I_head/(60°·1秒)`，提供历史状态，策略能判断累计偏向及需要恢复的方向。

新任务关闭随机速度指令采样、速度课程和指令分组指标。命令项仍用于相位频率、旧模型回放和 reset 的训练标记，不再给策略随机运动目标。不要根据旧的速度跟踪 std 或课程状态判断当前任务。

检查点元数据为第 22 版，记录 `objective=snake_world_x_capped_speed`、速度积分动作语义、最大目标角速度、目标角上限、前进奖励速度上限、初始yaw参考、头部与体轴阈值、尾site、头部姿态观测以及头部参考 site 等、坐标与分布信息。观测维度已变化，旧检查点不能直接续训；请从零训练。当前实验不再兼容旧检查点，回放和续训均使用第 22 版新模型。

策略保留 RSL 原生 Beta 分布、默认MLP初始化和完整 [-1,1] 动作范围。没有额外的初始均值、标准差或alpha/beta偏置覆盖；探索参数由策略网络输出并由PPO学习。默认初始化的动作标准差通常约0.47，具体取决于随机初始化与观测。加载检查点时恢复已学习的策略参数。

## 显示与运行

终端保留原生训练概览，完整列出十七项 `Episode_Reward/*`，追加世界 XYZ 速度、关节速度/目标顶限、头倾与复位计数摘要。保留19项关键诊断（含动态运动惩罚倍率、头部和体轴yaw积分绝对值），完整结果写TensorBoard。回合reward是框架归一化统计，不是单步速度数值；单次-5费用在30秒回合归一化日志中显示为约-0.16667，实际终止帧仍扣-5。

蓝色箭头表示蛇头水平前向，绿色箭头表示实际整机质心速度，起点均为 foot_0_sole site 原点。当前蛇形前向任务不使用 W/S/A/D/Q/E 目标速度控制；Enter 复位，空格暂停，R 切换可视化，反斜杠切换相机跟随。旧指令模型回放保留其原有速度键。

新任务从零训练，例如：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 3000 \
  --agent.run-name snake_forward_v22

./scripts/python.sh play.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
  --env.scene.num-envs 1 \
  --checkpoint-file 实际蛇形前向模型路径
```

不自动修改 train.sh/run.sh 或热更新正在运行的训练。原有 `scripts/evaluate_snake.py` 的八种速度指令场景针对旧指令任务，不作为当前蛇形前向任务的能力验收。

## 审视与物理对照证据

`outputs/snake-audit/` 保存了物理、奖励和性能对照结果及复现脚本。固定周期关节波已经能产生持续推进，说明机械模型不是完全无法移动。旧速度跟踪奖励曾给静止约 7.63、有效推进步态约 1.25，存在明显的反向激励。

历史平方Y/Z成本对照中，一条稳定步态10秒沿+X位移约0.397m，平均奖励+0.03281；这些分数不适用于当前配方。等权绝对值探针中，一些有效步态仍低于静止得分，随后训练平均回合缩短到约1.2秒，主要由头倾终止。零动作10秒不翻，而旧策略约1.78秒越过阈值，简单缩小动作反而更早失败，因此没有盲目降低动作速率或放宽reset。

`reset_diagnosis.json` 对旧策略的回放反事实比较显示：在首次翻倒之前，旧回报约+0.148；采用YZ权重-0.2和一次性头倾费-0.5后，回报约-0.200（gamma=.99折扣后约-0.030）。这验证了当前旧策略“冲一下就翻”的收益被扭转，不等于新策略已经学会稳定运动。

验收应看确定性策略在后半段是否保持沿世界+X推进和关节周期运动，同时统计 Y/Z 漂移及真实复位，不能只看总奖励上升。低速、有限候选的对照结果不是最终学习性能保证。

## 改变关节范围后迁移续训

保持51/66维观测、速度积分控制和其他检查点语义一致时，可显式指定 `--env.allow-resume-joint-limit-change True`，允许完整恢复旧模型后使用新的关节目标范围。该开关仅允许 `joint_target_limit` 差异，不放行其他动作、观测或坐标语义变化；纯Actor回放允许关节目标范围与最大目标角速度不同，使用当前回放配置。保存的新检查点使用新范围，并在 `infos.snake_resume_joint_limit_change` 记录来源与旧／新范围。

从±60°模型改用±180°继续1500轮：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
  --agent.resume True \
  --agent.load-run 2026-10-19_16-45-28_beta_body_v3 \
  --agent.load-checkpoint model_1499.pt \
  --env.allow-resume-joint-limit-change True \
  --env.actions.joint-pos.target-limit 3.141592653589793 \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 1500 \
  --agent.run-name beta_body_v3
```

这会恢复Actor、Critic、优化器和迭代计数，并保留当前实验配置；不会覆盖原训练目录或`bests/`归档。零动作保持当前目标角、积分抗饱和及物理关节限位规则保持不变。


## 渐进折叠惩罚

保留±180°控制目标、速度积分和原有reset条件，新增 `joint_fold_cost`，默认权重-1。对每个关节，取实际角和目标角相对初始角的偏差绝对值中的较大值，再取六关节中的最大值 `a`：

```text
fold = clamp((a - 120°) / (180° - 120°), 0, 1)²
fold_reward_rate = -1 * fold
```

120°以内无新增成本，135°为-0.0625/秒，150°为-0.25/秒，180°及以上为-1/秒。正负方向对称；一个关节折叠就完整计费，不按六关节平均稀释，也不因多个关节折叠而无限叠加。仅修改目标角不能掩盖实际关节仍然折叠；实际关节未到位时，折叠目标也会提前受罚。恢复到120°以内后停止该项成本，不累计历史。

阈值在 `rewards.joint_fold_cost.params.soft_angle` 和 `full_angle` 中设置，单位rad；权重在该奖励项的 `weight` 中设置。原有构型偏离和顶限成本继续保留，其他权重不变。无需新增观测或修改检查点维度，可以继续训练现有51/66维模型；不会自动修改已经运行的训练实例，也不保证仅改变奖励就能使已收敛策略脱离局部最优。
