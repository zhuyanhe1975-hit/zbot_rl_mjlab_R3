# 蛇足双向变形：独立的指令实验

> 本文包含按开发阶段追加的实验记录。不同阶段的观测维数、奖励和验证结果仅适用于对应实验；当前生效参数以源码及实际训练配置为准。文中 `outputs/` 检查产物已移至项目外备份，路径用于追溯历史，不保证在当前工作目录存在。

任务：`Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional`。
实验入口在 `src/zbot_rl_mjlab/experiments/bidirectional.py`，实现位于
`tasks/transformation/commanded.py`。双向实验与单向教师是 `TeacherTrainingExperiment` 的兄弟，共用镜像训练基础，各自安装行为与观测；不修改已归档模型。统一构建与状态时序见[架构说明](task-experiment-architecture.md)。

默认从零训练：`pretrained_checkpoint=None`、`resume=False`，Actor、Critic、优化器和训练计数重新初始化。
复用镜像动作、传感器和物理配置，不加载原站立或爬下模型。尚未运行正式PPO训练。

## 指令与初态

- 指令 +1：蛇变足，目标为既有双足站姿。
- 指令 −1：足变蛇，目标为六关节归零并使整个机体贴近地面。
- 多环境reset按当前实际指令补足待练方向，默认目标为各约50%；比例用 `--env.initial-standing-fraction` 调整。
  每批reset在待补方向的名额内随机分配，防止短时站立失败逐渐挤掉站立训练数据；自动切换后瞬时占比会随动作耗时变化，单环境仍逐回合随机采样两类初态。
- 蛇形初态沿蛇身纵轴随机侧滚±180°（覆盖完整360°），自动调整高度避免穿地，第一条指令为站立。
- 双足初态使用目标关节角、root单位四元数和按碰撞几何计算的地面净空，第一条指令为爬下。
- 默认`source`模式在当前目标附近连续稳定2 s后，原地自动切换相反指令：站稳→爬下→趴稳→站起。切换使用独立计时器，关节最大误差容差默认0.15 rad，其余站稳/静止判据沿用原定义；严格成功仍按0.10 rad连续稳定计时，二者互不替代。时长沿用`success_hold_s`。切换保留root姿态、关节、速度、执行器目标和回合计数，只清除目标相关的计时/失败状态。
- 训练回合总上限仍为16 s，失败/超时仍reset；一个回合可以完成多次双向变形。显式固定`stand`/`snake`模式不自动反向。

策略网络 37 维、价值网络 45 维：分别在原教师36/44列后追加一维指令。
原关节误差观测仍以固定站立构型为参考；策略结合方向指令推断当前目标。
策略网络不读取成功/失败历史。仍为3维Beta动作，对应6关节镜像目标，目标速度上限2π rad/s。
新模型输入维度不同，不能直接播放原 36 维教师检查点；原归档有原任务入口。

## 奖励与终止

五项基础奖励与两项增量按目标路由：

| 项目 | 站立指令 | 爬下指令 | 权重 |
|---|---|---|---:|
| `joint_standing` | 双足关节误差 + base相对重力朝向误差 | 零关节构型误差（不加朝向成本） | 12 |
| `base_height_progress` | base上升到站立高度 | base降低到蛇形高度 | 1 |
| `base_ascent_velocity` | 关节、抬升和朝向共同控制衰减，最高base≥0.23m后关闭 | 关闭 | 8 |
| `sole_ground_alignment` | −1～1，近水平陡峭曲线 | 关闭 | 1 |
| `reset_failure` | 失败惩罚 | 失败惩罚 | −10 |
| `stand_finish_joint_speed` | base高度≥0.23 m时，六关节实际速度平方均值 | 关闭 | −0.02 |
| `descent_ground_impact` | 关闭 | 超过两倍整机重量的接地力峰值成本 | −2 |

爬下由构型归零和低高度引导，不奖励反复下降，避免再次通过运动刷分。
站立时保留跌倒及早期base碰地判定；爬下时屏蔽这两项，因为base主动接地是目标的一部分。
两方向都保留连续静止超过1 s且尚未接近当前目标的reset，指令切换后重新计初始宽限。
成功判据分别复用原严格双足站稳和低位零关节判据，保持2 s成功计时，切换指令后清零。
此实验尚未验证受控下降的冲击大小；新策略的双向成功率需训练后评估。

## 训练与手动播放

在仓库根目录执行（由用户训练）：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 2000
```

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
  --checkpoint-file logs/rsl_rl/zbot_transform_snake_biped_bidirectional/<run>/model_<N>.pt \
  --env.scene.num-envs 1
```

原生可视化窗口中 **`]` → 站立，`[` → 爬下**（无需按Fn或Shift）。先点击MuJoCo窗口使其获得键盘焦点。
按键作用于全部播放环境，只改变观测中的指令，不重置姿态或执行器目标。
首次按变形控制键后进入手动控制，屏蔽超时、跌倒、早期base碰地和停滞导致的自动reset，保持当前指令直到再次按键。
未按键时，默认在目标附近连续稳定2 s后自动反向，并在16 s回合结束或失败后重新采样初态。自动切换不要求达到用于统计的严格成功，切换不重置物理状态。
只有用户主动reset（原生可视化窗口的Enter键）才重置环境，并恢复自动演示；手动模式下跌倒也保持当前物理状态继续执行指令。
手动控制不改变训练的16 s超时、失败判定和奖励定义。
固定测试单个方向可加 `--env.commands.transform.mode stand` 或 `snake`，同时将初态设为对应源姿态。
默认`source`模式保留训练初态/第一条指令配对，并在成功后自动反向；play每次启动随机种子，显式`--env.seed 42`可复现。
可加`--env.commands.transform.auto-switch-on-success False`恢复整回合保持目标，稳定时长通过`--env.success-hold-s`配置，切换专用关节容差通过`--env.commands.transform.auto-switch-joint-tolerance`配置。`success`指标记录回合内是否曾严格成功，指令管理器另记录每环境每回合的`completed_transitions`，用于观察有效变形次数。
键盘控制用于默认原生可视化窗口，未实现Viser的指令按钮。

无PPO更新的接口验证脚本：`outputs/autonomous/bidirectional_command_probe.py`。
它验证随机双初态、对应指令、37/45维观测、3维动作、推理有限值以及切换不瞬移；不声称随机策略已学会双向动作。

## 修复方向样本饥饿

逐回合独立50/50的旧采样会产生严重时间偏差：model_100回放站立平均0.119s就终止，
爬下持续16s，站立交互步仅约0.7%。reset按当前实际指令的方向名额补足，避免失败环境逐渐全部转入长回合方向。
统计使用当前指令，而非已因自动反向而过时的初态标记。新采样只在reset时纠偏，不阻止稳定后的即时反向，因此不保证瞬时50/50。
`stand_env_fraction`报告实际指令环境占比；原`transform_command`为回合末指令，不能单独用于判断交互步是否均衡。
未修改原鲁棒实验或放宽双向实验的跌倒判据。需要重启训练以使用新采样；已偏向爬下的旧模型不会仅因修改play采样就自动学会站立。


## 初始滚动与早期跌倒判据

只对双向实验调整早期base碰地reset：先要求本次站立指令下base历史最高高度≥0.05m，
再要求从该峰值回落≥0.01m，并有真实base接地及再次触地/向下冲击，才记录早期跌倒。
蛇形侧滚初态的base原点高度约0.0526m，小幅滚动、短暂离地再触地或几毫米下沉都不触发此项。
阈值通过`--env.early-contact-arm-height`和`--env.early-contact-drop-threshold`调整。
原“站起后跌落”保护的启用高度也从18cm降为5cm；为防止初态即reset，
必须从历史峰值回落至少1cm，且高度低于10cm持续0.1s，才触发该保护。
早期再次触地/冲击项不需要等待这0.1s。静止reset和五项奖励均保留。
手动play仍屏蔽自动reset。原鲁棒实验保留其原判据，未改动。
这是训练环境判据变化，需要重启训练进程，可从已有双向检查点继续训练；不要求从零重训。

## 防止躺着摆双足构型

站立指令的现有`joint_standing`改为`−(joint_cost + s * gravity_cost)`：
`joint_cost`仍是六关节的平均Smooth L1误差；`gravity_cost = 1−dot(g_base, g_target)`，
比较当前base坐标系中的完整单位重力方向与目标站姿FK参考，包含全部三个分量，世界偏航不影响分数。
`s=standing_pose_gravity_scale`默认1，可设为非负数。关节全对时，直立成本0，侧躺90°成本1，
翻倒180°成本2；全局权重12不变，即侧躺构型项为−12。爬下指令不受这项朝向成本约束。

现有速度项的衰减也不再仅看关节角。令`pose_ready`是原0.2～0.8rad最大关节误差门控的反量，
`lift_ready=clip((base_height−snake_height)/(standing_height−snake_height),0,1)`，
`upright_ready=clip(1−gravity_cost/2,0,1)`，速度项乘`1−pose_ready*lift_ready*upright_ready`。
因此关节摆对但身体仍侧躺、未抬升时，速度项不会被直接清零；接近真实站立时平滑衰减。
原历史最高base高度≥0.23m后永久关闭速度项的规则优先保留，下降或再偏离构型不能重新开启。

该朝向修复保留五项基础奖励；下述两项增量独立增加。观测仍37/45维，动作仍3维。初态采样、跌倒门槛和手动play不自动reset的规则不变。
更新定义需要重启训练进程，可从已有双向检查点续训；代码验证不等于已验证新策略的学习成功率。
静态FK分数对照在`outputs/autonomous/bidirectional_lying_pose_reward_audit.json`，修改前在同目录的`bidirectional_lying_pose_reward_before.json`。

## 最小奖励增量：收尾关节速度与趴下冲击

仅增加两项成本，沿用原五项奖励及动作映射：

- `stand_finish_joint_speed`：站立指令下，当前base高度达到现有`ascent_reward_stop_height`（默认0.23 m）时，成本为`mean(joint_vel²)`，权重−0.02（由−0.1减弱80%，给最后伸展和纠偏留出余量）。使用实际关节速度；回落到门槛以下或切换为趴下后关闭。此项不会改变原上升奖励，也不惩罚base水平速度或角速度。
- `descent_ground_impact`：仅趴下指令生效。机器人全部刚体对地使用逐刚体合力传感器，保留一个控制周期内全部物理子步，取`F_peak`。成本为`max(F_peak/(机器人总质量×重力加速度)−2,0)²`，权重−2，正常静态支撑不受惩罚。阈值通过该奖励的`max_body_weights`参数配置，默认2；这是训练初值，不是硬件承载上限。

指标`stand_finish_joint_speed`记录末段关节速度成本，`ground_force_peak_n`记录全方向的峰值接地力（N）。没有额外的下降限速、末段base余速或角速度惩罚。

已有检查点保持37/45维观测和3维动作兼容性；新增奖励需重启训练进程并微调后才会改变行为。可通过`--env.pretrained-checkpoint <已验证的双向检查点>`热启动策略网络／价值网络。验证应比较同初态的站起成功率、收尾关节速度、趴下耗时和接地力峰值；短时仿真验证仅确认奖励计算与加载正常，不代表已学会改善后的动作。
