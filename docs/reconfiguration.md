# 重构实验1：首尾接近与贴合

Basic于2026-10-26按用户选择回退到13:11独立建立时的版本，包括物理仿真、速度积分动作和59维观测。随后将`end_alignment`改为0～1的面对面法线分数，并增加原点相对接近速度与轴角误差减小速率奖励；用户最新把静态距离/法线权重均设为0.1。当前优先处理正确接近和贴合；磁吸、捕获和闭环机制暂不调整。不能把此次代码验证当成已经学会贴合。

| 版本 | Task ID | 参数卡 |
|---|---|---|
| Basic，标准单策略PPO | `Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Basic` | [Basic卡片](../src/zbot_rl_mjlab/experiments/cards/reconfiguration_head_tail_docking_basic.json) |
| PBT，保持回退前状态 | `Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-PBT` | [PBT卡片](../src/zbot_rl_mjlab/experiments/cards/reconfiguration_head_tail_docking_pbt.json) |

## 当前Basic

六关节从零角直链趴地开始，初始首尾site距离约0.66m。动作恢复为速度积分：

```
q_target = clamp(q_target + clamp(action, -1, 1) * pi * 0.02, -2, 2)
```

策略目标限位±2rad，模型机械限位±π。底层位置伺服kp=15、kd=1.5，力矩上限20N·m、armature=0.001。高斯策略初始标准差1，PPO训练采样动作；默认回放执行均值。当前不使用后续加入的0.25倍绝对位置动作、限位脱困辅助或25/81维观测。

物理恢复为5ms步长、4个子步，控制周期20ms；implicitfast积分器、Newton求解器、30次迭代、20次线搜索、50次CCD迭代、njmax=1000。地面摩擦为(0.6,0.005,0.0001)。这些参数与13:11保存的配置逐项核对；不沿用后续2.5ms/8子步调参。

默认8192环境、每回合10s（500个控制步）、每轮24步、500轮；γ=0.99、自适应学习率、每50轮保存。参数可在卡片调整，以实时卡片和运行的params快照为准。

### 奖励

`end_alignment`沿用原日志名称，但现在表示法线分数，不再表示完整姿态指数奖励。记两个sole的世界+Z为`Zh,Zt`，`u`为头site指向尾site的单位向量：

```
g = min(dot(Zh, Zt), -dot(Zh, u), -dot(Zt, u))
s = (1 + clamp(g, -1, 1)) / 2
```

头sole的+Z是内法线，尾sole的+Z是外法线，因此正确对接时两个+Z同向，且两端面朝向彼此。直链背离s=0；两轴同向但连线在侧面时s=0.5；正确面对面s=1。距离小于1微米时连线方向未定义，改用两条Z轴点积映射到[0,1]。不乘距离衰减，不要求方位角达标。

| 项 | 未加权值 | 当前权重 |
|---|---|---:|
| end_distance | exp(-距离/0.2m) | 0.1 |
| end_alignment | 上述面对面法线分数s，范围[0,1] | 0.1 |
| end_distance_velocity | 首尾原点沿连线的带符号接近速度，m/s | 1 |
| end_alignment_velocity | 带符号的轴角误差减小速率，rad/s | 0.5 |
| approach_velocity | exp(-距离/0.1m) × 相对线速度² | 0 |
| connected | 已连接为1 | 0 |
| success | 首次持续成功，一次性奖励 | 0 |
| action_rate | 动作变化平方和 | -0.002 |
| joint_velocity | 六关节实际速度平方均值 | 0 |
| limit_proximity | 实际角/积分目标取较小限位余量，缓冲区侵入比例平方的关节均值 | -2 |

用户纠正了前一版“得分变化率”的解释。当前速度项直接取真实site状态：

```
u = (p_tail - p_head) / ||p_tail - p_head||
v_close = -(v_tail - v_head) · u                         # m/s
omega_relative_head = R_head.T @ (omega_tail - omega_head)
w_align = -axis_error_head · omega_relative_head         # rad/s
```

轴角误差来自当前`R_head.T @ R_tail_dock`的最短旋转，沿用最近合法注册角补偿；轴用头site局部坐标表示。`w_align=-d(theta)/dt`，因此完整相对姿态误差缩小时为正。静态end_alignment仍为0～1面对面法线分数；轴角速度项评价完整相对姿态，包含其扭转分量。

按用户最新要求，只保留两个带符号速度项：原点靠近/轴角误差缩小时为正，分离/误差增大时为负，停住为0；两项速度由当前位姿、site线速度和角速度解析计算。沿连线的投影使侧向运动不获得接近速度奖励，投影到误差轴使无关转动不获得轴角对齐奖励。原点重合（距离<1微米）时接近方向不唯一，速率置0；误差接近0或π的奇异点也取中性0，避免任意轴/最短角分支造成方向跳变。

上一版的得分差分缓存已移除，速度项可以在首个有效状态直接计算；位置重置本身不会被差分成速度，重复调用不改变结果。新增诊断为`end_closing_speed_m_s`和`end_alignment_rate_rad_s`。静态权重保持用户当前的0.1/0.1，速度项权重分别为1/0.5；未改动59维观测、10秒回合、动作或物理。

用户最新要求加入接近限位惩罚、移除body_height，并把回合缩短到10秒；同时保留用户在卡片中将approach_velocity、joint_velocity、connected和success设为0的手动调整。实时权重以卡片为准，success权重为0不关闭成功终止条件。

`limit_proximity`使用与动作一致的有效限位（机械限位与策略目标限位的交集）。每关节同时检查实际角和积分目标，取距上下边界的最小余量c，代价为`mean(max(0, (margin-c)/margin)^2)`，margin=0.3rad。当前限位±2rad，因此|角度|≤1.7rad不扣分，进入缓冲区后平方增长；在边界时单关节代价1，六关节求平均后乘当前权重-2（用户后续从-0.5手动调整）。若只有一个关节在边界，扣分率为-2/6≈-0.3333；六个都在边界为-2。实际角超出目标边界时继续增长，不截断。

这是每步持续的非负状态代价×负权重：实际角和目标都回到安全区立即归零，不发离开奖励、不设计时/滞回、同一关节不重复累计两种代价。平方函数在缓冲区入口的斜率为0；各关节求平均，使一个关节退出就能降低代价。速度积分的action不当作角度惩罚。当前几何、物理、动作和59维观测保持原样。参数只提供软约束，尚未验证能否消除策略的限位停留或往返。

当前重点看`Episode_Metrics/end_distance_m`和`Episode_Metrics/end_normal_score`。`Episode_Reward/end_alignment`是乘当前0.1权重、积分并按最大回合时长归一化的统计，不是未加权的s；`end_angle_rad`仍记录供原捕获逻辑使用的完整姿态角差。

此前17:59:23运行第175轮的`end_alignment`约3.42e-6，显示四位小数为0.0000；当时仍使用旧公式`exp(-d/0.1-(angle/0.3)^2)`，平均距离约9.15cm、完整角差约125°。这是旧指数奖励在大角差时极小，不是本次0～1法线计算的结果。运行中的进程不会热加载，需重新启动训练才能采用新奖励。

### 59维观测

| 观测组 | 维数 |
|---|---:|
| 关节角、关节速度、积分目标、上一动作 | 24 |
| 头site局部首尾位移、最近合法注册角补偿后的相对四元数 | 7 |
| 头site局部首尾相对线速度、角速度 | 6 |
| 两端site局部重力方向 | 6 |
| 连接、捕获进度、保持进度、约束力/力矩、重捕获冷却 | 6 |
| 最近合法注册角one-hot | 4 |
| 头site局部实际线速度、角速度 | 6 |

Actor和critic均59维，保留原运行均值/方差归一化。新法线分数仅用作奖励和诊断，没有改动观测维度。

### 暂时保留的初版磁吸机制

Basic自动选择最近合法注册角；没有`dock`方位指令，也不使用`--env.commands.dock.target-yaw-deg`。相邻模块固定连接，首尾预编译4条inactive site-weld，满足条件时激活一条；不重编译、不瞬移端面。

初版磁力范围4cm、上限8N，径向阻尼0.4，完整姿态恢复力矩系数0.15、阻尼0.02。捕获门限为4mm、完整角差8°、相对速度0.1m/s、角速度1rad/s，持续0.03s。已连接且距离≤8mm、完整角差≤16°持续0.5s才计success；断开阈值80N或4N·m，冷却0.15s。磁力参数仍是未标定仿真假设。上述机制本轮不调整，法线高分本身不等于已连接或满足原success条件。

## 启动

`train.sh`、`run.sh`当前启用24.a Basic，使用独立日志根目录，保留所有旧日志和检查点：

```bash
./train.sh
./run.sh
```

对应日志：`logs/rsl_rl_reconfiguration_initial/zbot_reconfiguration_head_tail_docking_basic/`。回放可明确选择模型：

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Basic \
  --log-root logs/rsl_rl_reconfiguration_initial \
  --env.scene.num-envs 1 --checkpoint-file /path/to/model.pt
```

加`--stochastic True`可查看策略采样。25/81维旧检查点会在回放前被拒绝；相同59维也不保证动作、奖励语义相同，应检查模型的params配置。原生viewer仍显示端面间距、完整角差及原连接状态；Enter复位、空格暂停。

## PBT与恢复依据

PBT实现与卡片保持回退前状态：8×1024世界探索，有有效分化后4×2048；独立策略与优化器。PBT仍使用0.25倍高斯位置动作、59维观测和指定方位的对接任务；不能把Basic速度积分与自动最近方位的语义套用到PBT。

初版依据为`logs/rsl_rl/zbot_reconfiguration_head_tail_docking_basic/2026-10-26_13-11-06/params/`和找回的创建源码。该运行4环境、4步、1轮是当时的冒烟覆盖，标准默认训练配置仍为8192环境、24步、500轮。

回退前全部相关代码、配置、文档和测试保存在[备份目录](../outputs/reconfiguration/before_initial_reset_2026-10-26_17-51-06/)，含哈希清单和找回的初版源码。旧文档保存在该目录的`docs/reconfiguration.md`；历史决策继续保留在[项目记忆](project-memory.md)，已由本次回退及后续明确修改覆盖。

Basic实现位于`tasks/reconfiguration/basic/`，与PBT隔离。验证入口为`tests/test_reconfiguration*.py`，包括保存物理配置比对、真实GPU速度积分/59维/PPO保存加载、法线方向与旋转不变性，以及既有磁吸和PBT回归。短测试不证明策略已学会正确接近或贴合。
