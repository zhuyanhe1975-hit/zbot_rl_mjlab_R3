# 变形实验：双足与蛇形

> 本文包含按开发阶段追加的实验记录。不同阶段的观测维数、奖励和验证结果仅适用于对应实验；当前生效参数以源码及实际训练配置为准。文中 `outputs/` 检查产物已移至项目外备份，路径用于追溯历史，不保证在当前工作目录存在。

当前结构：实验配方在 `experiments/transformation.py`；环境、状态和回调在 `tasks/transformation/`，共用姿态在 `poses.py`，独立／镜像动作在 `actions.py`。热奖励控制与两个训练运行器统一在根包 `rl.py`。构建与继承约定见[架构说明](task-experiment-architecture.md)。

任务 ID：`Mjlab-Zbot-6dof-Transform-Biped-To-Snake`。

保留6维独立关节目标速度动作；不做强制左右映射。策略自行决定六关节如何接近零角度，
目标速率上限1rad/s，目标角遵循模型限位（当前±6.28rad），没有参考轨迹或阶段时钟。

## 两个核心奖励，加一个对称正则

| 项目 | 公式 | 权重 |
|---|---|---|
| 目标构型 | exp(-mean(q²))，六个关节角均以零为目标 | +3 |
| 全身质心下降速度 | clamp(1-((vz+0.03)/0.03)²,-4,1) | +1 |
| 关节对称正则 | mean((q[:3]+reverse(q[3:]))²) | -0.5 |

质心按所有刚体质量加权，速度使用各刚体质心的世界竖直线速度，不用base或foot_0代替。
速度项在静止时为零、以0.03m/s缓慢下降时为1，上升或下降过快为负值。
对称项鼓励q1≈-q6、q2≈-q5、q3≈-q4，但六个动作保持独立。

已移除三组接触传感器、接触历史、支撑包围盒、复杂足端几何、展开/贴地乘积奖励、
冲击/能耗/滑移/关节速度成本，以及额外成功/失败奖励。物理碰撞与身体摩擦照常计算。

## 成功与终止

成功只作为统计，不提前结束回合，避免策略为了保留持续收益而回避完成目标。
成功条件：各关节绝对角度<0.1rad、最高模块质心<0.1m、关节RMS速度<0.1rad/s、
base线速度<0.02m/s，连续保持2s。
安全终止仅保留非有限状态、全身质心下降超过0.3m/s、越出1.5m区域；回合最长16s。
评估在回合终点检查持续保持情况，达标的超时回合记为成功。

策略网络和价值网络均为30维：关节角6、关节速度6、命令目标6、上次动作6、质心高度/速度2、
base投影重力3、保持时间1。动作仍为6维，但观测已大幅精简，旧变形检查点不能直接恢复，
需新训练。旧行走任务配置与检查点保持不变。

初始探索标准差0.5、范围0.1–1.0、熵系数0.005，避免此前标准差持续增大。
物理步长0.005s、控制频率50Hz，机器人资产和执行器沿用原定义。

## 运行

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Biped-To-Snake \
  --env.scene.num-envs 8192 --agent.max-iterations 1500

./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Biped-To-Snake \
  --checkpoint-file <new-checkpoint.pt>

./scripts/python.sh scripts/transformation_eval.py --num-envs 4 --checkpoint <new-checkpoint.pt>
```

零动作保持当前站立目标。评估记录质心高度/速度、关节误差、保持时间和终止原因。
仿真限位/限矩不代表硬件标定范围；短PPO测试只验证训练接口，不证明策略收敛或平稳落地。

## 本次验证

全套148项测试、Ruff、Pyright、编译检查和独立审查通过。真实CUDA环境确认6维动作、
30维策略网络／价值网络；两轮PPO更新及检查点加载通过，短训尚未达到变形目标。
同机8192环境、24步零动作基准，精简前环境耗时约0.501s，精简后约0.178s
（三次采样平均，不含PPO更新；真实训练耗时取决于接触与重置情况）。

## 蛇形到双足站立

独立任务：`Mjlab-Zbot-6dof-Transform-Snake-To-Biped`，日志目录
`logs/rsl_rl/zbot_transform_snake_to_biped/`。动作仍为6维独立控制，策略网络／价值网络均30维，
使用脚-地接触传感器；没有动作示范或参考轨迹，课程按脚底表现切换阶段。

初始为六关节全零的贴地直链；浮动根节点旋转为水平并放到地面上方，常量定义于
`tasks/transformation/poses.py`。仅把关节设为零而保留根节点单位四元数会得到竖直链，
因此需要完整设置初态位置和姿态；之后由重力、碰撞和执行器正常模拟。

逆向任务采用两阶段课程：第一阶段学习双脚朝下，目标构型使用弱奖励，
`first_stage_joint_weight=0.1`；base高度和上升速度两项权重为0。每次回合结束、重置前，按本批结束环境的
`Episode_Reward/sole_ground_alignment`平均值判断；严格大于1，并且全体并行环境中
至少80%的环境最近一次完成回合时，已连续0.5秒满足双脚真实触地及脚底几何条件，
才统一进入第二阶段。失败终止不算达标，未完成过回合的环境也不算达标；
按每个环境最近一次回合记录统计，避免小批异步重置提前触发升级。
保持时长和比例由`curriculum_support_hold_s`、`curriculum_support_ready_fraction`配置。
进入第二阶段后，目标构型权重从第一阶段弱权重线性增加到`standing_stage_joint_weight`
（当前默认1.0）。base高度和上升速度权重在两阶段均为0，仅保留诊断指标。
默认`standing_stage_ramp_steps=2400`个策略控制步，按当前每轮24步相当于100个PPO迭代；
每个控制步更新权重，回合重置不清零进度。设为0可关闭渐增。该分数是权重2加权、按回合最大时长
归一化的回合收益，未加权分数最高只有1，单步收益还包含dt，二者不能直接用此阈值。
第二阶段保留脚底朝向及原有惩罚，关闭仅第一阶段使用的初始构型惩罚；
之后不因回合重置或分数回落退回第一阶段。
第二阶段额外保护双脚对地条件：两脚倾角均不超过`support_angle_limit_deg=25°`，
两个脚底位点的离地绝对高度均不超过`support_height_limit=0.01m`。
条件不满足时，base高度和上升速度的正收益为0；目标构型误差惩罚、其他负反馈、
脚底奖励和所有惩罚仍正常计算。目标构型项两阶段均保留负反馈。
每个环境每次重置后，先恢复一次有效双脚支撑才启用失守保护；之后连续违反条件
达到`support_failure_hold_s=0.1s`，以`foot_support_lost`失败重置并扣失败分。
恢复支撑会清零失守计时；重置清除该环境的保护状态，避免蛇形初态反复重置。
这约束了第二阶段的奖励和失败条件，不保证物理状态永远不发生瞬时离地或倾斜。
`Episode_Metrics/sole_support_valid`记录双脚同时满足条件的时间比例。
日志`Curriculum/alignment_to_standing/stage`显示阶段，`alignment`显示本批回合分数，
`ramp`显示第二阶段权重增加比例（0到1），
`support_ready_fraction`显示全体环境最近一次回合的稳定支撑达标比例。
play默认直接使用第二阶段的完整权重，不经过渐增。新检查点保存奖励权重、课程阶段和渐增进度，
完整续训时恢复；旧检查点没有这些元数据，若需从第二阶段开始可设置
`--env.curriculum-start-stage 2`。阈值和第二阶段权重可通过
`sole_alignment_stage_threshold`及`standing_stage_*_weight`配置。

第一阶段额外启用`initial_joint_pose`惩罚：
`first_stage_initial_pose_weight * mean((q-q_snake)²)`，默认权重-0.02，
初始蛇形六关节均为0。仅约束关节构型，不限制root位置或世界朝向。
进入第二阶段时立即将该项权重设为0，不随第二阶段渐增保留，避免拉回蛇形。

训练、播放以及所有阶段的回合重置均保留原始零关节蛇形起点，不混入近站姿样本，
不增加辅助支撑，不改变摩擦、阻尼或其他动力学参数。该起点的圆弧外壳仍可能滚动。
增加两项软惩罚来学习控制初始滚动并抑制错误构型分支：

- `end_joint_flip`：仅对joint1、joint6相对各自站立目标超出±1rad的部分计算平方均值，
  默认权重`end_joint_flip_weight=-1`。初始零角与目标站姿均在免罚区内，
  两个阶段始终保留该惩罚，不随渐增减弱；不限制中间关节弯曲，也不硬改关节限位。
- `axial_roll_rate`：base世界角速度在base局部Z纵轴世界方向上的投影平方，
  默认权重`first_stage_roll_rate_weight=-0.05`。这是奖励约束，不是额外施加物理阻尼。

只有`axial_roll_rate`在进入第二阶段时关闭；`end_joint_flip`、脚触地、脚底姿态及
支撑失守保护继续保留。

逆向任务另外限制两端关节的命令目标：joint1为[-0.688, 1.312]rad，
joint6为[-1.312, 0.688]rad，即站立目标各±1rad；四个中间关节保留模型原范围。
这是动作命令约束，不改MuJoCo物理限位、原始零关节初态或其他任务。
训练与播放均使用相同限制，配置保存在`actions.joint_pos.joint_target_bounds`。

### 奖励权重热改

逆向任务使用`SnakeToBipedHotRewardRunner`，训练时设置
`--env.reward-hot-reload-path outputs/transformation/autonomous/live_weights.json`。
JSON只允许修改权重；未填写的字段保持当前值。例如：

```json
{
  "config": {"first_stage_joint_weight": 0.3},
  "rewards": {"joint_symmetry": -0.5}
}
```

`config`支持阶段构型权重、初态偏离、base投影距离、防翻脚、滚动速度和触地权重；
`rewards`支持脚底朝向、脚底高度、关节对称、roll和失败惩罚。
仅在启动时及完整PPO更新结束后读取文件；下一轮采样才使用新权重，不重建模型或优化器。
使用临时文件写入后原子替换较稳妥。非法JSON、未知字段、非有限数值、错误正负号、
或使升级阈值不可达的朝向权重会整次拒绝，保留之前全部权重。
每次应用/拒绝及生效迭代记录到训练目录`hot_reward_updates.jsonl`；检查点也保存实际权重快照。
若修改脚底朝向权重，第一阶段升级会等待一个完整回合时长，让新旧权重混合的
回合统计退出判定；其他权重热改不增加等待，已进入第二阶段也不会因此回退。
完整恢复检查点后以当前热改文件为准；只加载actor不恢复训练奖励或课程阶段。
动作边界、观测、初态、物理参数和成功/保护条件不在热改范围内。

逆向任务的 runner 还会在初始化、完整恢复和每次优化器 step 后，将触界或越界的
高斯分布标准差参数投影到原范围的严格内部，避免裁剪导致探索方差梯度归零。
区间内参数、声明范围和 Adam 状态均保留；仅加载 actor 时不作投影。

第二阶段核心奖励为：

- 构型误差惩罚：`joint_standing=-mean(SmoothL1(q-q_stand, beta=0.5))`。
  `joint_standing_error_scale`配置平滑区间，默认0.5rad。小误差用`0.5*error²/beta`，
  大误差用`abs(error)-0.5*beta`。目标构型为0，其余为负，不做大误差截断；
  远离目标的关节改善仍有线性收益。第一阶段权重0.1，第二阶段渐增到1.0。
  无base高度门控，从初始化就生效；第一阶段使用弱权重，第二阶段渐增到强权重。
- 双脚实际触地奖励`both_feet_contact`：两个脚部刚体均与terrain存在接触时得1，
  默认权重`feet_contact_weight=0.5`，两个阶段均启用。仅site接近地面不算真实触地。
- base高度和世界Z速度仅作指标，独立奖励权重均为0，成功判定没有指定base高度门槛。
- 原软关节对称项，权重-1。
- 双脚底面朝向，权重+2：令 `c=sign_i*normal_z`、`angle=acos(c)`，逐脚分为
  `0.5*(c-1)+exp(-(angle/30°)²)`，最终取`0.5*(双脚平均分+较差脚分数)`。
  正负分均与脚底高度无关。
- 脚底离地距离成本，权重-1：逐脚计算 `min(abs(sole_height)/0.03,4)` 后取平均。
  悬空和穿地均扣分，独立于朝向奖励，不会削弱转正的正反馈。
- base原点投影支撑距离惩罚`base_support_distance`：计算base坐标原点的XY投影（`body_link_pos_w[:, base_id, :2]`），
  到两个脚底site的XY投影之间线段的最短距离`d`，成本为`min(d/0.05, 4)`，通过负权重转为惩罚。
  在线段上惩罚为0，偏离5cm成本为1，最大成本为4；超出两端也扣分，两脚重合时退化为到该点的距离。
  两个阶段均以完整权重`base_support_distance_weight=-0.5`启用，不受第二阶段渐增影响。
  距离尺度由`base_support_distance_std`配置；`Episode_Metrics/base_support_distance`
  记录回合平均距离（米）。该项为辅助稳定约束，不改变成功判定。

使用真实 `foot_0_sole` / `foot_1_sole` site 的法线，标准站姿的目标方向分别为
世界+Z、-Z，故 `sign_i=(+1,-1)`。正确朝向得+1、侧立得-0.5、翻面得-1。
各阶段奖励直接加权求和；升阶段仅改变构型、高度和速度三项权重，
不改变动作、姿态或重置初态。
逆向任务增加`feet_ground_contact`传感器，读取两脚与terrain的`found`/`force`。
距离成本的尺度沿用 `sole_alignment_height_std`，
角度奖励尺度由 `sole_alignment_angle_std_deg` 配置，观测和动作维度不变，重启训练后生效。

逆向任务保留最初的变形任务 PPO 与控制周期；与正向任务通过共同的 `TransformationExperiment` 复用 PPO 配置，
不再保留逆向专用覆盖：物理与PD周期5ms，策略周期20ms（`decimation=4`，50Hz）；
每轮24步/0.48s、4个小批次，`gamma=0.99`、`lam=0.95`，启用critic价值损失裁剪。
动作标准差初值0.5、范围0.1–1.0，熵系数0.005。奖励参数保持当前设置，其他实验不变。
重启训练后生效。复现临时80ms周期训练的逆向模型时，play需添加 `--env.decimation 16`；
原50Hz逆向模型和正向归档模型按默认周期回放。

逆向任务额外约束base自身绕局部Z轴的实际侧滚，包括关节动作引起的转动：成本为
`(roll_angle/limit)²`，奖励权重默认-1；绝对侧滚角超过默认90°时，以`base_z_roll`
作为失败原因自动重置回标准蛇形。阈值可通过`base_z_roll_limit_deg`修改。
异常终止还会通过`reset_failure`一次性扣10分，覆盖侧滚和其他安全失败。
正常16s超时不扣此分；多个失败条件同时触发只扣一次。该项除以控制周期后再进入
奖励管理器，抵消其dt积分，因此实际每次失败扣10分，不会缩小为0.2分。
测量将世界重力变换到base局部坐标，使用`atan2(-gravity_x, -gravity_y)`计算实际
侧滚角，再减去预设站立姿态的base侧滚角，并将差值归一化到[-180°, 180°)。
站立零位通过真实模型正向运动学计算，使用直立root与`standing_joint_pos`，
不再扣除root与base之间的关节旋转。惩罚、终止与角度指标共用此相对站立角度。
世界偏航不改变此角度；base Z接近竖直时侧滚无定义，按超限重置。
站立姿态为0°，允许范围为±90°。当前蛇形初态相对站立零位约+63.81°，
在允许范围内，不会因为初态侧滚角触发重置。
`base_z_roll_deg`记录每回合最大偏差。默认接触传感器只用于奖励、保护和成功统计，6维动作和30维观测保持不变。
回归覆盖正负90°边界、关节导致的base旋转、root姿态独立性和世界偏航不变性。

`q_stand=(0.312,0.837,-2.02,2.02,-0.837,-0.312)`，沿用机器人原站立定义。
成功统计要求六关节最大角误差<0.1rad、两脚均实际接触terrain、两足底site离地绝对高度
均<0.005m、两脚底倾角均<5°，并满足原有关节/base低速度条件，连续保持2s。低位初态合法，不触发站立
任务的高度跌倒判定。成功只统计，回合保持16s；数值异常、过快下坠和越界仍会终止。

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped \
  --env.scene.num-envs 8192 --agent.max-iterations 1500

./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped \
  --checkpoint-file <snake-to-biped-checkpoint.pt>

./scripts/python.sh scripts/transformation_eval.py \
  --task Mjlab-Zbot-6dof-Transform-Snake-To-Biped \
  --num-envs 16 --checkpoint <snake-to-biped-checkpoint.pt> \
  --output outputs/transformation/snake-to-biped-evaluation.json
```

两个方向分别训练和保存checkpoint。默认评估方向仍为双足到蛇形；逆向评估必须指定
`--task`。训练接口验证不代表逆向策略已经学会起立。

逆向任务接口验证：全套182项测试、静态检查及独立审查通过；真实CUDA环境确认
6维动作/30维观测、水平初态、部分环境重置与有限奖励。4个零动作环境保持贴地到16s，
没有误判站立成功；16环境两轮PPO更新、保存和加载通过。正向归档模型的回归与
SHA256校验保持通过；逆向策略尚未长训收敛。

站姿奖励漏洞修复：全套200项测试通过。使用旧躺姿策略
`2026-10-16_16-43-30/model_100.pt` 回放16s，末100步平均质心高度约0.063m；
相同姿态的加权站姿分由旧公式约2.986降至新公式约1.18e-13。
这是对旧行为的重新评分，验证躺姿不再获得高分；需重启训练并在新奖励下继续学习起立。


## 逆向任务可选支撑观测

默认 `support_observations=False`，末尾 `support_state` 返回零列，策略网络／价值网络
仍为30维，兼容已有检查点。训练时显式加 `--env.support-observations` 后，
策略网络和价值网络各增加13列，变为43维；评估相应模型时加
`scripts/transformation_eval.py --task Mjlab-Zbot-6dof-Transform-Snake-To-Biped --support-observations`。
43维模型与原30维模型必须使用各自对应的观测开关；该选项不参与奖励热更新。
正向任务不添加此观测项。

新增列按以下顺序排列：base本体坐标角速度XYZ（除以10 rad/s），foot_0/foot_1
真实触地标志，两脚sole相对地面的有符号高度（除以0.1 m），两sole朝下有符号余弦，
支撑保护是否已建立，连续失守时间除以`support_failure_hold_s`，课程阶段（阶段1为0、
阶段2为1），第二阶段权重渐增比例。角速度、高度、余弦截到[-1,1]，其余截到[0,1]，
全部输出处理NaN/Inf。读取已有支撑状态；尚未建立状态时保护列为0，不创建状态、不推进计时、
不改变重置或物理过程。显式阶段2起点尚无渐增起始步时，渐增比例按完整目标记为1。

## 可选粗构型升级门槛

`curriculum_joint_error_threshold: float | None = None` 默认关闭，保持现有实验行为。
独立实验可用 `--env.curriculum-joint-error-threshold 1.0`：第一阶段升级仍要求原有
朝向分数和至少80%环境的最近已完成回合达标；每个达标回合结束时，还必须同时满足
`max(abs(q - q_stand)) < 1.0 rad`、已有连续支撑时长和非失败条件。
`support_ready_fraction` 此时同时包含粗构型要求，分母始终是所有环境；未完成回合
不覆盖旧记录，完成但构型错误的回合会覆盖成未达标。`joint_error_threshold` 日志为
当前阈值，关闭时为-1。阈值须有限且大于0，不支持奖励热更新。

这用于防止两脚贴地、但中间关节处于错误折叠分支时过早切入第二阶段并启用阶段二
支撑失守保护；不改变初态、接触条件或保护本身。已进入第二阶段后不降级。
匹配评估可加 `--curriculum-joint-error-threshold 1.0`；默认从阶段2评估时不会因此
退回阶段1。

## 可选完整重力方向惩罚

实验项 `base_gravity_alignment` 默认权重 `base_gravity_alignment_weight=0`，不会改变当前
基线。成本为 `1 - dot(g_base_unit, g_stand_unit)`，范围[0,2]；无效姿态按最坏2处理。
站立参考来自 `standing_joint_pos` 和直立根部姿态的真实CPU正向运动学，按目标关节角缓存。
该项同时比较base本体系重力XYZ分量，补足原轴向横滚只比较XY方向、无法完整区分侧躺
的缺口，对世界偏航不敏感。原roll成本、roll终止阈值、初态和物理设置不变。

需要验证时可设 `--env.base-gravity-alignment-weight -0.5`，或在热更新JSON的`config`
中设置同名 `base_gravity_alignment_weight`；只允许有限非正权重。两个阶段使用相同权重，
不随第二阶段渐增缩放。旧检查点未保存此新字段时，保持当前配置的默认0；最新侧车文件
仍按原规则覆盖。该项是待训练验证的补充姿态信号，不单独保证起身成功。

## 可选正常步恒定偏置

`survival_weight=0` 默认关闭，原实验行为不变。`survival` 的值为
`(~termination_manager.terminated).float()`：正常控制步和纯`time_out`步为1，
真实失败发生的当步为0（失败与超时同时发生时也为0）。它是可选的恒定偏置，
不是成功奖励，不替代真实归位目标，不修改保护、成功条件或初态。

增大PPO折扣因子、延长优化视野时，现有负成本的累计可能使提前失败更有利；
正的正常步偏置用于纠正这种偏好。相同长度且都未失败的轨迹获得相同额外回报，
因此两者的回报差保持不变。两个阶段使用相同权重，不参与第二阶段渐增。
可用 `--env.survival-weight 4.0` 单独试验，或通过热更新JSON的`config.survival_weight`
调整为有限非负值。旧checkpoint缺该字段时保留启动配置的默认0；最新侧车仍按原规则覆盖。

## 逆向观测中的保持时间占位

逆向策略网络／价值网络原有第30列`hold_time`现在固定为0：保留原名称、顺序和1维宽度，
30维及43维模型的网络形状保持兼容。该列原先在大多数训练状态下恒为0，尚未学会归位时
没有训练到非零计时输入；在短暂满足目标后骤然输入保持进度，可能扰动控制。
此修正仅移除该未训练进度输入，不增加开关，不改变真实计时。
`mdp.state.hold_time`、推进逻辑、实际成功条件和连续保持2秒要求完全保留；
成功仍由真实状态计时判断。正向任务继续使用原始`mdp.hold_observation`。

## 独立 Beta 候选入口与自动准备课程

新增任务 `Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta`，使用原生 RSL
`BetaDistribution(action_range=(-1,1))`，独立日志实验名
`zbot_transform_snake_to_biped_beta`。默认8192个训练环境、4000次迭代，均可由标准CLI覆盖。
原逆向任务继续使用Gaussian，正向和walking配置不变；两个架构须使用对应任务ID严格加载，
不根据checkpoint形状猜测架构。Beta原始网络head为12值，由原生分布转换成6维动作，
play和原生导出保留此转换。手动课程训练的2550模型已单独归档；此入口新增的自动准备课程尚未完整冷启动复训验证。

Beta沿用原始零关节蛇形起点、物理设置、末端动作目标±1rad、支撑守护和默认30维观测；
`hold_time`保留为固定0占位。自动构型权重课程为：初始0.6；已完成回合的加权脚底朝向
分数严格大于1、且alignment热改冷却结束后，锁存准备状态并提高到3。准备期间仍是第一阶段，
不会启用第二阶段失守守护。全体环境中至少80%的最近完成回合同时满足连续真实支撑、
非失败和最大关节误差<1rad，才进入第二阶段，按原2400控制步从3渐增到6，不回落到0.6。
对称成本权重为-0.2。朝向之后变差不会撤销已锁存的准备状态。

原任务默认 `curriculum_prepare_pose=False`，保持旧课程。Beta启用该开关并设
`first_stage_prepared_joint_weight=3`；准备权重可通过热更新JSON的同名`config`字段调整为
有限非负数，布尔课程开关不可热改。课程日志和热更新metadata记录`pose_prepared`，
完整续训恢复此状态；旧metadata缺字段时第一阶段沿用未准备状态，自动模式第二阶段仍以
准备权重作为渐增基线。评估默认从阶段2完整目标开始。

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta \
  --env.scene.num-envs 8192 --agent.max-iterations 4000
./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta \
  --checkpoint-file <beta-checkpoint.pt>
./scripts/python.sh scripts/transformation_eval.py \
  --task Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta --checkpoint <beta-checkpoint.pt>
```

## 同侧浅弧起点探索

独立任务：`Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc`。
实验类为 `experiments/transformation.py` 中的 `SnakeToBipedArc`，日志目录为
`logs/rsl_rl/zbot_transform_snake_to_biped_arc/`。直接运行 `./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc`
开始训练，环境数和迭代数可通过下列参数显式指定。
采用 `[0, 0.04, -0.04, 0.04, -0.04, 0] rad` 和配套根部位姿作为低位浅弧起点。
根部四元数也必须采用 `tasks/transformation/poses.py` 中的值；只改关节、仍从直线蛇形的根部姿态慢慢施加
小目标角，在本轮筛选中不能形成同样的地面弧形。

选择浅弧时同时考虑防滚支撑和真实质心到两脚 脚底位点 地面投影线段的距离。
0.3 rad 候选的距离约57.7 mm，未采用。0.04 rad 浅弧停稳后约8.1 mm，承重接触
凸包内的质心余量约5.1 mm。原生 CPU 仿真在15/1.5 PD、5 Nm限矩下，分别保持8 s，
无推力及正负1 N横向推力持续0.2 s的三次检查均恢复稳定。这是单组接触参数下的筛选，
没有证明弧形到脚底落地、站立的完整路径可行，也没有证明实机稳定。

目标角指令限定在站立目标同侧的半区内，中间四关节不超出±π；末端沿用原范围与目标
半区的交集。六个动作仍独立，模型物理关节限位不变，实际关节在外力下仍可能短暂越界。
第一阶段初始构型惩罚改为相对所选浅弧计算。站立目标、课程、成功判据沿用Beta实验，
原始直线蛇形任务保持独立。

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc
# 可覆盖默认值，例如：
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc --env.scene.num-envs 4096 --agent.max-iterations 2000
./scripts/python.sh scripts/transformation_eval.py \
  --task Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc \
  --curriculum-stage 1 --num-envs 4 --output outputs/arc_hold.json
```

无checkpoint的评估仅保持浅弧。已在实际MJLab环境检查4个环境各16 s，均仅超时结束，
没有安全、roll或失守重置；这不计为站立成功。筛选记录位于
`outputs/autonomous/prebend_shallow_reset_validation.json`，MJLab记录位于
`outputs/autonomous/prebend_arc_native_hold.json`。本轮未启动PPO训练。

## 从头训练镜像动作抗扰动起身

当前暂时关闭外部扰动力和力矩，先训练无扰动起身；此设置同时作用于鲁棒和教师实验的
训练、播放与评估。零扰动事件直接返回，不再逐步计算扰动调度；接触传感器只收集接触标志。
教师Critic精简为44维，本次不保留旧教师checkpoint兼容。镜像动作、随机roll和reset继续启用。

独立实验类：`SnakeToBipedRobust`；任务ID：
`Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust`。
从原始零关节蛇形起点训练，沿用30维观测、物理参数、末端目标角限制及站立成功判据，
策略动作改为3维镜像Beta控制。鲁棒实验的训练与播放均移除`base_z_roll`、`unsafe`、
`foot_support_lost`失败终止项，恢复独立的跌倒reset。
只有base原点高度达到0.18 m，才锁存跌倒检测，不要求脚底先落地；此后base高度
低于0.10 m持续0.10 s，终止当前回合并reset为零关节蛇形，重新随机抽取roll，清零速度、
目标角、计时和残留扰动，再次尝试站立。蛇形初态尚未达到启用高度，不会反复触发reset。

另外加入`early_base_ground_contact`，通过独立base-地面接触传感器检测早期失败：目标角
开始偏离零超过0.01 rad后，base出现“接地→离地→再次接地”就立即reset，不要求先
升到0.18 m，也不设最低起跳高度。若软接触始终未断开，则高度从已观测峰值下降至少
1 mm且base世界Z速度低于−0.03 m/s、同时接地，也判为跌倒。初始静止贴地、初始化
0.1 mm间隙的自然落地不会仅因接触而触发该项。相关阈值为`early_contact_*`配置。

`abandoned_exploration`检测放弃探索：初始0.3 s缓冲后，若实际关节RMS速度低于
0.03 rad/s、base线速度低于0.01 m/s、base角速度低于0.05 rad/s，连续超过1 s则reset。
运动恢复、进入近目标区域均清零计时，不累计分散停顿；恰好1 s或更短的动作转换
暂停不会reset。只有base高度误差小于0.03 m且所有关节误差都小于0.25 rad，才允许
长期静止。判定使用实际速度，命令持续变化但机器人卡住也可触发。
对应参数为`stagnation_hold_s`、`stagnation_grace_s`、`stagnation_*_speed`及两项目标容差。

策略只采样3个独立动作`[u1,u2,u3]`，控制六关节目标
`[q1,q2,q3,-q3,-q2,-q1]`。目标角在镜像共同限位内累计：左关节允许区间与右关节
允许区间取负后的交集，避免分别裁剪六关节破坏对称。
鲁棒与教师实验的目标角速度上限均为2π rad/s（360°/s），50 Hz控制下每步目标变化
不超过约0.12566 rad；实际关节速度仍由执行器和接触动力学决定。
初始零姿态与站立目标都满足该关系。约束仅作用于目标角，不改写实际关节位置，外力下
实际角度仍可能暂时不对称。策略网络和价值网络的上次动作观测由3维展开为6维，维持30维输入。
策略导出输出为3个目标角速率动作，外部部署端也须执行镜像展开与目标角积分，不能直接
当作6个关节目标角发送。环境、播放和评估入口均已接入该映射。

默认不加载先验模型。仅保留显式加载旧模型的兼容路径：model_2550的六动作Beta头原有
12个alpha/beta原始输出；主动加载时将左alpha与对应右beta、
左beta与对应右alpha的权重和偏置配对平均，形成3个Beta变量所需的6个输出。
网络主体、critic及观测归一化参数保留。这是线性输出头投影，不保证原六动作策略的
非线性均值或完整起身路线保持不变。原始checkpoint不修改。

每次新回合初始化时，在原蛇形根部姿态上叠加均匀的±90° roll随机量，旋转绕蛇形机身
纵轴（root局部Z，在默认蛇形中对应世界Y），不会把整条蛇绕世界X轴翘起来。
保持六关节为零、初始速度为零，并依据旋转后的碰撞网格调整根部Z，使最低碰撞点
留有0.1 mm地面间隙，避免穿地。通过已有投影重力观测感知朝向，不增加观测维度。
只在真正的回合重置时抽样，包括跌倒reset和超时reset，回合过程中不重新随机位姿。

范围可用`--env.initial-roll-range-deg -45 45`缩小；设为`0 0`关闭随机化，
设为`-180 180`覆盖整圈。`--env.initial-roll-clearance`调整初始地面间隙。
训练记录`initial_roll_deg`指标，评估JSON记录每个环境的初始roll随机量。
旋转轴、73个角度下的网格地面间隙、随机种子和部分环境重置已纳入测试。

连续运动回蛇形的控制器不再接入本实验；策略始终使用3维镜像动作，不再发生控制权接管。
保留16 s超时reset，不再延长或重新计时。
阈值可通过`--env.fall-retry-arm-height`、`--env.fall-retry-height`、
`--env.fall-retry-hold-s`调整。`update_state`只更新站稳计时，始终返回false。
奖励惩罚仍保留，其他实验不变。浅弧实验仍有独立入口，本实验不改用浅弧起点。
历史连续回蛇形检查保存在`outputs/autonomous/continuous_snake_recovery_probe.json`及同名`.py`，
该脚本显式接入历史控制器，与当前实验的跌倒reset设置分开。

新训练从随机actor、critic和新的观测归一化统计开始，不读取model_2550或其他旧checkpoint。
默认`pretrained_checkpoint=None`、`resume=False`，优化器和训练计数均从零开始。初始学习率1e-4，
熵系数0.001，默认8192环境、2000轮。沿用PPO自适应学习率，不保证后续始终等于1e-4。
训练使用单一固定目标，没有脚底课程、阶段切换或权重渐增。base原点世界Z上升速度
奖励权重8，高度进度权重1，站立构型权重12。上升速度0时该项为0，0.075 m/s时
加权分数6，达到0.15 m/s后上升分数饱和为8；下降为负值。速度、高度的正奖励不受
脚底接触/朝向门控。鲁棒/教师的速度项按六关节最大绝对构型误差e平滑衰减：
e≥0.8 rad保留全强度，e≤0.2 rad完全取消，中间乘`x²*(3−2x)`，`x=(e−0.2)/0.6`。
上升收益和下降成本一并衰减，允许归位时缓慢下降；构型项固定权重从6增强为12。
此外，速度项只服务于本回合首次起身：base原点历史最高高度一旦达到0.23 m，
上升收益和下降成本永久归零，直到本环境reset；下降或偏离构型均不会重新开启。
阈值为`ascent_reward_stop_height`，复用已有最高高度状态，不增加观测或指标。
这避免通过上升时偏离构型、下降时接近构型反复套取速度分数。
无需阶段切换，也不新增奖励项。脚底朝下奖励`sole_ground_alignment=1`独立生效，与足端高度和接触无关，
同时考虑两脚平均表现和较差一脚，避免仅一脚转正。
当前鲁棒/教师使用逐脚分数`0.5*(cos(θ)-1)+exp(-θ/10°)`，范围−1～1，θ为脚底朝下的角度误差。
将原高斯峰改为指数尖峰，避免接近0°时斜率趋零；0°、1°、5°、10°的双脚一致加权分数
约为1.000、0.905、0.605、0.360。远处仍有余弦方向反馈，权重为1，无接触门控。
旧逆向课程实验保留原高斯曲线。曲线改变需重启训练进程，权重仍可热改。
脚底离地距离、双脚接触、
base到两脚site线段偏离的过程奖励及其他正则已移除，最终站稳判据保留，不限制起身过程接触顺序。
鲁棒和教师实验仅保留五项：`joint_standing=12`、`base_height_progress=1`、
`base_ascent_velocity=8`、`sole_ground_alignment=1`、`reset_failure=-10`。失败项按事件扣分，其他四项按原有时间积分。
跌倒reset与随机roll初始化继续启用。旧的`standing_stage_*`字段名
仅用于配置和checkpoint兼容，在本实验中始终作为固定权重使用。

六动作checkpoint（包括旧鲁棒实验）只能作为权重迁移输入：即使使用resume入口，迁移时也
重新初始化优化器和训练计数，并输出提示，避免加载尺寸不兼容的Adam状态。
新的三动作checkpoint可完整续训；以上固定目标相关权重采用当前实验配置，不重新启用旧脚底奖励。
预训练文件缺失或网络不兼容会报错，不会默默从头训练。
只有显式指定预训练输入时，才写入`pretrained_initialization.json`；默认从头训练不会生成该文件。

| 扰动设置 | 默认值 |
|---|---|
| 作用点 | base质心；三维世界坐标力和力矩独立采样 |
| 当前幅度 | 每轴力0 N、每轴力矩0 N·m（暂时关闭） |
| 持续时间 | 随机0.08～0.20 s |
| 冷却时间 | 随机0.8～2.0 s |
| 环境分组 | 受扰比例0，全部为无扰环境 |
| 扰动强度 | 当前无外力；没有强度分阶段或渐增 |

每回合重置会清除残留外力和计时器。播放/评估也不施加扰动。
删除恒定扰动强度指标及逐步扰动状态采集；奖励权重热改仅开放当前五项；脚底朝下权重通过`rewards.sole_ground_alignment`修改。

```bash
# 训练命令；可按需设置并行环境数和训练轮数。
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust

# 覆盖并行环境数和微调轮数
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust --env.scene.num-envs 4096 --agent.max-iterations 2000

# 续训新的三动作checkpoint：恢复网络、优化器与计数，不重新注入model_2550。
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust --agent.resume True \
  --agent.load-run <run-directory-name> --agent.load-checkpoint model_500.pt

# 使用新训练检查点做完整扰动评估；不要省略checkpoint，否则只是零动作保持。
./scripts/python.sh scripts/transformation_eval.py \
  --task Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust \
  --checkpoint <new-checkpoint.pt> --num-envs 16 \
  --output outputs/robust_evaluation.json
```

日志目录：`logs/rsl_rl/zbot_transform_snake_to_biped_robust/`。
可通过`--env.events.base-disturbance.params.force-range -5.0 5.0`等原生配置参数调整
扰动范围。若以后重新启用，需同时把force-range、torque-range和disturbed-fraction设为
非零；目前三个设置均为零。幅度加大后须重新检查成功率。

接口检查已完成64环境、2轮PPO更新，严格预训练加载和保存成功；该短运行不用于宣称鲁棒性。
原model_2550的完整强度基线评估记录在
`outputs/autonomous/robust_pretrained_full_push_baseline.json`，分别统计受扰组与无扰组，
并记录每个回合实际外力、力矩的峰值。单一种子的小样本结果不代表微调完成，也不保证实机表现。

## 全状态教师与后续学生

新实验：`Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher`，
类`SnakeToBipedTeacher`，日志目录`logs/rsl_rl/zbot_transform_snake_to_biped_teacher/`。
教师从头训练，保留3维镜像动作、随机roll、固定奖励和现有reset，旧鲁棒观测入口保留作对照。

教师Actor共36维，按下表顺序拼接。实际六关节状态不压缩成3维，因为外力下实际关节
可能偏离镜像关系；独立目标角和上次动作才各用3维。重力和角速度都在base坐标系，
不用有跨界跳变的欧拉横滚角。额外的世界Z速度直接对应上升奖励。关节角误差以站立目标为零点。

| Actor信息 | 维度 |
|---|---:|
| 六关节目标构型误差、实际关节速度 | 6+6 |
| 三个独立目标角、上次三维动作 | 3+3 |
| base投影重力、base坐标角速度 | 3+3 |
| base原点高度、base坐标线速度、世界Z速度 | 1+3+1 |
| 两脚与base接触标志 | 3 |
| 两脚sole高度、朝下法线余弦 | 2+2 |

价值网络共 44 维，在策略网络的36维上增加reset历史及计时状态8维。移除质心相对位置与
当前全零外力/力矩列。Actor特征每步仅收集一次，Critic复用；环境部分reset会使缓存失效。
不再收集未用于当前奖励和reset的支撑保护状态，站稳计时与成功判据保持原定义。

各项按物理单位显式缩放并裁剪，观测范围约±5；关闭策略网络／价值网络的第二层运行统计归一化，
避免接触标志、刚开始变化的计时列被很小的方差放大。网络仍为256/128 MLP，初始学习率
3e-4、熵系数0.005、Beta三动作。这是从头学习的起步配置，不是已验证的最快超参数。

鲁棒和教师均只保留五个指标：`joint_error`、`base_height`、`base_vertical_velocity`、
`success`、`joint_speed_peak`。其他探索与力矩诊断移出训练热路径，避免重复状态收集。

```bash
# 正式训练由用户执行。
./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 2000
```

教师36/44维观测与旧36/53维或30维checkpoint不兼容，需要从头开始，不使用旧先验。
教师训练稳定后，再做学生蒸馏：建议取Actor前24维（编码器、目标、上次动作、投影重力、
角速度），堆叠最近10帧约0.2 s历史，在学生实际访问的状态上查询教师动作。
目前只实现教师实验和观测布局，尚未实现学生训练/蒸馏，也未启动正式训练。
学生最终传感器配置仍需与实物可测量项核对。

鲁棒/教师play默认`seed=-1`，每次启动选择随机种子，避免重复seed=42的首组roll。
同一次运行每次reset仍继续独立采样；启动终端打印前8个环境的实际初始roll偏移。
需要复现时可显式传`--env.seed 42`，训练默认种子保持42。

## 独立蛇足双向指令实验

新增任务`Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional`，从零训练，
随机蛇/足初态对应站立/爬下指令，play支持手动切换。原鲁棒实验和成功归档不变。
详见[双向变形说明](bidirectional-transformation.md)。
