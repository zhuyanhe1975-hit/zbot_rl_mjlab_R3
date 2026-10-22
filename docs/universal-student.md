# 全能学生（R3 / 监督蒸馏、外部扰动与PPO）

任务ID：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student`。参数入口：[学生实验卡片](../src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student.json)。项目版本见[VERSION](../VERSION)。

## R3封存状态（2026-10-22）

综合学生学习暂时封存，保留现有实现与训练入口。无扰动学生归档为[`model_150.pt`](../bests/universal_student_clean_2026-10-22_iter150/README.md)（151次DAgger更新），最新受扰学生归档为[`model_499.pt`](../bests/universal_student_disturbed_2026-10-22_iter499/README.md)（500次DAgger更新）；后者实际训练轮数覆盖了22.d卡片默认的150轮。原始配置、指标、源码快照、SHA256和短程回放验证均随模型保存。两者均是监督蒸馏结果，不是PPO微调结果。

学生Actor每帧56维、连续5帧共280维，包含关节角度/速度、关节目标、前次动作、基座重力方向/角速度、任务/形态、速度指令及步态时钟；保留槽位为零。不包含yaw、真实线速度或外部扰动力。相对yaw的正余弦及估计漂移建模作为后续对照实验建议，本版本未实施。现有状态机仍依赖部分仿真特权信息作完成与恢复判断。

本实验冻结[已归档全能教师](../bests/universal_three_teacher_gaussian_mirror_2026-10-22_iter499/README.md)，训练使用观测历史的学生。保留四任务接口：双足运动、爬下、蛇形运动、站起。监督学习阶段只优化动作监督误差，环境奖励仅用于监测。完成监督学习后，使用独立的学生PPO任务按环境奖励微调。

当前三项奖励修正仅属于学生环境，冻结教师的权重、奖励配方和任务配置保持原样：变形也计算动作变化惩罚（权重-0.01，复用原动作变化公式）；双足运动启用首次触地足底平整惩罚（`foot_flat_touchdown`，权重-0.2，sigma=0.05）；蛇形改用三轴归一化速度误差及联合跟踪得分，覆盖超速、非指令轴漂移和组合指令。参数在学生卡片的rewards中调整，实现位于`tasks/universal/student_rewards.py`，不反向修改教师或独立蛇形任务。

蛇形评分为`k=exp(-((actual-command)/snake_std)^2)`、`score=2*(mean(k)+prod(k))`，范围0至4，逐轴平均与三轴联合各占一半；`snake_std=(0.05,0.05,0.15)`分别对应m/s、m/s、rad/s，不再设置0.2m/s的平移合速度奖励封顶。学生的双足速度奖励与变形阶段速度门控保留原规则。DAgger仍只优化动作监督误差；PPO阶段则使用这些环境奖励优化学生策略与价值网络。旧run与修改后的环境奖励曲线不能直接比较。动作变化惩罚不是硬限速，落脚平整项不保证全足底接触。

## 22.d：22.a学生蒸馏＋外部扰动（封存前训练入口）

任务ID：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-Disturbed`。参数入口：[22.d卡片](../src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_disturbed.json)，直接继承22.a的学生卡片。

```bash
./bests/test.sh universal_student_clean_2026-10-22_iter150
./bests/test.sh universal_student_disturbed_2026-10-22_iter499
# 如需重开150轮训练，由用户手动运行：
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-Disturbed \
  --log-root logs/rsl_rl_universal_student_disturbed --env.scene.num-envs 8192 --agent.max-iterations 150
```

长训由用户启动。默认8192环境、150轮，每轮24步，run-name为`universal_teacher_dagger_disturbed_22d`，日志与模型独立保存至`logs/rsl_rl_universal_student_disturbed/zbot_universal_snake_leg_omni_student_disturbed/`。不读取PPO检查点，也不自动续训原22.a学生；保留原教师冻结、五帧传感器历史、四任务Gaussian镜像动作、均衡回放与固定示范。20轮教师执行预热＋100轮逐步交接＋30轮学生独立执行，训练仅优化DAgger监督目标，不加入PPO或Critic学习。

外扰沿用已存在的基座冲量事件：每个轴力为[-5,5]N、力矩为[-0.5,0.5]N·m，每次持续0.08～0.20秒、间隔0.8～2秒。训练约25%的环境在创建时随机选为受扰组，其余保持静默。`pretraining_scale=1`使外扰从首轮教师预热开始生效，不依赖PPO进度；教师在同一受扰状态上提供标签，但权重与统计仍冻结。其余环境、奖励、动作、任务状态机及蒸馏参数继承22.a。

回放默认1个环境，因此22.d的play配置把受扰比例设为100%，避免单个机器人被随机抽为静默。扰动力、力矩、时长及间隔与训练相同，观测噪声仍沿用22.a的play关闭设置。`run.sh`只加载22.d的Actor，不需要重新读取教师。训练完成后默认末次检查点为`model_149.pt`。

22.a、22.b、22.c命令保留在启动脚本的注释中；PPO v2/v3/v4日志及配方不受本子实验影响。以下为保留实验资料；封存时`train.sh`/`run.sh`的实验命令均已注释，不再默认启动实验。

## 保留的PPO v2微调入口

任务ID：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO`。参数入口：[PPO卡片](../src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_ppo_v2.json)。

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO --log-root logs/rsl_rl_universal_student_ppo_v2
./scripts/python.sh play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO --log-root logs/rsl_rl_universal_student_ppo_v2
# 小规模或覆盖参数：
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO --log-root logs/rsl_rl_universal_student_ppo_v2 --env.scene.num-envs 4096 --agent.max-iterations 500
```

长训由用户启动。PPO v2卡片默认8192环境、500轮，每轮24步，独立日志位于`logs/rsl_rl_universal_student_ppo_v2/zbot_universal_snake_leg_omni_student_ppo_v2/`，run-name为`student150_ppo_v2`；使用上面的显式play命令回放该目录模型。尚未生成PPO模型时，可以显式给出原学生检查点进行Actor回放。

首次训练从已验证的v2 `2026-10-22_13-45-40_universal_teacher_dagger_v2/model_150.pt`加载学生Actor及其观测归一化统计，并校验卡片中的SHA256。重新建立Critic和优化器，PPO从0轮计数，不导入DAgger的优化器、回放、迭代或环境时钟。更换初始化学生时，需同时调整`agent.initial_student_checkpoint`与`agent.initial_student_sha256`；不自动选择训练目录的最新模型。

前10轮仅训练Critic，随后同时优化Actor和Critic。复用四任务独立价值网络、完整任务奖励和按任务归一化优势；Actor保持280维历史传感器观测，Critic使用56维特权观测。Actor归一化统计在PPO阶段冻结，避免改变已学策略的输入尺度。Gaussian探索标准差在首次迁移时设为0.1，保持固定；学习率3e-5，fixed schedule，clip=0.1，entropy=0。教师不加载、不查询、不更新。PPO之后用冻结的150代学生作轻量运动参考，保护双足和蛇形分支；变形分支不参与参考约束。

### v2修正范围

- **下降交接一致**：学生PPO训练/回放都要求低位接触与实际/目标关节归直。进入容差0.15rad，退出容差0.20rad；仍连续确认约2秒。小幅目标振荡可在容差带内保持确认，超过退出阈值或失去物理条件会重新计时，不写入物理姿态，也不冻结执行器。教师与DAgger保留原规则。
- **变形质量**：末段增加目标角、实际速度和持续速度动作代价；实际关节超速按站起3rad/s、下降2.5rad/s开始计费；近地下降速度阈值0.10m/s，起身速度奖励在0.10m/s后降低；冲击保护覆盖两个变形方向。代价有界，策略项排除执行器强制归直，冲击项仍保留。这些是仿真初始调参值，不是硬件安全限值或硬限速。
- **交接后运动**：训练中一半成功变形试验继续执行3秒目标构型运动，另一半按成功重置。发生跌倒或unsafe仍立即终止。这样会改变训练完成时间统计，不能直接与v1的成功回合耗时比较；独立验收另记实际交接耗时。
- **运动保留**：仅双足/蛇形任务参与源学生KL约束，每任务最多1024条访问状态，任务等权。系数从0.05按200次实际策略更新衰减至0.005；参考模型及其统计冻结，Critic预热时不做参考更新。参考修正使用无动量梯度步（学习率0.001）并与PPO动量隔离，避免零KL时被旧动量推离源策略；衰减系数直接缩放修正梯度。
- **蛇形课程**：前20次实际策略更新训练停车与正负单轴，随后20次加入两轴组合，再加入三轴组合。可用正负类别按打乱的循环均衡采样；速度范围从一半在100次策略更新内扩到完整范围。回放始终使用完整范围，手动指令不受课程限制。
- **周期验收**：训练前记录源策略基线，之后每25次实际策略更新做一次80环境、8秒标称独立评测，覆盖正负单轴、正负三轴组合、停车及双向交接共20种场景，每种4个初始朝向。分别记录三轴误差、运动/就绪比例、交接耗时与落脚倾角，不把正反方向混成一个平均值。JSON和对应检查点写入当前运行目录，TensorBoard保留细项，控制台只显示摘要。

旧PPO v1日志`logs/rsl_rl_universal_student_ppo/`和`model_99.pt`保留对照。v2按源150代学生重开，拒绝把v1作为完整续训；原v1及DAgger模型仍可显式仅加载Actor回放，回放使用当前v2环境判据。短程验证不等于质量目标已收敛。

```bash
# 续训只接受本阶段PPO检查点，不重新读取初始化学生：
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO --log-root logs/rsl_rl_universal_student_ppo_v2 --agent.resume True --agent.load-run <PPO运行目录名> --agent.load-checkpoint model_59.pt
# 回放指定PPO v2模型：
./scripts/python.sh play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO --log-root logs/rsl_rl_universal_student_ppo_v2 --checkpoint-file logs/rsl_rl_universal_student_ppo_v2/zbot_universal_snake_leg_omni_student_ppo_v2/<运行目录>/model_59.pt
```

PPO检查点保存独立阶段信息、Critic预热进度及源学生路径/SHA，完整续训会恢复优化器并重建任务计时；禁止把DAgger或教师模型作为PPO完整续训状态。Actor回放不依赖初始化学生或教师文件。

## 独立比较检查点

```bash
./scripts/python.sh scripts/evaluate_student_ppo.py \
  logs/rsl_rl_universal_student_v2/zbot_universal_snake_leg_omni_student/2026-10-22_13-45-40_universal_teacher_dagger_v2/model_150.pt \
  <PPO_v2检查点路径> \
  --seconds 12 --num-envs 80 --output outputs/student_ppo_comparison.json
```

该命令只推理评测，不执行优化。默认使用当前v2判据与无噪声标称环境，不能与其他判据/外扰条件下的旧报告直接混比。`handoff_fraction`表示真实FSM进入目标运动任务；严格跟踪`success`还要求持续姿态和速度达标，两者分别报告。

## 保留的DAgger入口

原DAgger实验、配置和日志保留。需要重做监督学习或回放原模型时显式指定原任务：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student \
  --log-root logs/rsl_rl_universal_student_v2 --agent.max-iterations 500
./scripts/python.sh play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student \
  --checkpoint-file logs/rsl_rl_universal_student_v2/zbot_universal_snake_leg_omni_student/<运行目录>/model_150.pt
```

原v1学生也可通过该play任务仅加载Actor回放，但不能作为v2 DAgger完整续训状态。本机脚本保留环境激活和显式Python命令；跨机器可使用`scripts/python.sh`并配置已有MJLab环境。

## 观测与动作

教师读取干净、单帧的56维真值观测。学生每帧保留56维兼容布局，连续五帧共280维；被移除的特权字段只保留零占位。学生保留关节状态、上一动作、指令、步态与软件状态机信息，以及base IMU对应的角速度与重力方向，并加入适度观测噪声。学生Actor不读取真实线速度、高度、接触、航向或质心信息。

学生IMU参考在所有构型中统一为 `base`。教师及仿真运动任务仍按双足/变形参考 `base`、蛇形运动参考 `foot_0_sole`。当前实现仍只有base IMU；已知硬件ma/mb映射不代表已实现多IMU观测，本次不更改传感器接口。

继承四任务独立Actor和分任务归一化，任务整数ID在网络入口转为one-hot。动作仍为Gaussian：双足6维位置、蛇形6维速度、爬下/站起各3维速度；变形执行镜像 `(a, b, c, -c, -b, -a)`。动作映射、速度采样与物理边界继承全能教师。

## v1诊断与修正依据

对v1学生 `model_150.pt` 与归档教师进行独立随机复位的探索性对照：共64环境、24秒，每项16环境。

| 指标 | 归档教师 | v1学生 |
|---|---:|---:|
| 站起成功 | 16/16 | 2/16 |
| 双足跌倒 | 1/16 | 15/16 |
| 爬下成功 | 16/16 | 16/16 |

该对照说明v1尚未完成教师能力迁移，并非完整性能验收。v1每轮采集196,608帧，但仅做4次、总计4096条样本的监督更新；站起回放近半处于执行器强制归直阶段，挤占正式站起的训练量。完全由初始学生执行、仅保留近期回放，也缺少稳定教师轨迹支撑。

## v2在线蒸馏

冻结教师始终在被采集状态上提供确定性动作均值标签。前20轮由教师控制环境，采集教师轨迹；随后100轮把教师控制概率从1线性降到0。每个环境的教师/学生控制选择在一轮开始时独立采样，并在该轮保持，避免逐步抖动切换。交接完成后由学生执行。

回放保留最近4轮，每轮每任务最多4096条样本，按任务均衡训练。默认 `dagger_updates=0` 采用epoch模式，`dagger_epochs=1`，每轮完整遍历均衡回放一遍，批量2048；学习率仍为 `3e-4`。这替代v1固定4次更新的方式。

教师预热阶段另保存固定样本，每任务最多4096条，后续不随近期回放淘汰；固定样本监督误差占总损失的0.25，用于保留教师轨迹约束。

恢复代码3.25对应执行器强制归直，仍统计采集数量，但不加入学习目标；正式站起继续训练。速度动作任务1、2、3的教师标签裁剪到 `[-1, 1]`，与实际执行范围一致；任务0的原生双足Gaussian位置动作保持原值。教师权重与归一化统计保持冻结，环境奖励仅记录，不参与优化。

## 验证边界

v1建立时曾通过3161项测试、静态检查、32环境短程训练及8环境回放；这些记录属于旧配方，不能作为v2修正有效或训练收敛的证据。v1建立记录位于 `outputs/diagnostics/universal_student_smoke/verification.json`。v2的最终检查结果应以本次实际验证记录为准。

当前软件状态机仍使用仿真真值完成阶段判定，因此尚未形成完整实机部署闭环。后续需要用可用传感器或估计器替换状态机输入，再验证连续切换与恢复能力。五帧历史和噪声不代替真实传感器标定；教师现有弱项也可能传递给学生。

R3 建立验证：3183项测试、全源码/脚本/测试Ruff、修改范围Pyright与Shell语法检查通过。64环境完成10轮有界验证（仅验证时将预热/交接缩为8/1轮），覆盖教师执行到学生独立执行，再续训1轮；v2和原v1学生均通过实际play入口8环境×32步检查，回放不加载教师。迁移后以全局步数4000验证了续训任务计时重置。结果见`outputs/diagnostics/student_v2_smoke/verification.json`。这些检查不代表v2技能已收敛；全量训练由用户启动。
