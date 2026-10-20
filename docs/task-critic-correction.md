# 200代不动：奖励分组版失效与按任务价值分离对照

> 新增独立[归档指导＋PPO特权教师](guided-privileged-teacher.md)，现为train.sh／run.sh默认实验。本文的纯PPO修正版使用`ZBOT_EXPERIMENT=taskrouted`选择。

## 先确认问题

用户检查点 `logs/rsl_rl_multicritic/zbot_universal_snake_leg_omni_teacher_multi_critic/2026-10-20_13-34-17_reward_groups_v1/model_200.pt`（201次更新）复测：请求前进.08m/s，双足实际vx仅.000210，蛇实际vx .0000133。额外扫过双足vx .05/.08/.15/.3/.5、vy .2、wz .8，均几乎静止。因此不只是play默认.05m/s太小，或键盘没有送入指令。

1000环境×24步无更新探针中，双足输入有效速度绝对均值为(.0809,.0351,.0855)，原始请求绝对均值(.1471,.0638,.1556)，指令存在。

## 上一版适配的问题

物理奖励按任务门控，分组求和仍等于原奖励，但**Actor目标并没有保留原配方的比例**。旧版使用四个奖励组Critic，对各组advantage独立归一化后求和，相当于再给每组乘以其advantage标准差的倒数；低幅度代价也会被放大到单位尺度。仅检查reward求和和人工单个组激活的测试不足以保护实际双足同时激活三个组的行为。

model200探针：

| 组 | 活跃样本原始adv标准差 | 双足归一化adv标准差 | 双足策略梯度范数 |
| --- | ---: | ---: | ---: |
| motion_tracking | .19536 | .3521 | .3573 |
| contact_gait | .006404 | 1.0000 | .8221 |
| regularization | .003514 | .7908 | 1.0498 |

运动组中还混合双足／蛇的统计。实际本次motion与contact／regularization梯度余弦为+.535／+.697，不能宣称它们在此处强烈反向抵消。确定的是目标权重被重分配，双足运动学习信号相对变小。

[Hutter多Critic论文§3.3](https://arxiv.org/html/2507.08656)将正则随locomotion／manipulation／contact schedule功能分组；不是单独加一个全任务regularization组。其同时移动操作任务也不等于当前四类互斥构型技能。此前按组适配过于粗糙，不能把“保留物理reward”当作“保留成功双足优化目标”。

## 新的按任务Critic

新增独立 `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Task-Critic`，不修改奖励组v1算法或旧检查点。

- 一个共享Actor、56维输入（构型1维／任务1维），网络、动作、物理reward及原PPO超参数不变。
- 四个互斥任务Critic分别估计**该任务完整标量奖励**：双足运动0、爬下1、蛇运动2、站起3。双足内部的速度、步态、能量、动作变化等比例保留，不再分别单位化。
- 在act时快照任务身份，避免env.step自动reset后把上一动作的奖励错路由；GAE在终止与任务边界截断，按活跃任务样本归一化。
- 上游Critic loss默认平均4列，实际每行只有1列活跃，因此内部value_loss_coef乘4，恢复单任务的标量Critic梯度；不是额外提高原配方的有效价值损失权重。
- 真实task0的return、timeout、advantage、Actor梯度及Critic梯度，均用独立标量PPO对照验证。不是仅用人工“单个奖励组”测试代替实际双足。
- Actor／Critic的观测统计仍全任务共享；此次先隔离奖励／价值目标改动，不把冻结策略替换std的实验当作必须修改观测归一化的证明。

任务顺序、角色和算法版本写入独立checkpoint契约。新任务完整续训会在修改权重之前拒绝奖励组v1或旧单Critic检查点。旧v1可继续用原任务回放，不重新解释其权重。

## 有界对照

同一seed42、8192环境、24步rollout，标称确定性回放20场景×4航向，指令切换后10秒统计后半程：

| 训练 | 请求前进vx | 实测vx | 正混合实测vx/vy | 最后探索std |
| --- | ---: | ---: | --- | ---: |
| 奖励组v1，纯双足100次 | .08 | .0000153 | 见JSON | .057 |
| 新按任务版，纯双足100次 | .08 | .030106 | .082997/.038584 | .265 |
| 新按任务版，联合100次 | .08 | .006095 | 见JSON | .334 |

**旧方案在纯双足条件下也失败，不能仅归因于混合构型的观测统计。** 新方案在相同预算恢复了平移学习，是支持纠正优化目标的实际对照；这些数值不是全部全向技能通过的证明。

另将新联合模型有界续训至250次更新，使40%双足样本量约等于纯双足100次。该试验结果以同目录 `task_mixed250_evaluation.json` 为准，最终表另记录于`comparison.json`。没有启动6000轮全量训练。

## 修改前存档与验证

此次修正前再次完整存档：

`/home/yhzhu/myWorks_vips/archives/zbot_rl_mjlab_R2/pre_taskcritic_20261020_133959/`

tar大小2,054,246,400字节；SHA256 `b87cca31311f13b580c523a499e82ad8bf617d356bda9a26986b46d3dd3b4e92`。原grouped源码、入口和用户model200与存档逐项匹配。最初大修存档仍保留。

当前2216项测试、Ruff和作用范围Pyright通过；128环境混合任务2次更新＋1次完整续训通过，实际play.main入口16环境×16步短回放有限值检查通过。旧bests不改动。原始记录及可重跑诊断在 `outputs/diagnostics/multicritic200/`。

在同一入口显式选择新实验：

```bash
ZBOT_EXPERIMENT=taskcritic bash train.sh
ZBOT_EXPERIMENT=taskcritic bash run.sh
```

旧奖励组版使用 `ZBOT_EXPERIMENT=multicritic`，已可行走的旧纯双足版使用 `ZBOT_EXPERIMENT=biped`。是否把新联合方案设为默认，要依据学习结果，而不只是程序冒烟测试。

## 后续对照：仅分任务归一化仍不足，策略分支恢复部分行走

后续全部仍为有界诊断，未启动6000轮训练。相同seed42／8192环境／24步；联合部署双足40%、蛇40%、站起10%、爬下10%。联合250次的双足帧数约等于纯双足100次，但全局扰动课程进度不同，因此不是完全等价的四次独立训练。

| 联合训练配置 | 更新数 | 前进指令.08的实测vx | 结论 |
| --- | ---: | ---: | --- |
| 完整奖励任务Critic，共享Actor／观测统计 | 250 | .000841 | 仍停滞 |
| 再分离Actor观测统计，Actor参数共享 | 250 | .000402 | 仅分任务归一化不足 |
| Actor按任务分支，Actor／Critic各按任务统计 | 250 | .026496 | 出现平移响应；.15指令实测.14945 |
| 上项再以低速指令续训100次 | 350 | .073901 | 双足前后及混合平移继续改善，全部技能仍未通过 |

旧单轴前进采样范围是.2～.5，默认play的.05只在混合指令中偶尔接近，不能据此认定低速任务已学会。新增独立任务 `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Task-Routed`：

- 每个任务保留完整标量奖励，Actor四个独立参数分支；共享一个PPO优化器、学习率调整和全局梯度裁剪，因此仍不等于四套独立PPO。
- Actor与Critic按原始任务码分别更新56维输入统计，构型／任务两个标量直接保留；普通旧教师和bests不采用此模型。
- 新任务初始速度上限(.15,.08,.4)，单轴幅度为上限的.1～1；旧任务采样函数默认仍保持.4，不改旧实验分布。
- 一个部署接口、一个检查点，可JIT导出；不是加载原walking教师，也没有动作监督。

**.05m/s的静止死区仍未解决。** 350次模型实测：.05指令vx=.000251；.08指令vx=.073208但伴随wz=-.3683；.15指令vx=.140983、vy=.007290、wz=.032758。正组合(.08,.05,.3)实测(.078726,.042893,.199231)。单独侧移／转向和蛇爬没有达到要求，不能报告全能训练成功。

默认play速度在此新任务中明确改为(.15,0,0)，用于观察已验证的行走响应；不以此宣称修好了低速学习。旧实验play默认速度不改。默认仍混合双足／蛇初态，蛇模型当前可能保持静止；只有键盘显式切换当前机器人时才变形。

## 当前入口与检查点

`train.sh`、`run.sh` 默认选择 `taskrouted`，保留原有 `taskcritic`／`multicritic`／`biped`／`walking` 分支，没有添加用户启动脚本。

```bash
# 直接查看已验证模型，不需要先重新训练
bash run.sh
# 从新实验最新检查点继续；max-iterations是额外更新次数
bash train.sh --agent.resume True
# 如需从头训练，默认不续训
bash train.sh
```

已验证检查点：

`logs/rsl_rl_task_routed/zbot_universal_snake_leg_omni_teacher_task_routed/2026-10-20_14-10-21_verified_routed_low350/model_349.pt`

来源为联合250次＋低速100次的诊断模型，数值模型由诊断目录移入正式模块；复制检查点只修改角色、算法标记与导入路径资料。212个模型／优化器状态张量逐一相等，原始检查点和原始params留存。转换记录为 `outputs/diagnostics/multicritic200/public_checkpoint_conversion.json`；正式模块与诊断模型另有逐位输出／状态一致性测试。注册任务的正式入口实际完成16环境×16步有限值回放，以及完整恢复优化器和课程后从350到351次的一次更新。此更新只写诊断目录，默认回放仍读取验证过的350次模型。

修改策略分支前第三次完整存档：

`/home/yhzhu/myWorks_vips/archives/zbot_rl_mjlab_R2/pre_routed_20261020_140711/`

大小2,167,162,880字节；SHA256 `d899b59957dcd7bdf0fce88175017bff9c71477b278937eddd10abd9547f0ced`。旧奖励组模型不转换、不覆盖。

最终验证：2269项测试通过；修改范围Ruff／Pyright通过；两个入口bash语法与--help通过。数值结论来自实际仿真，不以测试通过代替学习效果。完整结果在 `public_routed350_evaluation.json`、`routed_low350_evaluation.json`、`routed_low350_amplitude_sweep.json`，均位于上述诊断目录。

## Play摔倒后要求reset的修复

教师play使用`auto_reset=False`保持构型，但此前仍注册了训练的`biped_fallen`／`unsafe`及超时终止。首次终止后，MJLab标记环境等待显式reset，而viewer下一帧仍调用step，导致`must be reset ... auto_reset=False`。

教师族仅在play配置中清空episode终止条件，保持连续物理步进；摔倒不会自动复位或切换构型，Enter仍可手动复位。训练的摔倒、失败、超时和成功回收条件保持原样。五种教师配置的回归测试先复现失败再通过；16环境GPU实测强制1号机器人倒地、episode长度超过60秒后连续8步，无终止和待reset标记，显式局部reset后也可继续step。原walking／蛇形独立任务配置不变。


## Play定时随机速度采样

当前`Task-Routed`任务play启用`play_random_velocity=True`，替代上述固定.15m/s演示默认值。启动及每次6～12秒计时到期，按训练相同分布采样停车、带符号单轴和组合vx/vy/wz，受当前速度上限(.15,.08,.4)约束。只改变requested velocity，沿用平滑过渡，不清零实际速度指令，不改变构型、FSM任务或物理姿态。各环境的计时独立。

W/S/A/D/Q/E、X或构型键接管当前机器人后，该机器人保持手动指令；其他机器人继续定时随机采样。Enter复位后恢复自动采样。仅当前Task-Routed默认启用，训练采样配方及旧实验／bests的play默认行为保持原样。需要固定速度回放时可用`bash run.sh --env.commands.transform.play-random-velocity False`，此时仍采用原play_velocity配置。
