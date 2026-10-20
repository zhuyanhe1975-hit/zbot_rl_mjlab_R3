# 独立多Critic全能教师实验

> 此奖励分组v1在200代出现静止停滞，现保留为历史对照。当前入口、修正证据和未解决项见[按任务修正报告](task-critic-correction.md)。

## 修改前存档

按用户要求，生产修改之前先完成完整工作区存档：

`/home/yhzhu/myWorks_vips/archives/zbot_rl_mjlab_R2/pre_multicritic_20261020_132549/`

- `workspace.tar`：2,029,352,960字节，4963个常规文件；包括代码、配置、Git目录、训练记录和检查点。仅排除Python／测试／lint缓存。
- `manifest.json`：每个存档成员的大小与SHA256；257个关键源文件、测试、bests及纯双足250检查点已与实时文件逐项匹配。
- `workspace.tar.sha256`：`b4ad8dcd6c3c319913efbb43661f31a07a9ca850efe0905546363deacdd75013`。
- `git-head.txt`、`git-status.txt`、`working-tree.patch`：保留提交位置、未提交修改及Git二进制补丁；未擅自提交当前脏工作区。

恢复时先将tar展开到单独目录检查，避免覆盖之后的新训练记录。存档未放在仓库内，不会被本次改动或训练覆盖。

## 实验与入口

新增任务 `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Multi-Critic`，独立实验名 `zbot_universal_snake_leg_omni_teacher_multi_critic`。原基础Teacher、Wide-Biped纯双足任务与bests参数保持原配置。

```bash
# 新的联合多Critic实验，完整训练由用户启动
bash train.sh
bash run.sh

# 此前56维纯双足对照，继续使用原目录
ZBOT_EXPERIMENT=biped bash train.sh
ZBOT_EXPERIMENT=biped bash run.sh

# 用户存档前正在使用的原独立walking命令
ZBOT_EXPERIMENT=walking bash train.sh
ZBOT_EXPERIMENT=walking bash run.sh

# 新算法的纯双足／纯蛇对照；各自独立日志目录
ZBOT_SKILL=biped bash train.sh
ZBOT_SKILL=biped bash run.sh
ZBOT_SKILL=snake bash train.sh
ZBOT_SKILL=snake bash run.sh
```

默认新联合日志根为 `logs/rsl_rl_multicritic/`，run名 `reward_groups_v1`；新算法单模式日志在根目录下的`biped/`和`snake/`。旧56维双足日志仍为 `logs/ablations/biped_scalar_only/`。启动脚本将额外CLI参数传给对应任务。

默认8192环境、6000轮，每轮24步、每50轮保存，周期验收仍关闭；不新增100代阶段切换或阶段信息。新实验恢复双足40%、蛇40%、站起10%、爬下10%的联合任务采样。play为16环境，启动双足／蛇各半，默认前向请求.05m/s；不自动变形，速度和变形键只控制选中机器人，保持相机滤波。新任务仍使用规范FSM、site坐标约定及蛇运动±60°目标限位。

## 算法改动及与论文的关系

依据[Hutter等人的多Critic工作](https://arxiv.org/html/2507.08656)按奖励分量学习独立价值，参考实现调查见[rsl_rl多技能选型](rsl-rl-multiskill-references.md)。没有安装或直接覆盖未合并的PR #224；使用当前已安装RSL的PPO更新路径，仅增加分组存储、价值模型及回报计算。

保持一个Actor、56维共享特权输入（构型1维／任务1维）、6维Beta动作、256/128 ELU网络、原动作映射、奖励公式及权重、PPO超参数、随机化与扰动课程。新增四个独立同容量Critic：

| 奖励组 | 内容 | 活跃任务 |
| --- | --- | --- |
| motion_tracking | 原速度跟踪、停车 | 双足／蛇运动 |
| configuration | 原变形姿态、进展、抬升、完成、失败、落地项 | 站起／爬下 |
| contact_gait | 原抬脚、落地冲击、足底滑移、足底平行项 | 双足运动 |
| regularization | 原动作变化、关节代价、功率、直立及蛇专属罚项 | 所有任务，原函数仍保留其任务门控 |

所有23项恰好归属一组，新增未知奖励时必须显式指定归属。各组之和仍等于原标量奖励。接管当前环境reward manager的compute，在奖励已算好但尚未reset之前读取其加权分量；不再次执行带历史状态的奖励函数，也不从reset后的机器人重新推算奖励和任务mask。

各组分别保存价值、奖励、GAE及return。非活跃组价值固定零，并阻断跨非活跃阶段的GAE传播；每组只在活跃样本中归一化advantage，空组保持零。四组归一化advantage求和后，复用上游**一次PPO clipping**更新Actor。Critic损失继承上游对四列误差取均值，是本地适配；不宣称与论文所有细节完全相同。

Actor及Critic观测统计仍在整个rollout优化结束后更新一次；未同时引入分任务观测归一化或PopArt。四组奖励并不等于四套任务隔离训练：双足／蛇仍共享运动组，站起／爬下仍共享构型组。此版本验证“分解价值和优势”这一变量，不能承诺复现单任务训练速度或解决所有停滞因素。

## 存取与兼容

使用独立checkpoint角色 `privileged_omni_reward_group_teacher` 和 `reward_grouped_ppo_v1` 标识、固定奖励组顺序，保存四Critic及优化器状态、Actor、已完成更新数、FSM版本和原课程状态。完整续训在加载权重之前校验角色、布局、算法、分组顺序及Critic数量／形状。

旧单Critic checkpoint不能直接当作四Critic完整续训，新旧任务回放各选各自任务。原单Critic教师及bests仍使用原Runner、原文件与原观测布局。部署推理仍只执行Actor，四Critic不是四个需要切换的部署策略；当前教师含特权观测，学生训练依然暂缓。

## 实际验证

已执行的机器人训练仅为128环境×2次更新，随后完整续训1次，合计3次；未执行6000轮全量训练。

- 训练和续训退出码均为0，最后checkpoint为`model_2.pt`，完成更新数3。
- Actor normalizer样本数9216（128×24×3），与一次rollout更新统计一致。
- 独立新角色回放16环境×32步：双足8／蛇8，观测／动作／奖励有限，无自动构型切换；速度修改只影响选中机器人。
- 分组奖励求和与原总奖励最大绝对误差 `1.4901161193847656e-08`。
- 全套测试2175项通过；包含独立GAE参考、timeout、活跃掩码、单组归一化等价性、错误checkpoint加载前拒绝、旧任务契约。Ruff与作用范围Pyright均通过，bests的68个文件与修改前存档哈希完全一致。

原始日志及结果：`outputs/diagnostics/multicritic_integration/`。以上证明程序与兼容接口正常，不代表技能已经学会；学习效果需要用户启动后按各技能实际帧数与固定场景验收。

### play实际入口补验

修正 `play.py` 的检查点准备分派：调用注册Runner的 `prepare_playback_env_cfg`，使多Critic角色使用自己的校验，旧教师仍用原校验。此前直接构造Runner的短回放没有覆盖这一入口分支，新增入口回归测试可复现原错误并保护三类任务。用户实际`model_50.pt`已通过`play.main`完整加载及16环境×16控制步的无窗口检查（仅替换最后的viewer窗口），观测／动作／奖励有限。修正后全套2178项测试、Ruff、作用范围Pyright通过；无需重新训练检查点。记录在 `outputs/diagnostics/play_dispatch_fix/`。
