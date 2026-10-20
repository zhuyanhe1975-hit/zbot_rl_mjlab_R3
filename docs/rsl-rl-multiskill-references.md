# 基于 RSL-RL 的多技能项目选型

检索及GitHub API核对日期：2026-10-20。星数是此次读取时的快照，不是论文引用数。当前不修改纯双足训练、bests或外部已安装的rsl_rl，不启动完整训练。

## Hutter本人／团队的直接相关工作

1. [Advanced Skills through Multiple Adversarial Motion Priors in Reinforcement Learning](https://arxiv.org/html/2203.14912)：Multi-AMP，ICRA 2023（2022预印本）。Marco Hutter是共同作者。单策略按风格选择器选取对应判别器及动作先验，允许没有先验数据的技能仅用任务奖励；实机展示四足／直立两足构型切换、起立、导航、坐下。这比单纯多地形行走更接近蛇足构型切换。论文用one-hot风格选择器，不代表本项目必须恢复one-hot；保持现有1维task码做路由是本项目适配。未确认到作者发布的可直接复用训练仓库，不把其他同名MultiAMP库当作该论文代码。
2. [Multi-critic Learning for Whole-body End-effector Twist Tracking](https://arxiv.org/html/2507.08656)：CoRL 2025，Hutter为共同作者。一个Actor、三个奖励组的Critic：行走、操作、接触时序。分别计算GAE、归一化advantage，再组合优化PPO。官方[项目页](https://multi-critic-locomanipulation.github.io/)没有链接训练源码；该方法是按**奖励组**拆Critic，不是按互斥技能码只选择一个Critic。
3. [Parkour in the Wild](https://arxiv.org/html/2505.11164)：Hutter共同署名，多地形专家→DAgger统一策略→RL微调。更匹配后续多教师整合；当前学生暂缓，不直接切换训练路线。

Hutter团队的高星标底座是[RSL-RL](https://github.com/leggedrobotics/rsl_rl)，此次API读到3077星；本项目已经通过mjlab使用它。高星标基础PPO本身不意味着已实现多任务价值／奖励分离。

## 真实开源、基于RSL的多技能项目

下列扩展项目不是Hutter本人署名工作，但明确依托RSL-RL，适合作为可读实现。星数和SHA存入 `outputs/references/rsl_multiskill_survey/repositories.json`。

| 项目 | 此次星数 | 可借鉴机制 | 与当前阶段的关系 |
| --- | ---: | --- | --- |
| [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) | 1467 | 单策略接受速度、步态、身高／姿态等行为命令；按指令区域与成功阈值扩展课程 | 优先参考命令条件、能力覆盖和课程；多步态不等于已解决蛇↔双足的大构型切换 |
| [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour) | 1137 | 分技能策略与互斥专家路由、TPPO／DAgger等蒸馏链路 | 后续技能合并的主要代码参考；不把多专家路由器本身称为单部署网络 |
| [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour) | 1186 | 多障碍感知跑酷、课程与特权输入到视觉策略的训练流程 | 可借鉴课程和泛化验收；不是按任务分Critic的直接参考 |
| [Deep Whole-Body Control](https://github.com/MarkFzp/Deep-Whole-Body-Control) | 407 | 腿／臂价值估计、混合优势及协调课程 | 与奖励／学习信号解耦最贴近，优先读算法；不同动作子空间不能机械改成四种互斥任务 |
| [HugWBC](https://github.com/InternRobotics/HugWBC) | 168 | 无需动作库的人形统一全身行为命令、训练代码 | 人形任务条件控制参考；星数较少，不能称为千星成熟方案 |

这些仓库多数依赖旧版Isaac Gym及旧的 `ActorCritic/init_storage` 接口。当前环境为mjlab、新版TensorDict＋独立MLPModel，不能直接替换安装目录或复制旧PPO覆盖。

## 已读到的具体实现位置

- Deep Whole-Body Control：`rsl_rl/rsl_rl/modules/actor_critic.py` 中 `critic_leg_control_head`、`critic_arm_control_head`；算法 `ppo.py` 的 `mixing_advantages_batch` 与 `mixing_schedule`。腿／臂同时执行并有分动作的logprob，这是动作因子化／优势混合，不是我们当前六关节共用动作的直接同构实现。
- Walk These Ways：`go1_gym/envs/base/curriculum.py` 的 `RewardThresholdCurriculum`；训练代码为从RSL派生的 `go1_gym_learn/ppo_cse/ppo.py`，包名已改，不应因为顶层没有rsl_rl目录就判定无关。
- Robot Parkour Learning：`rsl_rl/rsl_rl/algorithms/tppo.py`、`modules/actor_critic_mutex.py`、`runners/two_stage_runner.py`。TPPO实际处理teacher action、蒸馏损失和teacher动作概率；互斥路由与最终学生策略应分清。

已按各仓库固定SHA读取9个关键文件用于比较，清单带原始URL和SHA256，存于 `outputs/references/rsl_multiskill_survey/source_manifest.json`；没有把它们接入当前训练路径。未运行这些旧Isaac Gym工程的完整训练。

## 最接近当前RSL接口的现成扩展

RSL官方仓库中的社区[Issue #223](https://github.com/leggedrobotics/rsl_rl/issues/223)及[PR #224](https://github.com/leggedrobotics/rsl_rl/pull/224)给出了多Critic PPO、storage、runner和测试。贡献者说明其曾在mjlab的G1上做两个奖励组验证，但这是贡献者报告，不能当作本项目实测。

本次通过GitHub API确认：PR仍open、未合并；head `fa0b057a22011e0d7c08809f9f292edeffd6ab7f`，来自 `ManuelActisCassa/rsl_rl`。其不是论文作者官方实现，也不能把主仓3077星转移算在该扩展头上。

参考源、许可证和来源哈希已取到 `outputs/references/hutter_multicritic_pr224/`。需要适配检查的地方：

- 它要求reward tuple、多个独立Critic、不同storage及runner；当前mjlab wrapper返回标量reward，OmniTeacherRunner还有旧checkpoint元数据和play布局适配。
- 原PR在 `process_env_step` 中更新normalizer，而当前安装PPO在优化结束后更新；不能在动作概率回算时不加检查地改变输入变换。
- 原PR分别clip每组PPO surrogate后取均值；论文描述为先组合归一化advantage再计算PPO。因为clip非线性，两者不严格等价，适配时必须明确选择并测试。
- 按奖励组拆Critic，与按FSM任务路由Critic是两种设计。不能把四类互斥任务的数据当成四个同时活跃奖励组，尤其不能给不活跃组随意中心化后制造非零训练信号。

## 本项目的具体采用顺序

本次选定 **Hutter多Critic论文作为学习信号分离依据，Deep Whole-Body Control作为已公开算法对照，Walk These Ways作为任务条件／课程参考**。Robot Parkour Learning保留到后续蒸馏阶段。

1. 保留当前纯双足run作为基准；另设明确区分的多技能实验，不覆盖其检查点和默认训练。
2. 首先确定奖励向量与任务mask：双足跟踪／步态、蛇运动、变形、共享正则各项保留身份，训练端再决定分组；每项必须恰好归属一组，分组求和应等于原标量reward。
3. 在现有RSL接口上实现分组价值、各组GAE／有效样本归一化与单Actor更新；明确使用paper-style“组合advantage后clip”。此项不是此前失败的仅对标量advantage按task分组归一化。
4. 观测归一化另做独立消融；不把一次冻结策略换统计的回放当作必须采用分任务观测统计的证明。
5. 核查终止／timeout的每组bootstrap、全零／未激活组、任务切换、只有双足时与标准PPO的一致性、保存／恢复／旧bests play。只做小规模程序验证，完整训练仍由用户执行。

本轮完成的是参考实现获取、来源核实及适配调查，不宣称已把PR接入生产训练或验证了多技能收敛。

## 已完成的本地接口验证

PR参考版使用本地现有RSL模型，完成16个合成CPU环境×8步、56维输入／6维Beta动作／3个Critic的一次PPO更新（1 epoch、2 minibatch）。损失有限、Actor参数发生变化、参考格式Actor保存回载输出严格相等。没有跑机器人训练，不代表多技能收敛。

检查发现其strict回载没有校验Critic数量：3个Critic会静默接受只含1个Critic状态的checkpoint。其独立clip与论文组合advantage后clip也由数值反例确认不等价。已将这些适配点与normalizer时序检查记录在 `outputs/references/hutter_multicritic_pr224/COMPATIBILITY_NOTES.md` 及 `compatibility_probe_results.json`。因此取用的是可审阅参考源，并未安装为当前训练后端。
