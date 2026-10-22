# 学生 PPO v2 诊断与 v3 实验

本次保留 v2 卡片、默认任务及 22.b 启动入口；v3 是独立注册的
`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO-V3`。
这是基于日志的优化假设和可运行实现，尚未用长训练证明收敛改善。

## 可复核证据

来源：`logs/rsl_rl_universal_student_ppo_v2/zbot_universal_snake_leg_omni_student_ppo_v2/2026-10-22_15-03-41_student150_ppo_v2/`。
使用 Python TensorBoard EventAccumulator 读取全部 **744 个 scalar tag** 和 **12 份 validation JSON**。
导出目录：[student_ppo_v3_20261022](../outputs/diagnostics/student_ppo_v3_20261022/)，含全部原始标量
`scalars.csv`、窗口均值 `curve_windows.csv`、全部验证指标 `validation.csv`、`metadata.json` 和 [曲线图](../outputs/diagnostics/student_ppo_v3_20261022/curves.png)。

重跑诊断（只读原日志）：

```bash
./scripts/python.sh scripts/diagnose_student_ppo.py \
  logs/rsl_rl_universal_student_ppo_v2/zbot_universal_snake_leg_omni_student_ppo_v2/2026-10-22_15-03-41_student150_ppo_v2 \
  --output outputs/diagnostics/student_ppo_v3_20261022
```

下表是 TensorBoard step 的窗口均值，区间右开。validation 的编号是 completed_updates，通常比 TB step 大 1。

| 指标 | 0–9 warmup | 10–34 | 125–149 | 275–299 |
|---|---:|---:|---:|---:|
| Train/mean_reward | -1.855 | 2.245 | 6.423 | 6.370 |
| Loss/value | .1671 | .0562 | .0442 | .2375 |
| Loss/surrogate | ≈0 | -.00171 | -.00215 | -.00139 |
| biped reward_rate | 1.876 | 1.900 | 2.323 | 2.290 |
| snake reward_rate | 1.679 | 1.542 | .561 | .532 |
| ascent reward_rate | -3.032 | -2.191 | -1.544 | -2.866 |
| reference coefficient | .05 | .0473 | .02143 | .005 |
| weighted reference loss | 0 | .00422 | .05377 | .03577 |
| reference KL task0 | — | .0451 | 1.194 | 5.183 |
| reference KL task2 | — | .1379 | 3.849 | 9.126 |
| snake frame fraction | .249 | .322 | .330 | .339 |
| ascent frame fraction | .194 | .168 | .188 | .250 |

旧日志没有 PPO old/current KL，也没有 critic value prediction、return 或 explained variance 曲线。
不能把 reference KL 当 PPO KL，也不能只凭 value loss 判定 critic 校准。
Episode_Reward 是跨任务、按回合汇总的已加权项，不能当蛇形专属分项或奖励消融结果。

## 问题：证据 → 结论

1. **弱基线加遗忘，而非原本完整技能被一次性毁掉。**
   baseline 的 snake_moving_success=.125 完全来自 yaw_right；所有蛇形平移 success 原本就是 0。
   snake_forward mean_vx=.0139→约0、峰值关节速度 2.208→.303 rad/s；backward mean_vx=-.0012→-.0137，仍远未跟上目标。
   yaw_left 在 185/210/235 为 1，yaw_right 在 0/35 为 1，因此“蛇形所有运动全程失败”不准确；260/285 才全部运动失败。
   biped_backward mean_vx=-.0958→-.0135、success=.75→0；35 时速度仍-.0881但 success 已0，说明串扰/稳定性/持续性也会使严格判据失败。
   mixed_negative 的 success 基线就是0，不能说它从成功退化。所有 validation_passed 均0。

2. **静止存在大量局部跟踪信用，但并未证明静止超过完美运动的总回报。**
   [student_rewards.py](../src/zbot_rl_mjlab/tasks/universal/student_rewards.py) 的蛇形分数为
   `2*(mean(exp(-e²)) + prod(exp(-e²)))`，std=(.05,.05,.15)。
   单轴目标下，两条未请求轴保持零就贡献约4/3；静止面对 vx=.04 得2.739、vx=.08 得1.539，而完美跟踪4。
   初始半幅课程放大了静止信用；离目标很远时高斯核对微小速度改善不敏感。
   蛇形正则还包括 action-rate、vertical-speed、joint-deviation、target-saturation。
   但 [guided_rewards.py](../src/zbot_rl_mjlab/tasks/universal/guided_rewards.py) 的 energy 只惩罚双足，standstill只在零命令启用，不能归因于“蛇形能耗过大”或“移动命令错误获得standstill”。
   总奖励上升而蛇形奖励率下降，表明不能用总奖励选择多技能模型。

3. **蛇形已有参考，实际约束逐渐不足；ascent/descent 没有参考。**
   [task_ppo.py](../src/zbot_rl_mjlab/task_ppo.py) 中任务0/1/2/3分别是双足/下降/蛇形/上升。
   v2 `[0,2]` 覆盖蛇形，冻结的是学生，非教师。每次20个PPO minibatch更新后只做一次参考SGD修正，参考学习率.001，系数.05在200次policy update后降到.005。
   系数下降伴随 task0/2 reference KL 持续升高，weighted loss后期反而变小不代表保留更好。
   clip=.1只约束当前rollout surrogate，不会限制300代相对初始学生的累积漂移。

4. **不是缺少逐任务优势归一化，也不是蛇形运动样本太少。**
   TaskPPO 已有四个独立critic、逐任务GAE与优势归一化；学生四个actor头也独立，没有共享可训练trunk。
   环境角色初始各25%；实际帧比例因handoff/continuation变化，蛇形后期约34%，并未饿死。
   snake课程7→19→27符号类，stop仅1/7→1/19→1/27，但后期每个纯轴也只占1/27。
   双足三轴连续均匀 `[-.3,.5]×[-.2,.2]×[-.5,.5]`，负vx仅37.5%，精确纯轴概率为0；与验证低速纯轴及两种mixed分布明显不同。
   v3增加的是归一化之后按实际帧数平衡策略权重，不是重复优势归一化。

5. **critic warmup机制正确，但10代是否足够没有充分观测。**
   冷启动只转移actor与冻结归一化；critic/Adam重新初始化，warmup仅更新critic，且不推进policy课程。
   value loss早期下降、surrogate在warmup≈0，与实现相符；晚期value loss升至.238，符合策略/状态分布恶化，不能据此认定warmup是主因。
   v3保守改20代并补各任务return/value/raw advantage统计；它仍是固定长度，不是自动收敛判据。
   std全程.1、learn_std=false；entropy负值变化主要受不同任务动作维数/样本组合影响，不能说std塌缩。
   设大entropy_coef在固定std下不会增加探索。当前小噪声不保证能发现缺失的协调蛇形步态，v3未盲目增噪。

6. **ascent遗忘属实，惩罚不是硬限速，无法从日志唯一归因。**
   baseline handoff=.75，35/60/135/160为1，185/210为.75，235以后为0；峰值速度6.034→7.490。
   ascent没有reference；joint_standing权重12、ascent权重8是局部代理，不能保证最终保持、接触与姿态。
   `transition_joint_overspeed` 对六关节取均值：仅一个关节7.5rad/s时，limit3对应代价只有-.375。
   settle_speed在|qd|≥1时已饱和；settle三项还要求高度≥.23且最大姿态误差≤.35，失败姿态可能绕开该区域。
   TB的reset_failure从125–149均值-.00065变为275–299的-.02457；settle_target从-.00641变为-.02681，伴随ascent reward率与value loss恶化。
   这些证据支持累积漂移/代理目标错位，不能替代奖励消融确认某单项的因果作用。因此v3先保留变形代价，加参考和验证保护。

7. **验收比密集训练奖励严格，旧runner只是记录失败。**
   [universal_omni_evaluation.py](../src/zbot_rl_mjlab/universal_omni_evaluation.py) 检查各请求轴误差≤50%、未请求轴漂移限制、持续成功≥.6、ready≥.9等。
   validation_passed要求每个场景/初始yaw成功率≥.75；80env时每个yaw只有一个副本，整体通过是很粗的二元量。
   训练切换可用16秒，验证请求后8秒，慢切换也可能判失败；未放松这个口径。
   v2虽保存验证checkpoint，却没有据验证停止或选择best，因而退化能持续。

## v3具体变化与边界

- 新卡片继承v2（再继承原学生卡），同一150代checkpoint及SHA256冷启动，独立task/log/experiment名称。
- 四任务参考固定`.2→.2`，镜像变形只算三个独立控制维。仍是学生参考，没有加载/更新教师；PPO lr降至1e-5，其他clip/std/entropy保持。
- 保留逐任务GAE与归一化，opt-in inverse-frequency权重平衡实际rollout的policy贡献。
- 双足/蛇形分别循环平衡9类：stop、±x、±y、±yaw、mixed±。幅度为验收目标附近1–1.5倍并受原边界限制；关闭蛇形半幅ramp。
  这是一阶段能力恢复分布，覆盖范围小于完整原命令包络；后续需增加更大幅值/异号mixed评估，不能直接宣称全包络泛化。
- v3替换运动跟踪为实际有符号速度进展：运动命令静止0、完美跟踪4、反向/严重超速最低-4；请求轴均值与最差轴各半，未请求漂移扣分。stop仍单独奖励静止，transition为0。没有以关节抖动作为“运动”奖励。
- 每10次policy update验证。各方向、stop、切换最终成功与handoff任一比初始baseline下降超过.25即不合格，连续3次不合格停止；完整失败checkpoint照存。
  `model_best.pt`在合格候选中先最大化四技能最差成功率，再比较平均值；baseline也可为best。`validation_best.json`记录来源；resume恢复保护状态及可用的历史best文件。
- 增加各任务value/return/advantage/样本比例、权重和有界采样的old→post-PPO KL。KL在reference修正前测量，不能当作修正后最终策略KL。

## 训练与验收

先运行短程100代（20代critic+80代policy），不要仅因总奖励增长就延长：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO-V3 \
  --log-root logs/rsl_rl_universal_student_ppo_v3 \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 100 \
  --agent.run-name student150_ppo_v3 \
  --agent.resume False
```

`train.sh` 的22.c保留注释掉的300代示例。不要直接执行仍以22.b为活动命令的train.sh来启动v3。
看 `validation_best.json`、各 `validation_policy_*.json`，不要只取最后checkpoint。
首要观察snake_forward/backward/left/right是否首次成功、各轴RMSE和有符号均速是否接近命令；同时守住biped_backward、stop、ascent/descent success和handoff。
关注ascent峰值关节速度不再持续增长、handoff时间不恶化，以及reference KL不持续失控。
峰值速度3/2.5是奖励调参目标，当前实现并未提供硬安全约束。
最终仍以原验证所有方向与yaw通过为准，并扩到320env及8秒同口径复测：

```bash
./scripts/python.sh scripts/evaluate_student_ppo.py /path/to/model_best.pt \
  --num-envs 320 --seconds 8 --output outputs/student_ppo_v3_acceptance.json
```

该脚本使用共同v2验收环境，适合保持几何/FSM口径一致；v3 actor接口相同。
额外做12–16秒评估可区分“很慢”与“完全不会”，但不能拿长窗口结果替代8秒验收。
当前验收固定seed且无物理随机化；部署前仍需另做多seed/扰动验证。

更可靠的主线是先改善DAgger学生。至少看到蛇形各平移方向实际成功，再用PPO保留和优化；仅把150改成500不能保证解决教师/学生分布差异。
替换初始学生时同时改`initial_student_checkpoint`与其真实SHA256，使用新的run-name。
如v3仍被保护器停止，应检查失败方向并改进DAgger，不应直接关掉保护器长跑。
建议后续分别消融reference/命令采样/进展奖励/任务权重，当前组合实验不能单独量化各修改贡献。

## 本次验证

相关pytest **488 passed**（4条上游torch.jit弃用警告），修改文件ruff通过、pyright 0 errors/0 warnings；
`git diff --check`与`bash -n train.sh`通过。80env、3更新（1次critic+2次PPO/reference）、
完整8秒周期验证的冒烟在180秒超时内正常退出，日志和checkpoint保留在
`logs/student_ppo_v3_smoke/zbot_universal_snake_leg_omni_student_ppo_v3/2026-10-22_15-38-07_smoke_final_8s_3_updates/`。
另用共同v2环境对原始DAgger与冒烟checkpoint做了80env/8秒配对评估，记录在
`paired_smoke_evaluation.json`。这次新鲜基线的变形handoff与历史baseline JSON并非完全一致，
所以本报告的退化诊断使用历史同一轮曲线，不把跨运行baseline差异解释为v3收益。
平衡采样仅对training启用；测试锁住评估时与v2相同的命令/gait/RNG顺序。
完整验证命令和日志索引见
`outputs/diagnostics/student_ppo_v3_20261022/verification.json`。
冒烟只证明收集、critic预热、PPO、四任务参考、独立验证和checkpoint链路可运行，不证明学习收敛。
没有git commit，没有删除旧日志或checkpoint。

本次文件清单（保留工作区已有未提交修改）：

- `src/zbot_rl_mjlab/{__init__,student_ppo,student_ppo_selection}.py`
- `src/zbot_rl_mjlab/experiments/student_ppo.py`
- `src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_ppo_v3.json`
- `src/zbot_rl_mjlab/tasks/universal/{student_ppo_env,student_ppo_rewards}.py`
- `scripts/diagnose_student_ppo.py`、`train.sh`、本报告
- `tests/test_student_ppo.py`、`tests/test_student_ppo_evaluation.py`、`tests/test_student_ppo_selection.py`
- `tests/test_student_ppo_v3_motion.py`、`tests/test_student_ppo_v3_experiment.py`、`tests/test_task_contracts.py`
- `tests/data/experiment_configs.json.gz`：只增加v3快照和v2新增关闭开关的默认值；程序断言剔除新增字段后v2与旧快照完全相同，其他实验快照未改。
