# 蛇形三轴速度奖励自适应权重

新增独立任务 `Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher`。原全向任务仍使用固定权重，不受修改。

```bash
bash scripts/python.sh train.py Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher \
  --env.scene.num-envs 8192 --agent.max-iterations 6000

bash scripts/python.sh play.py Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher \
  --env.scene.num-envs 1
```

使用独立日志目录 `logs/rsl_rl/zbot_snake_site_adaptive_teacher/` 和检查点角色。需新开训练，固定权重任务的检查点不会被误作自适应任务加载。

## 调整规则

1. 将配置中的前向、侧向、转动权重之和作为固定总预算B，三轴初始均分为B/3。基准权重可为0，但不得为负，总和必须为正；当前原任务1/1/0对应B=2，新任务初始各2/3，明确启用yaw奖励。
2. 使用PPO采集的**原始未归一化61维观测**估计跟踪能力：取生效指令和实测head速度。它是动作前观测中的跟踪状态，不是同一步奖励的精确复算。
3. 每轴仅统计绝对指令不小于该轴速度上限5%的有限样本，排除停车、小指令和非有限样本。每轮至少128样本才更新该轴EMA；从未有效覆盖的轴不参与强弱比较，直到三轴都覆盖过。
4. 根据原始速度得分 `r=clamp(1-2*abs(c-v)/abs(c),-2,1)` 估计能力 `p=(r+2)/3`。每轴能力EMA系数0.05，首个有效批次直接初始化。静止对应原始分数−1、能力1/3，不是能力0。
5. 目标权重为 `B*(0.5/3 + 0.5*softmax(-2*p_ema))`。弱轴获得更大权重；权重总和恒为B，单轴限制在均值的0.5—2倍范围。权重再以0.05的系数缓慢靠近目标，不突变。
6. 在PPO更新前归约观测统计；完成PPO更新后才写入权重，从下一轮rollout生效。不会在同一轮内改变奖励公式或重算已采集的奖励。全部停车、没有有效轴样本的轮次冻结权重。

只调整三个速度项。alive、姿态、关节限位、action_rate和失败惩罚的权重保持各自配置。

## 观测与恢复

控制台逐轴显示 `Adaptive vx/vy/wz weight (used -> next)` 和未加权分数EMA。TensorBoard记录 `Loss/velocity_weight_used_*`、`velocity_weight_next_*`、`velocity_score_*`、`velocity_score_ema_*`、`velocity_samples_*`。不要把加权奖励增长当作实际能力增长；应同时看未加权分数、head实测速度和COM位移。

检查点保存预算、当前权重、能力EMA、轴覆盖状态和更新计数。完整续训恢复全部状态；只加载Actor进行新的训练时保留新控制器初态。回放恢复保存的权重，冻结不再自适应。校验在模型加载前完成，避免坏控制器状态造成部分加载。资格检查点保存延后到当轮自适应状态提交后。

## 限制

这是有界的能力反馈规则，不保证三项奖励单调或同步增长，也不是GradNorm或多Critic算法的复现。三轴难度、互相干扰及指令曝光量仍会影响结果；左右共用同一个vy权重，不能保证左/右分别均衡。

平移跟踪点仍是foot_0_sole，参考系仍随head转动。此前实测表明head速度可接近目标而COM净推进很弱；自适应权重不会修复这一度量问题，应继续独立检查COM速度和净位移。本次只做实现与有限验证，未运行训练收敛实验。

## 本次验证

106项相关测试通过，Ruff、Pyright通过。CUDA下32环境、48控制步的有限仿真检查通过（PPO更新次数0）：预算2，初始权重各0.666667，两次统计更新后约为0.659010/0.663962/0.677028，总预算保持不变；运行时奖励权重、完整保存加载和回放权重恢复均通过。记录位于 `outputs/diagnostics/adaptive_velocity_weights/report.json`，该记录不代表学习效果。

实现复用原PPO和速度奖励；新增 `adaptive_site_snake.py` 控制器/runner，`experiments/site_snake.py` 与包入口注册独立任务，`training_console.py` 增加按轴权重与分数显示。未添加依赖。
