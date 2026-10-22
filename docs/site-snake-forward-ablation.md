# 纯前进奖励对照

诊断目标：全向模型在前进指令下近乎静止，区分任务混合干扰与瞬时侧向/转动代价对探索的抑制。

两组从头训练，固定蛇头系指令 `(0.1, 0, 0)` m/s、m/s、rad/s。保留原先 61 维观测、六关节速度积分、±π/2 目标限位、2π rad/s 最大目标速度、物理模型、姿态/限位/失败惩罚及 PPO 参数。原全向任务不改动。

| 组别 | 任务后缀 | 前向权重 | 侧向权重 | 转动权重 |
|---|---|---:|---:|---:|
| A | `Snake-Site-Forward-Teacher` | 4 | 4 | 1 |
| B | `Snake-Site-Forward-Relaxed-Teacher` | 4 | 0 | 0 |

A 将原来的线速度和拆成前向、侧向两项，总奖励与原公式一致。B 仅取消侧向、转动代价；这两项的原始值继续作为指标记录。两组都不提供周期动作或参考步态。

固定指令由命令采样器处理，reset 和重采样始终恢复前进。手动控制仍有优先权。训练保留随机初始 yaw，Play 默认也是前进。两组使用不同日志目录与 checkpoint role/reward_recipe，拒绝交叉加载或加载原全向模型。

## 复现短训

```bash
bash scripts/python.sh train.py Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher \
  --env.scene.num-envs 1024 --agent.max-iterations 300 --agent.seed 42 \
  --agent.run-name forward_baseline_seed42 --log-root outputs/site-forward-ablation

bash scripts/python.sh train.py Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher \
  --env.scene.num-envs 1024 --agent.max-iterations 300 --agent.seed 42 \
  --agent.run-name forward_relaxed_seed42 --log-root outputs/site-forward-ablation
```

300 轮、每轮 24 步、1024 环境，每组共 7,372,800 个环境步。这是单种子的短训诊断，不代表收敛结论。

## 固定前进独立评测

```bash
bash scripts/python.sh scripts/evaluate_site_forward.py \
  --task Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher \
  --checkpoint <A组检查点> --output outputs/site-forward-ablation/baseline-evaluation.json

bash scripts/python.sh scripts/evaluate_site_forward.py \
  --task Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher \
  --checkpoint <B组检查点> --output outputs/site-forward-ablation/relaxed-evaluation.json
```

确定性策略，72 环境均分为初始 yaw 0/90/180/270°，先停车稳定 1 秒，再前进 10 秒；不自动重置。最后 5 秒记录 head vx/vy/wz、限位比例、平均关节速度、前向奖励与按 A 权重计算的侧向/转动代价。额外记录整个 10 秒的质心和 head 沿初始朝向的净位移，避免仅凭瞬时 head 速度误认推进。失败样本保留在统计中，并单列失败率。

`scripts/evaluate_site_snake.py --task <任务>` 仍可用于原 72 个全向场景的迁移评测；它要求全向全部通过，不能替代这里的固定 0.1 m/s 前进对照。

解释边界：如果 B 明显推进而 A 停止，支持“即时侧向/转动代价抑制推进探索”；如果 A/B 都推进，则任务混合是候选原因；如果两组都不推进，还需检查动作表达、接触动力学和周期协调。B 的侧移或转动变大属于放松代价的预期风险，不直接作为可部署策略。

## 本次运行状态

按用户要求中断本次启动的两组后台训练，保留 `outputs/site-forward-ablation/` 的日志、检查点与早期评测；未完成统一最终评测，不作300轮结果对比结论。主全向任务目前前向权重4、侧向权重1；A/B 对照显式固定4/4/1与4/0/0，不随主任务调权变化。

相关回归281项通过，Ruff、Pyright通过；主任务16环境、2轮PPO及前向4/侧向1命令行调权冒烟通过。

后续策略已切换为G1式高斯（初始标准差1、熵系数0.01），A/B子任务同步继承。此处保留的早期中断检查点属于旧Beta策略，不能直接恢复为高斯训练；新对照需重新启动。
