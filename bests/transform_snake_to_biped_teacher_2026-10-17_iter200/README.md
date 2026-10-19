# 随机滚转蛇形起立教师：第 200 次迭代

任务：`Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher`。
来源训练：`logs/rsl_rl/zbot_transform_snake_to_biped_teacher/2026-10-17_17-24-59`。保存的是最新已落盘模型 200，训练日志继续到 215 次迭代；未保存的最终权重不在此模型中。

已有独立确定性评估：16 环境、16 秒、种子 42、蛇形纵轴随机滚转 ±90°、初始关节角为零、无外部推力。严格成功 16/16，均由回合超时结束；最终足底倾角不超过 0.069°，最大关节误差不超过 0.037 rad，连续站稳至少 13.78 秒。这是仿真证据，未验证硬件或外部扰动。

Actor/Critic 观测分别为 36/44 维，3 维镜像 Beta 动作，最大关节目标速度 2π rad/s。五项奖励权重为构型 12、高度 1、首次上升速度 8、足底方向 1、失败 −10。基座历史峰值高度达到 0.23 m 后，本回合永久关闭上升速度项，无额外奖励阶段。

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher \
  --checkpoint-file bests/transform_snake_to_biped_teacher_2026-10-17_iter200/model_200.pt \
  --env.scene.num-envs 4
```

每次播放启动使用新种子，可加 `--env.seed 42` 复现。每次重置独立采样纵轴滚转，直接写入实际根四元数，蛇形纵轴仍保持水平。

启动配置保存在 `env.yaml`、`agent.yaml`，模型中的最终奖励元数据见 `final_effective_weights.json`，摘要见 `evaluation_summary.json`。完整评估轨迹、源码和资产快照、训练日志保存在项目外原始归档，位置见上级 `归档位置.json`。外部 MJLab 和 Python 环境不随归档提供。未来若动作、观测或物理定义已变化，不能仅恢复奖励权重来复现实验。
