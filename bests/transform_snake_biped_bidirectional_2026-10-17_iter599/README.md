# 蛇足双向变形：最新训练 iter599

此入口已按用户要求替换旧归档，目录和模型文件名保持不变。

- 任务：`Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional`。
- 来源：`logs/rsl_rl/zbot_transform_snake_biped_bidirectional/2026-10-17_21-39-14/model_599.pt`，600轮，从头训练。
- 用户反馈：本次训练结果好于之前归档；此为用户评价，未做同条件统计优越性检验。
- Actor37维 / Critic45维，3维镜像Beta动作，6个执行器目标。

## 当前训练配置

五项基础权重为构型12、高度1、首次上升速度8、脚底朝下1、失败−10。
两项增量为：base高度≥0.23 m后关节实际速度平方均值成本，权重−0.02；趴下指令时超过2倍整机重量的峰值接地力成本，权重−2。峰值覆盖全部物理子步。

默认`source` 模式在目标附近稳定2 s后原地反向。自动切换关节容差0.15 rad，严格成功仍为0.10 rad；其余接地和静止条件保留。切换保留物理姿态和执行器目标；失败和16 s回合超时仍重置。

## 验证范围

严格加载通过；种子 42、8环境、18 s 无界面回放完成23次自动切换（站立→爬下12次，爬下→站立11次），第800步8环境按16 s回合上限重置。
已检查数值有限、37/45维观测、新指令观测与旧指令奖励顺序、切换瞬间的物理状态及执行器目标连续性。PPO更新次数为0。

这是功能验证，未独立统计严格成功率、触地冲击改善量、抗扰动或硬件安全性；自动切换次数不能视为严格成功次数。

## 播放

仓库根目录执行：

```bash
./bests/test.sh transform_snake_biped_bidirectional_2026-10-17_iter599 --env.scene.num-envs 1
```

点击MuJoCo窗口后，`]`站立、`[`爬下。按键进入手动锁定并关闭自动重置；Enter重置并恢复自动演示。未按键时默认自动反向。

## 保存内容

`model_599.pt`含Actor/Critic、优化器及训练计数。`env.yaml`、`agent.yaml`原样保留训练启动配置，`final_effective_weights.json`来自模型检查点的最终权重元数据。
完整 TensorBoard 日志、训练差异、源码和资产快照、回归测试包已移到项目外原始归档，位置见上级 `归档位置.json`。原始 `evaluation_probe.py` 也在该归档内；外部 MJLab 和 Python 环境不随归档提供。

详细结果见`evaluation.json`及`metadata.json`；文件完整性见`SHA256SUMS`。

原归档替换时全量回归 668 项通过（含严格加载与执行器兼容检查），原验证日志位于项目外归档的 `verification/pytest.txt`。此次清理的验证结果见 `docs/cleanup-report.md`。
