# 蛇形世界X推进：第1499次迭代

来自 `logs/rsl_rl/zbot_snake_omni_teacher/2026-10-19_16-45-28_beta_body_v3/model_1499.pt`，保存最终检查点，未在所有迭代中重新择优。

```bash
./bests/test.sh snake_world_x_2026-10-19_iter1499 --env.scene.num-envs 1
```

整机质心世界X速度线性奖励在0.2 m/s封顶，世界Y绝对速度成本；头部与体轴均有相对初始yaw的瞬时和有符号积分惩罚。Actor/Critic为51/66维，原生Beta六维动作积分为关节目标角，目标范围±60°，最大目标角速度360°/s。

`env.yaml`、`agent.yaml`是原始训练启动配置。此模型实际保存的动态倍率为 `S=clamp(0.25+5*max(v_world_x,0),max=1)`，不是此前讨论过的0下限版本。头部与体轴yaw积分奖励权重均为-0.5；阈值分别为60°和30°。播放时配置应与`metadata.json`中的检查点recipe一致。

最终TensorBoard标量：世界X速度约0.149914 m/s，世界Y速度约0.006568 m/s，平均回合长度1484.29控制步（约29.686秒），平均回报28.200。以上是训练统计，不等同于独立确定性策略评估。

- `metadata.json`：来源、迭代、检查点语义及原文件SHA256。
- `training_metrics.json`：TensorBoard各标量的最后一次记录及步号。
- `source_snapshot.tar.gz`：当前源码、机器人资源、运行脚本和任务说明，含未提交代码；源码是归档时工作区版本。
- `workspace.patch`：归档时相对Git HEAD的已跟踪文件改动；未跟踪源码包含在快照中。
- `playback_args.json`：回放入口固定训练时的关节范围和航向、惩罚倍率等参数，不改当前训练默认配置。
- `verification.json`：归档后加载及回放结果。
- `SHA256SUMS`：当前归档文件校验和。

独立确定性回放：CUDA、8环境、默认世界+X初态、30秒、关闭自动复位，失败0/8。平均世界X速度0.082800 m/s，平均绝对世界Y速度0.050901 m/s，30秒平均前进约2.496 m，存在约1.128 m的单侧净漂移。仅验证固定默认初态，未验证全随机朝向、扰动或实机。

归档兼容性测试24项通过，严格Actor加载通过。复现本次回放：

```bash
./scripts/python.sh bests/snake_world_x_2026-10-19_iter1499/verify.py
```

原始训练日志保留原位。
