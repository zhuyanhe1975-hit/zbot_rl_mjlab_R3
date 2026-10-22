# 速度与频率联合跟踪：iter499

来源：`logs/rsl_rl/zbot_velocity_tracking/2026-10-22_09-00-05/model_499.pt`。完成500轮训练，使用09:00:05启动的最新运行结果替换原归档；保存最后检查点，未按评分重新择优。

任务：`Mjlab-Zbot-6dof-Velocity-Frequency-Tracking-Walking`。8192环境，训练步频0.25–1.0 Hz、前向速度指令0–0.5 m/s；关节加速度惩罚权重−0.001。归档回放参数固定训练时的步频编码范围及速度指令范围。

```bash
./bests/test.sh velocity_frequency_tracking_2026-10-22_iter499 --env.scene.num-envs 16
```

- `model_499.pt`：模型与优化器状态，与原文件SHA256一致。
- `env.yaml` / `agent.yaml`：原始训练启动配置。
- `training_metrics.json` / TensorBoard事件文件：训练标量，不能代替独立性能评估。
- `metadata.json`：来源、迭代与归档边界。
- `playback_args.json`：归档回放参数。
- `training_git/`：训练程序保存的Git差异记录。
- `source_snapshot.tar.gz` / `workspace.patch`：归档时源码、资源和工作区改动；不是训练启动时的完整工作区，未打包外部MJLab或Python环境。
- `verification.json` / `SHA256SUMS`：验证记录及文件校验和。

未运行独立仿真性能评估，不宣称速度或步频跟踪精度、抗扰能力或实机表现达标。原训练日志保留原位。
