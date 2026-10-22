# 按指定频率原地踏步：iter50

来源：`logs/rsl_rl/zbot_periodic_stepping/2026-10-22_07-58-11/model_50.pt`。归档最新一次训练的最后检查点，未按评分重新择优。任务为 `Mjlab-Zbot-6dof-Periodic-Stepping`，共配置51轮训练，步频0.25–1.0 Hz，8192环境。

此版本取消前进速度奖励，水平速度平方惩罚权重−5，步频跟踪权重3；旧 Walking 模型不能作为本实验结果。

```bash
./bests/test.sh periodic_stepping_2026-10-22_iter50 --env.scene.num-envs 8
```

- `env.yaml` / `agent.yaml`：原始训练启动配置。
- `model_50.pt`：模型与优化器状态；SHA256与原文件一致。
- `training_metrics.json`、TensorBoard事件文件：训练统计，不能代替独立回放评估。
- `metadata.json`：来源与归档边界。
- `source_snapshot.tar.gz` / `workspace.patch`：归档时源码、资源和未提交改动，非训练启动时的完整工作区；外部MJLab和Python环境未打包。
- `training_git/`：训练程序保存的Git差异记录。
- `verification.json`：CPU兼容性与严格策略加载检查结果。
- `SHA256SUMS`：归档文件校验和，自身不计入。

未运行独立仿真性能评估，不宣称步频精度、原地漂移或抗扰能力达标。原始日志保留原位。
