# 22.a 无外扰蒸馏学生 iter150

任务：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student`。源运行：`logs/rsl_rl_universal_student_v2/zbot_universal_snake_leg_omni_student/2026-10-22_13-45-40_universal_teacher_dagger_v2`。

按用户指定保存同目录的 `model_150.pt`。检查点编号150，保存状态记录已完成151次DAgger更新；原始启动配置上限500轮，不代表该检查点完成500轮。

## 模型与外扰

学生使用五帧传感器历史：每帧56维，共280维；任务编码后每个Actor分支输入305维。四任务独立Gaussian策略，动作自由度6/3/6/3，变形镜像展开为六关节动作。训练为DAgger v2：冻结全能教师，20轮教师预热、100轮逐步交接，之后学生独立执行并继续学习教师动作标签；环境奖励仅监测，不做PPO或Critic优化。

训练继承22.a：外扰预热系数为0，纯DAgger不推进PPO外扰计数，因此训练时基座冲量未启用。默认归档回放通过`playback_args.json`将基座受扰比例设为0，保证无外扰对照；其他物理随机化仍保留原任务定义。

## 回放

```bash
./bests/test.sh universal_student_clean_2026-10-22_iter150
./bests/test.sh universal_student_clean_2026-10-22_iter150 --env.scene.num-envs 8
```

W/S前后、A/D侧移、Q/E转向、X停走，`[`请求蛇形、`]`请求双足，Enter复位。回放只加载学生Actor，无需加载全能教师；完整DAgger续训需要冻结教师归档及匹配配方。

## 归档内容和验证边界

- `model_150.pt`、`env.yaml`、`agent.yaml`为原文件逐字节复制，未按当前默认配置重写。
- TensorBoard原事件、`training_metrics.json`记录训练统计；指标不等于独立技能验收。无扰动学生的摘要只取检查点编号以内的事件，原始事件文件可包含后续采样。
- `training_git/`为训练程序记录的Git差异；`workspace.patch`为归档时已跟踪源码差异，`source_snapshot.tar.gz`同时包含当前未提交源码、脚本、文档、测试和机器人资源，不是训练启动时完整工作区，也不包含其他归档模型或外部MJLab环境。
- `metadata.json`记录来源、动作/观测契约及实际更新数；`verification.json`记录真实有界回放；`SHA256SUMS`覆盖全部保留文件（自身除外）。

归档验证只保证严格加载、原文件一致性和短程有限值回放，不代表全方向速度跟踪、连续构型切换、抗扰性或实机性能已全部达标。原训练目录保留，检查点未重命名或改写。
