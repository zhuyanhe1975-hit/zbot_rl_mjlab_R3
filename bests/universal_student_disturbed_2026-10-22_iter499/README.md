# 22.d 外部扰动蒸馏学生 iter499

任务：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-Disturbed`。源运行：`logs/rsl_rl_universal_student_disturbed/zbot_universal_snake_leg_omni_student_disturbed/2026-10-22_18-06-59_universal_teacher_dagger_disturbed_22d`。

保存最新完整运行的最终 `model_499.pt`，保存状态记录完成500次DAgger更新；这次运行显式训练500轮，虽然22.d卡片默认150轮。按最新结果归档，未经独立性能排名择优。

## 模型与外扰

学生使用五帧传感器历史：每帧56维，共280维；任务编码后每个Actor分支输入305维。四任务独立Gaussian策略，动作自由度6/3/6/3，变形镜像展开为六关节动作。训练为DAgger v2：冻结全能教师，20轮教师预热、100轮逐步交接，之后学生独立执行并继续学习教师动作标签；环境奖励仅监测，不做PPO或Critic优化。

训练基座各轴力[-5,5]N、力矩[-0.5,0.5]N·m，持续0.08～0.20秒、间隔0.8～2秒，约25%环境受扰；预热系数1，从教师执行阶段起生效。当前归档回放沿用22.d的100%受扰展示，避免单环境随机抽成静默组。

## 回放

```bash
./bests/test.sh universal_student_disturbed_2026-10-22_iter499
./bests/test.sh universal_student_disturbed_2026-10-22_iter499 --env.scene.num-envs 8
```

W/S前后、A/D侧移、Q/E转向、X停走，`[`请求蛇形、`]`请求双足，Enter复位。回放只加载学生Actor，无需加载全能教师；完整DAgger续训需要冻结教师归档及匹配配方。

## 归档内容和验证边界

- `model_499.pt`、`env.yaml`、`agent.yaml`为原文件逐字节复制，未按当前默认配置重写。
- TensorBoard原事件、`training_metrics.json`记录训练统计；指标不等于独立技能验收。无扰动学生的摘要只取检查点编号以内的事件，原始事件文件可包含后续采样。
- `training_git/`为训练程序记录的Git差异；`workspace.patch`为归档时已跟踪源码差异，`source_snapshot.tar.gz`同时包含当前未提交源码、脚本、文档、测试和机器人资源，不是训练启动时完整工作区，也不包含其他归档模型或外部MJLab环境。
- `metadata.json`记录来源、动作/观测契约及实际更新数；`verification.json`记录真实有界回放；`SHA256SUMS`覆盖全部保留文件（自身除外）。

归档验证只保证严格加载、原文件一致性和短程有限值回放，不代表全方向速度跟踪、连续构型切换、抗扰性或实机性能已全部达标。原训练目录保留，检查点未重命名或改写。
