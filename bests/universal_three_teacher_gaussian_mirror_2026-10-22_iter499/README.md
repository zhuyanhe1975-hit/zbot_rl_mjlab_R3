# 三教师全能教师：Gaussian镜像版 iter499

任务：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided`。

按用户确认，归档最新完整run `2026-10-22_12-28-29_three_teacher_gaussian_mirror_v1` 的最终 `model_499.pt`，完成500轮；不是按独立评分择优。全能Actor统一Gaussian，任务0/1/2/3的随机动作维数为6/3/6/3，变形严格镜像。原始观测56维、网络输入61维，四任务分支及分任务统计保留；双足参考base，稳定蛇形参考foot_0_sole。

## 回放

```bash
./bests/test.sh universal_three_teacher_gaussian_mirror_2026-10-22_iter499
# 多机器人回放
./bests/test.sh universal_three_teacher_gaussian_mirror_2026-10-22_iter499 --env.scene.num-envs 8
```

W/S前后、A/D侧移、Q/E转向、X停止，`[`请求蛇形、`]`请求双足，Enter复位。当前源码包含归档后的play同步修正：变形期间速度门控，到位后再切运动策略；选中机器人按键不改变其他机器人。回放不需要加载三个专家，完整续训仍依赖原三个bests专家与匹配配置。

## 保存内容与边界

- `model_499.pt`：模型、Critic、优化器与指导进度，逐字节复制原检查点。
- `env.yaml`、`agent.yaml`：训练启动时原始配置；不以归档时的新配置覆盖。
- `metadata.json`、`training_metrics.json`、TensorBoard事件：来源、动作/观测契约及训练统计。
- `training_git/`：训练程序记录的Git差异；`workspace.patch`与`source_snapshot.tar.gz`记录归档时当前源码，包含后来play修正和未提交文件，不宣称为训练启动时完整快照。源码快照排除其他bests、训练日志和外部运行环境。
- `verification.json`、`SHA256SUMS`：实际验证结果和全文件完整性清单。

归档验证只保证严格加载、文件一致性和有界有限值回放，不代表全部速度、构型切换及抗扰场景已独立达标。历史站起统计异常及诊断记录见项目文档，不据本次归档宣称问题已全部解决。模型原始训练目录保留。
