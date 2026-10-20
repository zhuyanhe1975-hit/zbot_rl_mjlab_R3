# 蛇全向 site 教师：最终第 599 号模型

来源：`logs/rsl_rl/zbot_snake_site_omni_teacher/2026-10-20_19-07-44/model_599.pt`。
已完成 600 次 PPO 更新。本归档保存用户指定的最终模型，未从历史迭代重新择优。

```bash
./bests/test.sh snake_site_omni_2026-10-20_iter599 --env.scene.num-envs 16
```

Actor/Critic 为 56/56 维，六维有界 Beta 策略。参考点和指令坐标系均为
`foot_0_sole`，不使用整机质心作为速度跟踪点。关节目标范围 ±60°，目标积分速度
上限 2π rad/s；不是仿真实际关节速度的硬截断。

## 实际训练配方

启动配置原样保存在 `env.yaml` 和 `agent.yaml`，不要用旧聊天记录替代：

| 奖励项 | 权重 |
|---|---:|
| track_linear_velocity | 4 |
| track_angular_velocity | 1 |
| site_tilt | −1 |
| joint_pos_limits | −1 |
| joint_target_limits | −1 |

线速度指数宽度 `min(0.1, 0.02+0.5*norm(v_cmd_xy))` m/s；角速度宽度 0.25 rad/s。
指令平移上限每轴 ±0.2 m/s，yaw 上限 ±0.5 rad/s。实际关节和目标角超过 ±54°
开始软限位扣分；site 倾角超过 60° 或自碰撞时重置，不设超时重置。
每轮终端显示总奖励率和全部五项加权贡献，100/500 轮定期报告关闭。

## 已有证据及边界

严格 Actor 加载通过，CUDA 独立确定性评测已完成：72 个指令/速度/初始朝向组合，
每个场景 10 秒，不自动重置。**仅停止场景通过，严格全向合格结论为未通过（1/9 类）。**
前后、侧移、转向及混合指令仍未满足严格跟踪要求，不能将本模型称为合格全向控制器。
各场景实际速度、持续跟踪比例及失败记录见 `verification.json`；尚未实机验证。
这不影响将该版本作为实验进度归档，避免丢失可回放的结果。

`training_metrics.json` 是原 TensorBoard 每个标量的最后一次记录，包含步号。
例如最终采样窗口中线速度奖励约 2.8186/s、角速度约 0.4845/s，
总奖励率约 3.1475/s（包含三项惩罚）。训练奖励不能替代上述独立评测结论。

## 文件与复现

- `model_599.pt`：Actor、Critic、优化器及任务元数据。
- `env.yaml` / `agent.yaml`：原始训练启动配置。
- `metadata.json` / `runtime.json`：来源、配方、检查点 SHA256、运行库版本。
- `training_metrics.json` / `verification.json`：训练标量与独立评测证据。
- `source_snapshot.tar.gz` / `source_manifest.json`：当前源码、资源、测试和脚本快照与逐文件 SHA256。
- `workspace.patch`：快照时相对基础提交的已跟踪文件差异，未跟踪源码包含在源码快照中。
- `SHA256SUMS`：归档文件校验和（自身除外）。

```bash
(cd bests/snake_site_omni_2026-10-20_iter599 && sha256sum -c SHA256SUMS)
./scripts/python.sh bests/snake_site_omni_2026-10-20_iter599/verify.py
```

验证脚本将新结果写到 `outputs/archive_verification/`，不覆盖归档证据。
严格评测未通过时按约定返回退出码 1。原训练目录、其余检查点及 TensorBoard 文件保留原位。
