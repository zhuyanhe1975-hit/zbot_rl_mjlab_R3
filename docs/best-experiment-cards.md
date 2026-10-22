# bests 实验卡片差异说明

2026-10-22 对照 `bests/` 中17个归档的 `env.yaml`、`agent.yaml`、保存时奖励记录和回放参数，将相对父卡片的差异显式放在对应实验卡片。`description` 标明来源及历史差异；没有覆盖的字段继续继承父卡片。归档模型、原始配置和校验和不作改动。

卡片位于 `src/zbot_rl_mjlab/experiments/cards/`。

| 卡片 | 对应 bests 目录 | 显式配置及核对结果 |
|---|---|---|
| `universal_snake_leg_omni_student_disturbed.json` | `universal_student_disturbed_2026-10-22_iter499` | 实际500次DAgger更新，原启动YAML保留500轮；当前卡片默认150轮。五帧280维Gaussian学生，训练25%环境受扰、预热外扰系数1，默认归档回放100%受扰。 |
| `universal_snake_leg_omni_student.json` | `universal_student_clean_2026-10-22_iter150` | 用户指定检查点iter150，实际151次DAgger更新；教师冻结。归档回放参数将基座受扰比例设为0，原始训练YAML不改写。 |
| `universal_snake_leg_omni_teacher_guided.json` | `universal_three_teacher_gaussian_mirror_2026-10-22_iter499` | 500轮、三教师指导，全Gaussian、6/3/6/3自由度，原生采样范围及干净单帧观测；结构化模型由任务实现配置。 |
| `periodic_stepping.json` | `periodic_stepping_2026-10-22_iter50` | 步频下限0.25 Hz、8192训练环境、51轮；保留原地踏步奖励。 |
| `walking.json` | `zbot_walking` | 抬脚奖励0.5、落地惩罚−0.02、600轮；保留已有前进及步频配置。 |
| `disturbed_walking.json` | `zbot_disturbed_walking` | 抬脚0.5、落地−0.02、600轮；扰动力/力矩和域随机化已完整列出。 |
| `velocity_tracking.json` | `velocity_frequency_tracking_2026-10-22_iter499` | 抬脚2、落地−0.001、500轮，避免随Walking父卡片改变；保留关节加速度−0.001等已有覆盖。 |
| `velocity_energy_tracking.json` | `velocity_energy_tracking_2026-10-10_iter499` | 抬脚0.5、落地−0.02；已有能量、加速度、侧向惩罚和移除步频奖励配置。500轮与父卡片一致。 |
| `velocity_energy_disturbed.json` | `velocity_energy_disturbed_2026-10-10_iter499` | 基座扰动、足部摩擦、编码器偏置和质心范围已完整列出；其余归档数值沿能量父卡片继承。 |
| `omni_velocity_energy.json` | `omni_velocity_energy_2026-10-10_iter499` | 已有三维速度范围、各轴跟踪奖励和移除侧向/航向惩罚配置；共同奖励沿能量父卡片继承。 |
| `deploy_walking.json` | `deploy_walking_2026-10-11_iter1499` | 1500轮、部署观测噪声、动作延迟、质量/增益/摩擦/编码器/质心随机化等差异已列出。 |
| `transform_biped_to_snake.json` | `transform_biped_to_snake_2026-10-16_iter199` | 8192训练环境、200轮；已有动作、变形奖励和物理参数。 |
| `transform_snake_to_biped_beta.json` | `transform_snake_to_biped_beta_2026-10-17_iter2550` | 启动计划1000轮；Beta分布及当前自动课程已有独立覆盖，保留训练/回放权重区别。 |
| `transform_snake_to_biped_teacher.json` | `transform_snake_to_biped_teacher_2026-10-17_iter200` | 关节/高度/上升速度权重12/1/8、随机roll、恢复、学习率、归一化等已有完整差异；启动计划2000轮，200是检查点编号。 |
| `transform_snake_biped_bidirectional.json` | `transform_snake_biped_bidirectional_2026-10-17_iter599` | 600轮、成功指标聚合max；已有随机roll、双向指令、站立比例及恢复约束。 |
| `snake_omni_teacher.json` | `snake_world_x_2026-10-19_iter1499` | Beta分布、熵及其他父卡片差异已显式列出；补充历史与当前独立调参的区别。 |
| `snake_site_omni_teacher.json` | `snake_site_omni_2026-10-20_iter599` | Beta分布和熵等差异已显式列出；补充拆分跟踪奖励等后续调整与历史归档的区别。 |

## 历史配方边界

- 旧行走归档曾使用body测高，当前使用修正后的足底site测高；保留修复，不从历史YAML恢复旧结构。
- Beta起立归档经过手动热更新，启动权重、保存时有效权重与当前自动准备课程不同。卡片保留当前0.6→3→6课程及play关节权重6；失效的热更新文件、resume和日志路径不复制进卡片。
- 世界X蛇形归档使用8192环境、±60°关节目标、1500轮；当前卡片已独立调整为4096环境、±90°、3000轮，部分奖励也不同。归档回放遵循 `bests/test.sh` 及目录中的 `playback_args.json`。
- site蛇形归档采用旧线速度跟踪奖励；当前卡片已有拆分前向/侧向、yaw速度和倾斜角速度等奖励，以及15秒回合、6000轮。旧回调不能仅靠权重恢复，不能把当前卡片视为历史完整配方。
- Scene初始化生成的 `terrain.num_envs/env_spacing`、Python枚举/切片的序列化形式、回调和模型类路径不是普通配方差异，不重复写入卡片。

父卡片调整后，`walking_tuning.json`、`flat_foot_velocity_energy.json` 与 `transform_snake_to_biped_arc.json` 显式保留自身已有奖励或训练计划，避免改变非归档实验。

`tests/test_best_experiment_cards.py` 检查归档来源及可编辑标量相对父卡片的差异是否在本卡明确列出；`tests/test_bests.py` 检查归档文件校验和、观测/动作兼容性和严格策略加载。这些检查不证明训练收敛或独立仿真性能。
