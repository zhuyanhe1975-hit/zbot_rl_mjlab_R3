# 已保存模型

本目录保留十四个可供加载检查的模型、训练启动配置及简要结果。2026-10-21已按要求删除项目外旧完整快照；当前仍保留模型目录内的源码快照（如有）、配置、简要结果和校验和，位置见 [归档位置.json](归档位置.json)。

## 播放与校验

```bash
./bests/test.sh                         # 列出模型
./bests/test.sh 1                       # 按序号播放
./bests/test.sh transform_snake_biped_bidirectional_2026-10-17_iter599 --env.scene.num-envs 1
./scripts/python.sh -m pytest -q tests/test_bests.py
```

每个模型目录只保存一个 `model_*.pt`。`env.yaml`、`agent.yaml` 是训练启动时的配置；`final_effective_weights.json`（如有）记录模型保存时的奖励状态，两者可能对应不同训练阶段。目录内 `SHA256SUMS` 覆盖当前保留文件，自身不参与校验：

```bash
(cd bests/transform_snake_biped_bidirectional_2026-10-17_iter599 && sha256sum -c SHA256SUMS)
```

## 模型用途与已有证据

| 目录 | 用途与验证边界 |
|---|---|
| [velocity_frequency_tracking_2026-10-22_iter499](velocity_frequency_tracking_2026-10-22_iter499/README.md) | 速度与频率联合跟踪，最新09:00:05运行的500轮训练iter499（已替换原归档）；步频0.25–1.0 Hz、速度0–0.5 m/s、关节加速度惩罚−0.001。保存模型、原始配置、训练统计及归档时源码；独立仿真性能尚未评估。 |
| [periodic_stepping_2026-10-22_iter50](periodic_stepping_2026-10-22_iter50/README.md) | 按指定频率原地踏步，最新51轮训练的iter50，步频0.25–1.0 Hz；取消前进奖励、水平速度平方惩罚−5。保存模型、配置、训练统计及归档时源码；独立仿真性能尚未评估。 |
| `snake_site_omni_2026-10-20_iter599` | foot_0_sole site 全向速度教师，56/56维观测、6维Beta动作，完成600次PPO更新。已严格加载并完成72场景CUDA回放；严格全向评测仅停止通过（1/9类），作为实验版本归档，不宣称全向能力达标。见目录内配置与验证记录。 |
| `snake_world_x_2026-10-19_iter1499` | 世界X推进、世界Y侧滑惩罚，51/66维观测、6维速度积分Beta动作。最终训练日志世界X速度约0.150 m/s；实际倍率下限0.25、增益5。8环境30秒确定性回放无失败，平均世界X速度0.083 m/s，有单侧漂移；见目录内配置和验证记录。 |
| `transform_snake_biped_bidirectional_2026-10-17_iter599` | 蛇形／双足双向变形，37/45 维观测、3 维镜像 Beta 动作。此前严格加载及 8 环境、18 秒回放通过，发生 23 次自动切换；未统计完整严格成功率或实机表现。 |
| `transform_snake_to_biped_teacher_2026-10-17_iter200` | 随机纵轴滚转 ±90° 的蛇形起立教师，36/44 维观测、3 维镜像 Beta 动作。此前 16 环境、16 秒评估严格成功 16/16，无外部扰动。 |
| `transform_snake_to_biped_beta_2026-10-17_iter2550` | 30 维 Actor、6 维 Beta 动作。此前原生任务 4 环境、16 秒评估严格成功 4/4；训练中手动热更新奖励，自动准备课程未完成从头训练验证。 |
| `transform_biped_to_snake_2026-10-16_iter199` | 双足趴成蛇形，6 维独立动作。此前 16 环境完成 16 秒且无不安全终止，但严格零角度／保持成功为 0/16。 |
| `deploy_walking_2026-10-11_iter1499` | 面向部署的前向行走，28 维 Actor 不读取基座线速度和偏航误差；此前仿真评分场景均无跌倒，尚未实机验证。 |
| `velocity_energy_tracking_2026-10-10_iter499` | 节能前向速度跟踪；此前 16 环境、1000 步无跌倒。 |
| `velocity_energy_disturbed_2026-10-10_iter499` | 节能速度跟踪加扰动；此前 16 环境、1000 步无跌倒。 |
| `omni_velocity_energy_2026-10-10_iter499` | 全向速度指令，平移跟踪较好；纯偏航较弱，不能视为转向已解决。 |
| `zbot_walking` | 按指定频率平地行走的旧基线，具体任务映射见 `test.sh`。 |
| `zbot_disturbed_walking` | 带扰动的频率行走旧基线，具体任务映射见 `test.sh`。 |

以上是既有实验记录，不代表此次清理重新完成了仿真评估。现存证据以各模型目录为准；项目外的原始备份已删除。未列入的旧平足模型不在当前归档中。

## 兼容性注意事项

除 `deploy_walking_*` 外，早期行走模型训练时足部高度扫描附着于脚体原点，后续代码改为足底站点。高度不是策略观测，旧模型仍可加载，但新旧训练奖励不宜直接比较。

`snake_site_omni_2026-10-20_iter599` 对应的原site任务已按源码快照恢复为56维Beta、±60°关节目标和原指数奖励；可直接用 `./bests/test.sh snake_site_omni_2026-10-20_iter599` 回放。61维Gaussian自适应/base/forward任务保持独立，检查点仍不能互换。

模型不能随意跨任务播放：观测顺序、动作维度、分布、控制周期和执行器参数均须匹配。`tests/test_bests.py` 检查这些配置及严格加载；修改物理参数后，还应运行 `scripts/fall_check.py` 或相应变形评估工具验证实际回放。
