# 已保存模型

本目录保留十个可供加载检查的模型、训练启动配置及简要结果。完整源码快照、训练日志和大型评估明细已移到项目外；位置见 [归档位置.json](归档位置.json)。原始归档中的文件和校验和保持原样。

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

以上是既有实验记录，不代表此次清理重新完成了仿真评估。完整评分表和逐步轨迹可在项目外的原始 `bests-original/README.md` 及对应模型目录查阅。未列入的旧平足模型不在当前归档中。

## 兼容性注意事项

除 `deploy_walking_*` 外，早期行走模型训练时足部高度扫描附着于脚体原点，后续代码改为足底站点。高度不是策略观测，旧模型仍可加载，但新旧训练奖励不宜直接比较。

模型不能随意跨任务播放：观测顺序、动作维度、分布、控制周期和执行器参数均须匹配。`tests/test_bests.py` 检查这些配置及严格加载；修改物理参数后，还应运行 `scripts/fall_check.py` 或相应变形评估工具验证实际回放。
