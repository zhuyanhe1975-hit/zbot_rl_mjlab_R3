# 蛇足全能部署学生

> 这是历史四教师、世界+X前爬配方。当前目标为两种形态的site坐标全向 `vx/vy/wz`，使用[新全向实验](snake-leg-omni.md)。`train.sh` 已切到新任务，本文旧命令仅用于复现。

独立实验 `Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy`，日志目录 `logs/rsl_rl/zbot_universal_snake_leg_deploy/`。保留旧综合实验配置；新学生从头训练，避免用旧三教师运行完整续训导致教师来源不匹配。

四个冻结教师共同监督同一个六关节 Beta 学生：

| 教师 | 归档 | 适用能力 |
| --- | --- | --- |
| walking | velocity_energy_disturbed_2026-10-10_iter499 | 双足前进 |
| omni | omni_velocity_energy_2026-10-10_iter499 | 双足平移 |
| transform | transform_snake_biped_bidirectional_2026-10-17_iter599 | 爬下、起立及静止构型 |
| snake | snake_world_x_2026-10-19_iter1499 | 世界 +X 规范初态前爬 |

学生 Actor 沿用五帧、每帧37维的部署观测：编码器位置/速度、软件目标角/上一动作、基座IMU、构型与速度请求、步态时钟。Critic 使用53维特权状态。部署只需要学生 Actor、相同历史缓冲和目标角积分器；教师及其特权输入仅用于训练。该输入约定具备部署接口，不代表已经完成实机验证或产出合格模型。

预热按环境编号均衡分配双足前进、双足侧移、爬下、起立、蛇前爬五组。蛇前爬使用归档初始四元数与零关节姿态，目标速度0.08 m/s，避免弯曲起立初态替换前爬示范。首版蛇形指令固定使用世界 +X 参考，与归档优化目标一致，头部左右摆动不会旋转速度请求。腿走保持原基座前向约定。学生可见蛇教师的0.7Hz步态时钟。教师观察记忆独立保存，观测重建不改动学生环境缓存。

每个教师分别缓存、均衡抽样和保留验证样本。达到最低预热轮数后，学生必须同时满足四教师样本覆盖与动作标准差门槛，并连续通过五组独立闭环验收，才能进入 PPO。蛇爬验收要求持续向 +X 推进、构型到位、有限且安全，并限制横向漂移和转动；不能仅靠静止或一次冲刺通过。PPO 后继续退火模仿并保护已通过验收的学生轨迹，质量和增益随机化、摩擦随机化与外扰沿用部署配方。

第一版蛇教师未证明后退、侧移、转向或任意世界朝向能力，蛇形采样先限定为前爬和停车，教师也仅在已覆盖的姿态/指令域提供标签。双足方向速度课程继续保留。扩大蛇形运动范围需要补充经过闭环验证的教师及对应验收场景。

训练：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 3000 \
  --agent.run-name four_teachers_v1
```

历史训练 checkpoint 内保存四个教师的路径与SHA256；完整续训必须保持教师一致。查看 TensorBoard 的 `teacher_snake_*`、`warmup_blocked_snake`、`snake_success` 与 `distillation_warmup`，区分仍在预热和已进入 PPO。达到3000轮本身不代表验收通过；若门控持续失败，应先检查各能力闭环指标。

回放指定学生检查点：

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy \
  --checkpoint-file logs/rsl_rl/zbot_universal_snake_leg_deploy/<运行目录>/model_2999.pt \
  --env.scene.num-envs 1
```

导出和硬件接入仍需使用与训练相同的IMU坐标、历史顺序、控制周期、目标积分速度及关节边界，并在部署前单独完成验证。

## 采样性能与终端输出

教师只对路由到自己的环境执行网络推理和动作转换。蛇观测在本次查询中复用；walking/omni 共用本次查询的速度、角速度、重力和朝向。所有共享状态只存活于一次查询，重复读取同一步号下已变化的动作或环境仍会刷新。统计计数保留在GPU，每轮更新时统一读取。预热复用学生已有的分布均值，保留原随机采样、随机数状态和 rollout 存储。

默认 `--agent.console-mode compact`：每轮约4～6行，保留进度、吞吐、collection/learning time、蒸馏阶段、回报、教师覆盖和验收门控。完整的逐项奖励/损失仍由原日志器写入TensorBoard；警告、异常及非训练诊断继续输出。需要原终端长列表时使用 `--agent.console-mode full`。该开关适用于综合训练运行器，新代码在下次启动或续训时加载。

复测当前版本：

```bash
./scripts/python.sh scripts/benchmark_collection.py \
  --num-envs 8192 --repeats 6 \
  --output outputs/collection_benchmark.json
```

如保留了优化前的 `teachers.py` 和 `distillation.py`，可用 `--baseline-dir <目录>` 在同一进程中交替对照。脚本先以IEEE FP32检查路由、权重和动作等价，再以训练默认TF32计时；子批矩阵乘法可能产生不同的TF32舍入，报告单独记录其误差。完整采样计时包含仿真、教师、学生、有限值检查、存储及逐步日志统计；每次重置后先运行24步起步阶段，再测量后续24步连续采样。教师查询微基准使用固定状态，与完整采样耗时分别报告。

2026-10-20 在RTX 5090上对照8192环境、24步、交替6次：完整采样中位数1.0315秒 → 0.9077秒（下降12.0%），纯采样吞吐19.06万 → 21.66万步/秒；固定状态教师查询12.840ms → 8.903ms（下降30.7%）。结果保存在 `outputs/collection_benchmark.json`。这是教师驱动预热阶段的独占GPU基准，不能直接外推到PPO阶段或完整训练时长。TF32误差字段来自单次共享状态查询，并非全训练轨迹的误差上界。

请在GPU空闲时测量，避免并行训练/测试干扰结果。脚本不更新策略、不保存训练检查点，不会自动停止训练；`--pause-pid` 仅在显式指定时临时暂停该训练进程，正常及异常退出均恢复。
