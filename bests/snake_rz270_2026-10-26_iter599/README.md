# RZ270 蛇形：用户认可结果 iter599

任务：`Mjlab-Zbot-6dof-Snake-RZ270-Teacher`。原运行：`logs/rsl_rl/zbot_snake_rz270_teacher/2026-10-26_10-14-22`。

按用户“这次结果可以归档”保存最新完成运行的最终 `model_599.pt`，共 **600 次 PPO 更新**、8192 环境。不是从历史模型中按分数搜索出的最优项。原训练目录和所有旧归档保留。

## 实际训练配置

- 相邻模块固定安装 RZ270；每次趴地、蛇头朝世界 +X，关节零位不变。
- Actor/Critic 均为63维观测，256/128 ELU Gaussian，6维独立目标角速度积分动作；控制周期0.02秒。
- 前行速度0～0.2 m/s、10%停车，世界目标航向[-π,π]；**每5秒重采样**。
- 速度奖励为蛇头 `foot_0_sole` site 前向实际速度相对指令的线性符合度，反向负分、超速衰减，原始值域[-1,1]，**权重2**；停车分支尺度0.05 m/s。
- 航向为 `−abs(最短航向误差)`（弧度），权重1。
- 蛇头倾斜角平方权重−1，横向速度绝对值权重 **−2**。以原始 `env.yaml` 为准。
- 关节目标角±120°，目标角变化速度上限2π rad/s；保持原执行器定义。

以上按该次训练启动配置记录，不用文档早期默认值替代。模型、`env.yaml`、`agent.yaml` 和TensorBoard事件均逐字节复制。

## 回放

```bash
./bests/test.sh snake_rz270_2026-10-26_iter599
./bests/test.sh snake_rz270_2026-10-26_iter599 --env.scene.num-envs 8
```

`playback_args.json` 固定该归档的指令范围、5秒采样间隔和各奖励权重，后追加的命令行参数可覆盖。原有W/S、方向键/小键盘速度与航向控制、实际水平速度和wz箭头均可使用。

重新执行有界验证：

```bash
./scripts/python.sh bests/snake_rz270_2026-10-26_iter599/verify_archive.py
```

该脚本会重新写入 `verification.json`，再次运行后需重建该目录的SHA256SUMS；常规回放不修改归档。

## 资料与验证

- `metadata.json`：来源、模型SHA256、更新数、观测/动作契约和实际奖励定义。
- `training_metrics.json` 与原始事件：0～599步训练统计，不当作独立性能验收。
- `verification.json` / `verification.log`：严格Actor加载及CUDA确定性均值回放，8环境×20秒、前5秒预热、1000步，观测/动作/奖励有限，**0次复位**。归档兼容性与可迁移性检查78项通过。含停车、0.05/0.1/0.2 m/s与0/±90°/180°固定航向的诊断数据。
- `experiment_card.json` 与 `resolved_experiment_card.json`：归档时原始及展开的任务卡。
- `source_snapshot.tar.gz` / `SOURCE_MANIFEST.json`：归档时工作区的源码、机器人资源、脚本、测试和文档，含未提交文件；排除其他归档、训练日志、缓存和外部运行环境，不声称是训练启动瞬间的完整快照。
- `workspace.patch` 与 `training_git/`：归档时已跟踪文件差异与训练程序留下的Git资料（本次训练Git目录为空）。
- `environment.json`：已验证的公共运行环境版本。
- `SHA256SUMS`：校验所有保留文件，自身除外。

该归档来自用户目视认可；上述回放是有界诊断，完整速度/航向精度见记录，不作为全部技能或实机性能验收。

最终全量回归：**3539通过，1跳过**，见 `full_tests.log`。
