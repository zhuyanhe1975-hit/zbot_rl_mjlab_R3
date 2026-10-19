# 蛇形起立 Beta 模型：第 2550 次迭代

使用明确的原生 Beta 任务播放：

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta --checkpoint-file bests/transform_snake_to_biped_beta_2026-10-17_iter2550/model_2550.pt
./bests/test.sh transform_snake_to_biped_beta_2026-10-17_iter2550
```

已有 4 环境、16 秒数值评估从初始关节全零的蛇形起点严格成功 4/4。原训练曾手动热更新奖励；当前任务保留 Beta 架构并加入自动锁存准备阶段，但该自动课程尚未完成从头训练验证。

`env.yaml` 和 `agent.yaml` 保留训练启动配置，`final_effective_weights.json` 保留模型 2550 保存时的权重和阶段，二者对应不同时间。不要用 Gaussian 任务加载本模型。

原生 Beta 任务评估同样严格成功 4/4，摘要见 `goal_audit.json`；完整 `evaluation.json` 和 `native_evaluation.json` 已移到项目外原始归档，位置见上级 `归档位置.json`。未包含 60 秒／16 环境测试、ONNX/JIT 导出或硬件验证。
