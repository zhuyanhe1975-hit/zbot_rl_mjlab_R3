# R3综合学生实验封存

封存日期：2026-10-22（Asia/Shanghai）。GitHub仓库：<https://github.com/zhuyanhe1975-hit/zbot_rl_mjlab_R3>。

本版本保存当前源码、40个注册实验及实验卡片、机器人资源、测试、文档和17个bests归档。沿用R2的Git提交历史；本地`origin`指向R3，原R2远端保留为`r2`，不向R2推送本次封存提交。

综合学生阶段暂时停止在此：冻结全能教师、无扰动DAgger学生`model_150.pt`（151次更新）及外扰DAgger学生`model_499.pt`（500次更新）均在bests中保存原始权重和配置。保留PPO v2/v3/v4配方及评测工具；不将这些配方视作已完成质量验收的最终策略。未加入yaw观测，未启动新训练。`train.sh`、`run.sh`中的实验命令均已注释，手动回放使用`bests/test.sh`。

`logs/`、`outputs/`、缓存、虚拟环境和`.omx/`留在本机，不上传。归档目录中已有的训练指标、事件文件、源码快照和校验清单随模型上传，保持字节不变。两个学生模型各100588939字节，低于GitHub单文件100MiB限制，沿用原仓库的普通Git文件方式。

## 封存验证

- `./scripts/python.sh -m pytest tests -q`：3491项通过，38条警告。
- 注册检查：40个Zbot任务；归档严格加载及校验包含在全量测试中。
- `bash -n train.sh run.sh bests/test.sh`及`git diff --check`通过。
- 本次调整的学生入口与归档测试文件Ruff通过；全仓Ruff仍有`tasks/universal/mdp.py:401`的既有B010诊断，未为封存修改训练运行逻辑。
- 全仓Pyright：456项既有类型诊断，与此前发布基线数量一致；不声明类型检查通过。
- 待提交文件常见凭据模式未发现匹配；没有超过100MiB的文件。

短程GPU回放记录保存在两个学生归档的`verification.json`中。这些检查验证文件一致性及可加载性，不代表实机部署或所有运动质量目标已达标。后续相对yaw观测实验见[学生文档](universal-student.md)。
