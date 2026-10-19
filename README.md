# Zbot 强化学习实验

基于 MJLab 的六关节 Zbot 仿真训练项目，包含行走、蛇足变形和综合运动任务，共注册 22 个实验。
行走类覆盖周期踏步、速度跟踪、能耗、扰动和部署配置；变形类覆盖双足变蛇形、蛇形站起和指令控制的双向切换。
新增综合任务通过多教师蒸馏预训练一个统一策略，联合学习双足运动、蛇形运动和变形；尚需完整训练验收。
独立的[蛇形全向教师](docs/snake-omni-teacher.md)覆盖六向运动、头部稳定、柔和接触和自碰撞检测，支持键盘控制。
综合部署实验从第一轮起使用受限观测学生，在同一受扰环境中接受教师指导，再继续节能与抗扰强化学习，见[最终学生培养](docs/universal-locomotion.md#最终部署学生)。

## 环境与运行

需要外部 MJLab 环境及其 PyTorch、MuJoCo 等依赖，本仓库不复制 MJLab。`scripts/python.sh` 优先使用
`MJLAB_ROOT/.venv/bin/python`，默认 `MJLAB_ROOT=/home/yhzhu/AI/mjlab`；可用 `ZBOT_PYTHON` 直接指定已安装依赖的解释器。
若指定解释器不可执行，包装脚本回退到系统 `python3`。脚本自动加入本项目源码路径。

```bash
export MJLAB_ROOT=/实际路径/mjlab
# 如需独立解释器：export ZBOT_PYTHON=/实际路径/venv/bin/python

./scripts/python.sh train.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
  --env.scene.num-envs 8192 --agent.max-iterations 2000

./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
  --checkpoint-file /实际路径/model.pt --env.scene.num-envs 1
```

行走训练可将任务 ID 改为 `Mjlab-Zbot-6dof-Walking`。各任务参数可在任务 ID 后加 `--help` 查看。
`train.sh`、`run.sh` 直接列出各实验命令，暂不使用的命令整段注释；当前执行综合任务。切换实验时手动调整注释。需要命令行指定任务和参数时，使用 `scripts/python.sh train.py/play.py`。
评估和诊断工具统一放在 `scripts/`，例如 `./scripts/python.sh scripts/transformation_eval.py --help`。

## 目录

| 路径 | 用途 |
|---|---|
| `train.py`、`play.py` | 训练与回放入口 |
| `src/zbot_rl_mjlab/experiments/` | 实验注册与训练配方 |
| `src/zbot_rl_mjlab/tasks/walking/` | 行走环境、指令、奖励与动作 |
| `src/zbot_rl_mjlab/tasks/snake/` | 蛇形全向运动指令、头部稳定与接触约束 |
| `src/zbot_rl_mjlab/tasks/transformation/` | 变形环境、动作、状态、奖励与课程 |
| `src/zbot_rl_mjlab/tasks/universal/` | 综合运动指令、模式奖励及统一全身接触环境 |
| `src/zbot_rl_mjlab/teachers.py`、`distillation.py` | 冻结教师适配、监督预热与联合 PPO |
| `src/zbot_rl_mjlab/tasks/common.py` | 跨任务共享的物理扰动行为 |
| `src/zbot_rl_mjlab/rl.py` | 公共 PPO 默认值、热奖励控制与训练运行器 |
| `src/zbot_rl_mjlab/` | 机器人定义、命令行与可视化 |
| `assets/` | 唯一运行模型 `zbot_6/ZBOT6.xml` 及其网格 |
| `scripts/` | Python 环境包装、评估、诊断与训练预览工具 |
| `tests/` | 回归测试 |
| `bests/` | 10 个归档模型、配置与精简复现资料 |
| `docs/` | 任务、指标、执行器、坐标约定及历史记录 |

训练日志与评估输出是可再生产物，按需保存在 `logs/`、`outputs/`，不纳入版本管理。
完整历史快照及大型评估明细已备份到项目外，位置见 [归档索引](bests/归档位置.json)。
变形评估默认只保存摘要，需要逐步轨迹时加 `--save-trace`。

## 检查与说明

在已安装开发工具和仿真依赖的环境中执行：

```bash
./scripts/python.sh -m pytest tests
./scripts/python.sh -m ruff check src scripts tests train.py play.py
./scripts/python.sh -m pyright --pythonpath /实际路径/venv/bin/python
```

当前 Pyright 检查仍有既有类型诊断，清理不会视为类型检查通过；细节见 [清理记录](docs/cleanup-report.md)。

测试通过只说明其覆盖的接口和行为符合预期，不代表策略收敛或实机可用。已有模型对任务 ID、观测维数、动作映射和执行器定义有兼容性要求。

- [蛇形全向教师](docs/snake-omni-teacher.md)：训练、键盘控制与无窗口验收
- [综合运动与多教师蒸馏](docs/universal-locomotion.md)：训练、键盘控制及续训说明
- [任务与实验结构](docs/task-experiment-architecture.md)
- [变形实验](docs/transformation.md)与[双向变形](docs/bidirectional-transformation.md)
- [坐标系约定](docs/zbot-coordinate-system.md)
- [步态指标](docs/gait-metrics.md)与[RS00 执行器](docs/rs00-actuator.md)
- [历史项目记录](docs/project-memory.md)与[历史重置计划](docs/reset-plan.md)
