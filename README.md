# Zbot 强化学习实验

基于 MJLab 的六关节 Zbot 仿真训练项目，包含行走、蛇足变形和综合运动任务，共注册 30 个实验。
行走类覆盖周期踏步、速度跟踪、能耗、扰动和部署配置；变形类覆盖双足变蛇形、蛇形站起和指令控制的双向切换。
新增综合任务通过多教师蒸馏预训练一个统一策略，联合学习双足运动、蛇形运动和变形；尚需完整训练验收。

当前先做[全能特权教师的纯双足对照](docs/privileged-omni-teacher.md)：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Wide-Biped` 全部环境固定双足，只采样全向 `vx/vy/wz`、混合指令和停车；蛇形及变形暂不采样。仍用56维真值输入（构型1维、任务1维），网络与PPO参数保持不变。通过 `ZBOT_EXPERIMENT=biped bash train.sh` 训练、`ZBOT_EXPERIMENT=biped bash run.sh` 回放；日志独立保存在 `logs/ablations/biped_scalar_only/`，完整训练由用户启动。
当前默认入口为[分任务教师指导＋PPO的全能特权教师](docs/guided-privileged-teacher.md)：`bash train.sh` 从头训练新实验，产生检查点后用 `bash run.sh` 回放。每轮同时执行PPO和按任务监督；双足全向归档指导行走，双向变形归档指导站起／爬下；默认不加载世界X蛇形教师，蛇形全向由PPO学习。变形执行镜像目标与归直后重试规则。保持56维特权观测和分任务网络，学生训练继续暂缓。没有100代统一切换。旧纯PPO实验使用 `ZBOT_EXPERIMENT=taskrouted`，其他既有选项和检查点保留。
部署学生训练暂缓；原[全向学生及CUDA Graph分析](docs/snake-leg-omni.md)保留供后续使用。
原[四教师世界X实验](docs/snake-leg-universal.md)和[独立蛇形任务](docs/snake-omni-teacher.md)保留用于历史复现。
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
`train.sh`、`run.sh` 使用 `ZBOT_EXPERIMENT` 选择实验；默认 `guided`，无需另建启动脚本。需要命令行指定任务和参数时，使用 `scripts/python.sh train.py/play.py`。
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

双足动作范围的独立对照任务与100轮不动诊断见[动作范围检查](docs/omni-v3-100-action-range.md)；完整训练仍由用户启动。

当前统一入口为 `bash train.sh` / `bash run.sh`，采用[显式任务状态机](docs/omni-task-fsm.md)，默认关闭100/500轮周期验收输出。

当前500轮停滞的诊断与单技能对照用法见[受控对照说明](docs/fsm-training-stall-control.md)，仍使用`train.sh`和`run.sh`。

[标量观测更新](docs/scalar-observation-update.md)仅用于当前Wide全能任务；旧bests文件及play兼容验证通过。

### 最新蛇形 site 全向实验归档

`bests/snake_site_omni_2026-10-20_iter599` 保存 600 次 PPO 更新后的最终检查点、
实际训练配置、源码快照、训练标量与独立评测。严格全向评测目前仅停止场景通过，
其余方向尚未达标；本归档用于保留实验进度，不代表合格的全向控制器。

```bash
./bests/test.sh snake_site_omni_2026-10-20_iter599 --env.scene.num-envs 16
```

详情见 [归档说明](bests/snake_site_omni_2026-10-20_iter599/README.md)。
