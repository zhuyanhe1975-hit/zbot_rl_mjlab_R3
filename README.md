# Zbot 强化学习实验

**[打开实验导览与参数卡片](experiments.md)**：35个实验的网络、PPO、环境与奖励参数统一从卡片修改。

基于外部 MJLab 的六关节 Zbot 仿真训练项目，当前注册 **35 个实验**：双足行走、蛇形运动、双向变形，以及多教师指导的综合运动。训练使用 MuJoCo、PyTorch 和 RSL-RL；本仓库不复制 MJLab 或安装新的运行依赖。

## 当前实验与入口

`train.sh`、`run.sh` 保留本地逐项命令和注释选择方式，实际启用项以脚本内容为准。跨机器使用时，推荐下面的 `scripts/python.sh` 入口，通过 `MJLAB_ROOT` 或 `ZBOT_PYTHON` 选择已经安装 MJLab 的环境。

本地 `train.sh`、`run.sh` 的环境激活路径属于原作者机器配置；分享前未改动其选中实验或参数。回放已归档模型可直接使用 `bests/test.sh`，避免依赖已清理的训练日志。

独立模型索引仍可通过原有 `bests/test.sh` 使用：

```bash
./bests/test.sh                         # 列出归档
./bests/test.sh snake_site_omni_2026-10-20_iter599 --env.scene.num-envs 1
# 自定义参数使用Python入口，不改用户启动脚本：
./scripts/python.sh train.py Mjlab-Zbot-6dof-Walking --agent.max-iterations 600
./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
  --checkpoint-file bests/transform_snake_biped_bidirectional_2026-10-17_iter599/model_599.pt
```

历史文档中的 `ZBOT_EXPERIMENT`、`ZBOT_SKILL` 和“默认实验”记载属于旧入口，不适用于当前脚本。

`scripts/python.sh` 优先使用 `$MJLAB_ROOT/.venv/bin/python`，默认 `MJLAB_ROOT=/home/yhzhu/AI/mjlab`；可通过 `ZBOT_PYTHON` 指定解释器，不可执行时回退到系统 `python3`。包装器自动设置源码路径和工作目录，入口可从其他目录调用。

```bash
export MJLAB_ROOT=/实际路径/mjlab
# 或 export ZBOT_PYTHON=/实际路径/venv/bin/python
./scripts/python.sh train.py Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher --help
./scripts/python.sh scripts/evaluate_site_snake.py --help
```

## 代码导航

| 路径 | 职责 |
|---|---|
| `train.py`、`play.py`、`cli.py` | MJLab训练入口、模型回放和项目参数适配 |
| `src/zbot_rl_mjlab/experiments/` | 任务注册、训练配方、PPO配置和Runner选择 |
| `src/zbot_rl_mjlab/tasks/walking/` | 双足行走、速度指令、奖励和动作 |
| `src/zbot_rl_mjlab/tasks/snake/` | 独立蛇形运动、site/base参考与前进对照 |
| `src/zbot_rl_mjlab/tasks/transformation/` | 变形、镜像动作、课程和恢复 |
| `src/zbot_rl_mjlab/tasks/universal/` | 综合运动、任务状态机、特权观测和共享动作 |
| `src/zbot_rl_mjlab/*teacher*.py`、`*ppo.py`、`distillation.py` | 教师适配、PPO、蒸馏和检查点契约 |
| `src/zbot_rl_mjlab/base_snake_teacher.py` | 当前实验的两阶段自适应调权 |
| `src/zbot_rl_mjlab/robot.py`、`assets/zbot_6/ZBOT6.xml` | 六关节模型、执行器与物理参数 |
| `scripts/`、`tests/` | 评估诊断工具和回归测试 |
| `bests/` | 14个正式归档模型、配置、校验和与复现资料 |
| `docs/` | 任务说明、坐标约定和历史实验记录 |

任务层定义环境行为，实验层组合配方，Runner负责训练和模型兼容；实验构建钩子按MRO由基类到子类执行一次，细节见[架构说明](docs/task-experiment-architecture.md)。

## 验证与资料

```bash
./scripts/python.sh -m pytest tests -q
./scripts/python.sh -m ruff check src scripts tests train.py play.py
./scripts/python.sh -m pyright src scripts tests train.py play.py \
  --pythonpath /实际路径/venv/bin/python
```

类型检查有既有诊断，结果见[源码分级与卡片验证记录](docs/source-structure-report.md)。测试不代表策略收敛或实机可用。检查点必须匹配任务、观测顺序、动作映射、分布、坐标系和执行器定义。原site全向任务已按归档恢复为56维Beta配方，可通过 `./bests/test.sh snake_site_omni_2026-10-20_iter599` 回放；自适应/base/forward任务仍使用各自61维Gaussian配方。

## 分享与复现

分享完整源码目录，保留根目录 `assets/` 和 `bests/`；当前不提供脱离源码树的独立 wheel。运行依赖外部 MJLab 环境，`uv.lock` 不锁定这套外部运行环境。

```bash
python3 scripts/package_release.py
```

生成的源码包位于 `dist/`，包含文件清单和SHA256，排除本机环境、Git历史、训练日志、临时结果和缓存。环境准备、解压验证、许可证状态及检查边界见[发布与复现说明](docs/release.md)。

- [模型归档及验证边界](bests/README.md)
- [坐标系约定](docs/zbot-coordinate-system.md)、[执行器](docs/rs00-actuator.md)
- [蛇形site奖励](docs/site-snake-g1-rewards.md)、[自适应权重](docs/site-snake-adaptive-weights.md)、[前进对照](docs/site-snake-forward-ablation.md)
- [双向变形](docs/bidirectional-transformation.md)、[综合运动](docs/universal-locomotion.md)
- [特权教师](docs/privileged-omni-teacher.md)、[指导教师](docs/guided-privileged-teacher.md)

`logs/`、`outputs/`、`work/`及缓存为本地训练/诊断产物，不纳入版本管理。2026-10-21已按要求删除项目外旧完整快照，未再创建完整备份；保留的本地模型资料见[bests索引](bests/归档位置.json)。
