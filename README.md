# Zbot 强化学习实验（R3）

[永久记忆簿](docs/project-memory.md) · [新会话启动语](docs/new-session.md)

**[打开实验导览与参数卡片](experiments.md)**：43个实验的网络、PPO、环境与奖励参数统一从卡片修改。

基于外部 MJLab 的六关节 Zbot 仿真训练项目，当前注册 **43 个实验**：双足行走、蛇形运动、双向变形，以及多教师指导的综合运动。训练使用 MuJoCo、PyTorch 和 RSL-RL；本仓库不复制 MJLab 或安装新的运行依赖。

2026-10-22：综合学生实验在此封存为R3，见[封存与验证记录](docs/r3-freeze.md)。冻结教师、无扰动学生`model_150.pt`及受扰学生`model_499.pt`均已归档，入口见[bests索引](bests/README.md)。学生仍使用五帧传感器历史，未加入yaw；相对航向观测仅为后续待验证方案。PPO各版本保留作实验记录，不表示已达到质量目标。原始训练日志和临时诊断结果仅保留在本机。

新增独立 [RZ270 蛇形实验](docs/rz270-snake.md)：相邻模块固定安装角为 270°，关节零位不变；每次从趴地、蛇头朝世界 +X 开始。

新增 [重构实验1：首尾磁吸闭环](docs/reconfiguration.md)：旧180°构型直链趴地，机械限位±π，首尾被动连接；磁吸参数为可调仿真假设。Basic已恢复13:11初版：59维观测、速度积分目标±2rad和原始物理配置；PBT版先以8个策略探索，再根据评估集中为4个策略，总训练环境数保持8192。

独立新增[关节规划教师模仿实验](docs/reconfiguration-plan-guided.md)：IK与关节路径生成动力学示范，学生只做监督学习，验证独立稳定贴合后停止，无PPO阶段。

## 当前实验与入口

`train.sh`、`run.sh` 保留本地逐项命令和注释选择方式，实际启用项以脚本内容为准。跨机器使用时，推荐下面的 `scripts/python.sh` 入口，通过 `MJLAB_ROOT` 或 `ZBOT_PYTHON` 选择已经安装 MJLab 的环境。

封存前最后训练分支为[22.d外部扰动蒸馏学生](docs/universal-student.md)：直接继承22.a的纯DAgger学生流程，冻结原全能教师，不加载PPO学生。基座各轴随机力±5N、力矩±0.5N·m，持续0.08～0.20秒、间隔0.8～2秒；训练约25%环境受扰，从教师预热第一轮起生效。卡片默认150轮（20轮教师预热、100轮交接、30轮学生独立执行），最后实际训练覆盖为500轮。封存时`train.sh`、`run.sh`的实验命令均已注释，保留8192环境、500轮训练及归档模型回放示例；这些学生命令保持注释；当前运行脚本已切换到重构实验Basic。22.a及PPO v2/v3/v4配置和本地已有日志保留。

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

`scripts/python.sh` 按 `ZBOT_PYTHON` → `$MJLAB_ROOT/.venv/bin/python`（默认 `$HOME/AI/mjlab`）→ `${ROBOTICS_ENV:-$HOME/dev-envs/robotics-mjlab-py313}/bin/python` 选择解释器。显式指定的解释器无效时直接报错，不回退系统 Python。包装器自动设置 R3 源码路径和工作目录，入口可从其他目录调用；缓存默认保留在 `~/.cache`。本机直接复用 R2 公共环境，无需安装依赖，详见[开发环境说明](docs/development-environment.md)。交互使用可执行 `source scripts/activate.sh`。

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
| `src/zbot_rl_mjlab/base_snake_teacher.py` | base蛇形实验的两阶段自适应调权 |
| `src/zbot_rl_mjlab/robot.py`、`assets/zbot_6/ZBOT6.xml` | 六关节模型、执行器与物理参数 |
| `scripts/`、`tests/` | 评估诊断工具和回归测试 |
| `bests/` | 18个正式归档模型、配置、校验和与复现资料 |
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
- [特权教师](docs/privileged-omni-teacher.md)、[指导教师](docs/guided-privileged-teacher.md)、[全能学生](docs/universal-student.md)

`logs/`、`outputs/`、`work/`及缓存为本地训练/诊断产物，不纳入版本管理。2026-10-21已按要求删除项目外旧完整快照，未再创建完整备份；保留的本地模型资料见[bests索引](bests/归档位置.json)。
