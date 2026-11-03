# 公共开发环境

建立日期：2026-10-25。面向本机后续兼容的机器人、强化学习及 PyTorch 项目。

| 项目 | 位置 / 选择 |
|---|---|
| 公共 Python 环境 | `~/dev-envs/robotics-mjlab-py313` |
| 环境登记与消费者清单 | `~/dev-envs/README.md` |
| 完整安装版本 | `~/dev-envs/robotics-mjlab-py313.freeze.txt` |
| 解析约束 | `~/dev-envs/robotics-mjlab-py313.constraints.txt` |
| 上游依赖覆盖 | `~/dev-envs/robotics-mjlab-py313.overrides.txt` |
| 包下载与解包缓存 | `~/.cache/uv`，uv 配置采用 hardlink |
| Python 下载 | uv 管理的 CPython 3.13，不修改系统 Python |
| 基础版本 | MJLab 1.6.0、PyTorch 2.14.0；其余以完整清单为准 |
| GPU | RTX 5090，32 GB |
| NVIDIA 驱动 | 系统现有 580.178.04，保留使用 |
| 用户级工具 | `~/.local/bin/uv`、`uvx`、`git` |

## 日常使用

本项目无需创建 `.venv` 或执行安装命令：

```bash
./scripts/python.sh -c 'import sys, torch; print(sys.executable); print(torch.cuda.get_device_name())'
./scripts/python.sh scripts/check_environment.py
./scripts/python.sh -m pytest tests -q
./scripts/python.sh train.py Mjlab-Zbot-6dof-Walking --help
```

交互式开发可执行 `source scripts/activate.sh`；其他兼容项目可直接 `source ~/dev-envs/robotics-mjlab-py313/bin/activate`。不自动激活整个登录 shell，不替换 `/usr/bin/python3`。

入口优先级：显式 `ZBOT_PYTHON` → 原有 MJLab 的 `.venv` → `ROBOTICS_ENV` 指定目录或上述公共目录。显式指定的解释器无效时直接报错。`train.sh`、`run.sh` 仅调整环境激活入口，实验选择与训练参数保持原样。

## 后续依赖管理

1. 先查公共环境登记表和现有包版本，兼容项目直接共用解释器。将新消费者记入登记表。
2. 安装前保存 `uv pip freeze --python <公共解释器>`，并对照消费者检查影响。使用 `uv pip install --dry-run --python <公共解释器> --constraint <约束文件> --overrides <覆盖文件> <包>` 预览。
3. 在现有约束下新增依赖；需要升级约束时，先确认现有消费者兼容，完成相应验证，再更新清单。不要把单个项目的依赖表拿来对公共环境执行 `uv pip sync`。
4. 只有无法兼容的依赖才另建具名环境；仍使用同一个 `~/.cache/uv`。缓存与环境位于同一文件系统，可通过硬链接复用包文件。不要直接混用不同环境的 `site-packages`。
5. 不将清空 uv、Warp、Torch 编译缓存作为例行维护；缓存保留在 `~/.cache`，不放 `/tmp`。

NVIDIA 驱动由系统共用。PyTorch 的 CUDA 用户态运行库由 Python 依赖提供，`nvidia-smi` 的 CUDA Version 表示驱动能力，不表示系统已经安装 CUDA Toolkit。当前训练先使用包内运行库；将来确需编译自定义 CUDA 扩展时，再单独评估 `nvcc` / Toolkit。

本仓库 `pyproject.toml` 的依赖列表为空，`uv.lock` 不包含外部训练依赖，因此不要在仓库根目录用 `uv sync` 准备训练环境。共享环境独立管理；本项目源码由入口注入路径，机器人资源和模型仍从完整源码树读取。

## 版本依据与检查

参考 [MJLab 官方安装说明](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html) 与 [1.6.0 依赖声明](https://github.com/mujocolab/mjlab/blob/v1.6.0/pyproject.toml)。官方声明固定 RSL-RL 5.4.2，而本项目历史验证使用 5.5.1。本次先装官方约束版本，全量测试出现 26 项失败（日志缺少 `rnd_cfg` 等、PPO归一化行为差异）；切换至 5.5.1 后相关 60 项测试通过。因此共享环境显式覆盖为 5.5.1，不修改第三方源码或安装元数据。

这是已记录的上游依赖声明冲突：常规 `uv pip check` 会报告 MJLab 对 RSL-RL 的版本要求不符。后续安装必须携带 `--overrides ~/dev-envs/robotics-mjlab-py313.overrides.txt`，不要让解析器自动降回 5.4.2。当前完整版本清单使用该覆盖文件进行 `uv pip install --dry-run --requirements <版本清单>` 检查，无需变更任何包。首次按官方约束安装的清单保留为 `robotics-mjlab-py313.upstream-baseline.freeze.txt`，便于对照。

`scripts/check_environment.py` 检查实际 GPU 矩阵计算及反向传播、Warp CUDA 设备、40 个任务注册、EGL 渲染和 4 个 Zbot 环境的短步进。它用于验证基础环境，不代表策略收敛或实机效果。


## R3 迁移（2026-10-26）

参考 R2 的公共环境配置接入同一个解释器，不安装或升级任何包。R2 的历史验证结果不代替 R3 验证。训练、回放、归档与 sim2sim 入口均从 R3 源码树解析路径；本次环境迁移不调整实验选择；用户随后启用的回放命令予以保留。PPO v2 初始化路径由缺失的历史 `logs/` 改为 SHA256 相同的 `bests/universal_student_clean_2026-10-22_iter150/model_150.pt`，v3/v4 继承该路径。

R3 基础验证：40 个任务注册、RTX 5090 CUDA 矩阵计算与反向传播、Warp CUDA、EGL 渲染、4 环境 8 步仿真通过；训练入口 `--help`、跨目录启动、显式无效解释器报错检查通过。迁移使用 Python 3.13.17、MJLab 1.6.0、PyTorch 2.14.0（CUDA 13.0）、MuJoCo 3.11.0、Warp 1.18.0、RSL-RL 5.5.1。未执行正式训练。

回归记录：全量首次运行 3487 通过、1 跳过、3 失败。其中历史日志模型路径失败已通过归档路径修复，相关 PPO 实验测试复测 8 通过；另外两项为用户启用 `run.sh` 后与“脚本全部注释”的封存断言冲突，保留用户选择。针对性注册/归档/架构测试 104 通过；变更文件 Ruff、Shell 语法与 Git 空白检查通过。完整测试输出：`/tmp/zbot-r3-pytest.log`，基础环境输出：`/tmp/zbot-r3-environment-check.log`。

## mjbatch 独立环境（2026-11-02）

独立任务使用 `~/dev-envs/robotics-mjbatch-py313` 和 `./scripts/python_mjbatch.sh`。mjbatch 0.1.4 固定 MuJoCo 3.14.0，与原 MJLab 环境的 3.11.0 冲突，因此单独隔离；原环境和共享 NVIDIA 驱动未修改。Torch 2.14.0、CUDA 用户态依赖、SciPy 和 pytest 从现有 uv 缓存离线硬链接复用，不从另一环境拼接 site-packages。安装清单为 `~/dev-envs/robotics-mjbatch-py313.freeze.txt`，登记见 `~/dev-envs/README.md`。
