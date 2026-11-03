# 分享发布与复现

发布形式为完整源码包，包含41个注册实验、实验卡片、机器人资源、测试、文档及18个bests归档。当前代码依赖仓库根目录的 `assets/`、`bests/`，不将独立wheel作为交付形式。

## 生成分享包

在项目根目录运行，无需安装新的依赖：

```bash
python3 scripts/package_release.py
```

默认生成 `dist/zbot-rl-mjlab-0.2.0-source.tar.gz`。包内 `RELEASE_MANIFEST.json` 记录每个文件的路径、字节数和SHA256；固定输入生成相同的压缩包。文件来自当前工作区，包含尚未提交的源码和卡片，不依赖Git提交状态。

包含：源码、机器人XML/网格、实验卡片、文档、测试、原始入口和全部bests资料。排除：`.git/`、`.venv/`、`.omx/`、缓存、egg-info、`logs/`、`outputs/`、`work/`、`build/`、`dist/`、MuJoCo诊断日志。选中范围内的符号链接会被拒绝，不读取项目外的文件。

本地虚拟环境和编排状态只留在原工作区，不进入分享包；历史归档内的训练配置、源码快照及溯源路径保留原样，并继续使用各目录原有 `SHA256SUMS` 校验。

## 接收方运行

1. 解压完整目录，不单独移动 `src/`、`assets/` 或 `bests/`。
2. 准备已安装MJLab及其MuJoCo、PyTorch、RSL-RL等依赖的Python环境。项目不随包提供这些第三方依赖。
3. 指定环境并验证注册、模型兼容性，然后选择回放任务。

```bash
cd zbot-rl-mjlab-0.2.0
export MJLAB_ROOT=/实际路径/mjlab
# 或 export ZBOT_PYTHON=/实际路径/venv/bin/python
./scripts/python.sh -c 'import zbot_rl_mjlab; from mjlab.tasks.registry import list_tasks; print(len([t for t in list_tasks() if t.startswith("Mjlab-Zbot")]))'
./scripts/python.sh -m pytest tests/test_bests.py tests/test_release_portability.py -q
./bests/test.sh
./bests/test.sh velocity_frequency_tracking_2026-10-22_iter499 \
  --frequency 0.5 --target-speed 0.3 --env.scene.num-envs 16
```

注册数量应为41。源码入口会设置模块搜索路径，不要求先安装本项目。需要将本项目注册到外部MJLab环境时，可在该环境使用已有构建工具执行 `python -m pip install --no-deps --no-build-isolation -e .`；采用editable方式，保留完整源码树。

`train.sh`、`run.sh` 通过 `scripts/activate.sh` 使用已有运行环境，保留既有实验选择和参数；它们不是接收方环境的自动安装脚本。跨机器优先使用上述通用入口。依赖版本与共享环境管理见[开发环境说明](development-environment.md)。

## 验证环境及边界

2026-10-22清理前测试环境：Python3.13.7、MJLab1.6.0、MuJoCo3.11.0、PyTorch2.14.0、RSL-RL5.5.1、Warp1.14.0、TensorDict0.14.2、Tyro1.0.1。这些是已验证环境记录，不是重新解析安装的依赖锁；`uv.lock`仅描述本项目。

清理前全量3006项测试通过，Ruff通过。全仓Pyright基线仍有456条既有类型诊断；本次不通过屏蔽诊断或大改训练类型体系来声称类型检查全部通过。新增发布工具与回归测试单独检查类型。最终验证记录见[发布清理报告](release-cleanup-report.md)。

发布检查验证源码可搬迁、注册和配置完整、模型严格加载及文件一致性，不代表新策略训练收敛、实机可用或旧模型在所有指令组合下达标。只加载自己信任的检查点。

本仓库尚未选择开源许可证，本次清理没有代替作者授予许可。对外开源时应明确代码、模型及机器人资源的授权范围。
