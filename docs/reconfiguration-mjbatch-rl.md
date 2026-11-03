# 独立 mjbatch 首尾闭合强化学习任务

该任务直接使用 `mjbatch.Batch` 和 MuJoCo 3.14 CPU 并行仿真，不导入 MJLab。六模块、初始姿态及 90° 注册端口按 `scripts/view_headtail_mjbatch_refined.py` 所用资产重建；MuJoCo 3.14 的碰撞生成与原查看器 3.11 有差异，旧查看器的成功记录不等同于本任务验证。

```bash
# 默认 256 环境、使用可用 CPU 核心、200 轮；旧日志在重训时自动改名保留
./train_mjbatch.sh

# 更改参数；有检查点时只有显式传 --resume 才续训
./train_mjbatch.sh --num-envs 32 --threads 8 --iterations 200
./train_mjbatch.sh --resume logs/mjbatch_headtail_no_collision/model_100.pt

# 优先读取通过独立验证的 best.pt；否则提示并回放最新检查点
./run_mjbatch.sh
# 指定检查点，或不加载模型检查零残差控制
./run_mjbatch.sh --checkpoint logs/mjbatch_headtail_no_collision/model_100.pt
./run_mjbatch.sh --zero-policy --headless
```

使用 `~/dev-envs/robotics-mjbatch-py313`；脚本入口为 `scripts/python_mjbatch.sh`，可通过 `ZBOT_MJBATCH_PYTHON` 指定兼容解释器。该环境单独安装 mjbatch 0.1.4 和 MuJoCo 3.14；Torch/CUDA 等大依赖使用 `~/.cache/uv` 的现有硬链接，不改变共享 MJLab 环境。原始 mjbatch 仓库 `~/Github/mjbatch` 的线程池头文件用于每环境原生约束载荷审计，物理批量状态推进实际调用 `mjbatch.Batch`。

## 任务

从五个内部连接闭合、首尾打开的固定直链开始；策略输出 7 个有界残差（6 关节速度及 1 个速度尺度）。每 1 ms 根据实测首尾位姿、六关节及五个内部连接载荷，计算关节空间终点方向与对齐雅可比；每 20 ms 采一次策略动作。不读取时间索引轨迹。内部连接按刚性链近似建立规划雅可比；MuJoCo 原有物理接触仍参与动力学。

按用户要求，当前实验不检测碰撞、不因碰撞终止，也不使用碰撞雅可比避让。非目标焊接载荷超过 80 N、电机持续过载或状态异常仍会失败；力矩请求裁剪至 ±20 Nm。gap<1 mm、角差<2°、线速度<0.1 m/s、角速度<0.6 rad/s 且保持 30 ms 时建立首尾焊接，稳定 250 ms 后成功；成功不代表无碰撞。

观测 40 维：6 关节角、6 角速度、3 对接相对位置、3 相对姿态、5 内部载荷、7 上一动作、3 阶段、1 归一化耗时、6 规划速度。奖励包含成功时 `20 / 耗时`、每秒最多 0.1 的位姿接近引导、失败 -20 和每秒 -0.2 的耗时惩罚；日志分别使用 `Reward/`、`Penalty/`、`Metrics/`。20 秒期限作为失败结算。

当前关节速度上限采用精修查看器卷曲阶段的 `30°/s × 12 = 2π rad/s`，不设置加速度限制。位置伺服 Kp=15、Kd=1.5，执行器净力矩限制 ±20 Nm；实际控制仍受80 N非目标载荷保护影响。关节终点采用查看器固定 IK 分支，实时生成速度，不读取查看器离线控制轨迹。

PPO 默认 256 环境、使用进程可用的 CPU 核心数（本机为 20）、24 步 rollout、CPU 网络、3 轮 4 小批次，演员只在运动阶段更新；控制动作经 `tanh` 限幅。每 50 轮及最终额外启动独立固定初态均值策略评估，所有世界成功才参与按时间选择 `best.pt`。验证增加实际时间；可用 `--eval-interval 0` 临时关闭。`--device cuda` 仅把策略网络放到 GPU，mjbatch 物理仍在 CPU。

预测采样 MPC 为可选实验，默认关闭；使用 `--mpc-candidates 4 --mpc-horizon 2` 才启用。它从实时状态复制候选并用独立 `mjbatch.Batch` 模拟，每个候选通过相同的子步保护；策略动作是候选中心。它增加计算开销，目前仅测试单步预测与实步一致，不保证提升学习速度或取得更优行为；用户当前要求先用默认简化方案跑通。

## 验证边界

`tests/test_mjbatch_headtail.py` 检查实际 Batch、无 MJLab 导入、独立重置、与直接 MuJoCo 一致的焊接载荷、碰撞不再作为终止原因、MuJoCo warning 计数故障处理及可选 MPC 的一步预测。固定初态零残差、短 PPO 更新及独立评估的结果在项目记忆中记录；这些有限验证不证明扰动鲁棒性、任何随机策略安全或实物可行性。目标速度上限不代表实际关节必然达到该速度。
