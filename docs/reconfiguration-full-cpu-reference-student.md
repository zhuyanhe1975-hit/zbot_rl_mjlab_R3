# CPU 全流程教师轨迹与残差 PPO 学生

独立实验覆盖直线→卷曲闭环→[270,270,180,90,90]→首尾被动断开→展开。CPU MuJoCo 状态机负责阶段切换、几何规划、载荷断开、重连和保护；PPO 给运动阶段增加 6 维有限位置残差和 1 维曲线阶段速度修正，初始/保持和力矩加载阶段不加残差。模型以阶段、目标、连接状态、关节、载荷与当前阶段对应教师的 q/qvel/qref 作为观测。轨迹参考按阶段、目标、进度序号和阶段时间匹配，未出现的阶段以零参考表示。

奖励包括阶段/连接进度、时间、60 N 以上接近 80 N 的其他连接面载荷惩罚、教师关节轨迹跟踪误差、成功及失败信号。实际每 1ms 检查其他连接面达到 80 N 被动断裂并触发失败/重置；关节14/20 Nm持续/短时保护、首尾实际80 N被动断开、接触保护仍由状态机执行。原状态机加载求解器及展开预测的较保守内部优化缓冲仍保留，因此80 N是失败标准，不表示规划器总会用满该阈值。速度修正仅影响卷曲、分离、旋转、接近的时间参数，最终展开当前仅由位置残差修正，尚不能用本实验保证4秒内展开。

教师参考采集及训练：

```bash
./scripts/python.sh scripts/collect_full_cpu_reference.py
./scripts/python.sh scripts/train_full_student.py --iterations 1000 --log-dir logs/full_cpu_reference_student
./scripts/python.sh scripts/eval_full_cpu_student.py logs/full_cpu_reference_student/latest.pt
```

采样数据为50 Hz实际动力学状态，保存在 `outputs/reconfiguration/full_cpu_student/teacher_reference.npz`，同名JSON包含教师完整报告。训练是单环境CPU PyTorch PPO，非MJLab GPU并行训练、也非RSL-RL实现；一个512步rollout对应10.24s仿真，跨PPO更新保留当前物理回合。保存latest.pt及metrics.jsonl；恢复请用 `--resume`，网络和优化器恢复但物理回合重新开始。训练检查点依赖原参考文件。所给奖励和完整演示只说明训练机制，不保证速度或安全裕度优于教师；须用独立回放及扰动验证。

## GPU 速度参考与力矩修正实验

`train.sh` 当前选择 4096 环境 GPU 原型。教师数据中的 `qvel` 是状态机实际动力学关节速度，以阶段、目标、序号和阶段时间匹配；运动阶段将该速度与策略速度修正相加，积分为 MJLab 原生位置执行器目标。卷曲阶段对参考速度做符号镜像平均，保持同步；第七维可调节轨迹时钟。首尾捕获与整个接近和对接阶段均叠加基于实际注册端口位姿误差的雅可比速度修正。闭环加载阶段在雅可比加载力矩上叠加策略力矩修正；其他连接面达到80N被动失效、训练复位，关节14Nm持续/20Nm短时保护保留。物理步长1ms，控制周期20ms。

该GPU实现尚处实验阶段，运行日志及检查点模式为 `experimental_full_gpu_velocity_torque_v8`。位置残差版本和CPU单环境版本检查点不兼容；之前的运行数据不作为本版本性能证明。`--velocity-scale` 默认0.5rad/s，`--torque-scale` 默认2Nm；`--residual-scale` 只适用于旧CPU单环境版本。单次采样及参数更新不等于完整重构成功，仍需检查各阶段和终态。`run.sh` 当前仍指向旧CPU回放入口，尚不能回放GPU检查点。

每个活动阶段使用独立的仿真时间超时看门狗；超时记为失败并自动复位该环境。加载阶段上限40s，首尾捕获4s，连接保持2s，最终直线保持3s，展开120s；曲线运动阶段还按教师段时长设定上限，允许最慢0.5倍时钟并额外留4s，且有30s绝对上限。初始保持4s。超时使用不受策略时钟缩放影响的物理仿真时间，正常换段优先于恰好到边界的超时。日志的 `phase_timeout` 是 `failure` 的子集。

训练终端采用 RSL-RL 风格输出，包含迭代号、环境步数、吞吐、采样和学习的 CUDA 流耗时、分项损失、完成回合的平均奖励和长度、动作标准差、成功/失败/超时数量及六个连接面载荷峰值。没有完整回合时奖励和长度显示 n/a。非有限载荷不计入有限峰值，另以 `Non-finite load world-steps` 明确计数；对应异常状态仍触发失败保护。TensorBoard 标量与 `metrics.jsonl` 同时保存。计时是相关 CUDA 流的事件间隔，包含流中空闲与发射间隙，不应理解成纯 GPU 核心运算时间。训练算法仍为此项目的自实现 PPO，而非 RSL-RL 算法实现。

性能路径使用 Torch/Warp 共享内存和同一 CUDA 流；1ms 执行器和载荷监测固定计算使用 CUDA 图，自动复位采用 GPU 固定形状掩码。当前内力加载求解仍需通过 CPU 获取活动环境索引和约束行数，故尚不是完全无 CPU 同步；不要把它描述成全部调度、全部求解均已彻底 GPU 化。GPU 掩码复位只支持本实验当前无命令、传感器、课程、状态化奖励和观测历史的配置，不生成 MJLab 标准的逐回合 summary 字典；本训练脚本直接记录完成回合、奖励和失败等统计。

## 精修正向速度参考（2026-10-29）

GPU全流程原型默认参考更新为 `assets/reconfiguration/refined_forward_velocity_reference.npz`，由 `./scripts/python.sh scripts/collect_refined_forward_reference.py` 通过 `view_headtail_mjbatch_refined.py` 的正向真实CPU动力学采集。1ms物理、20ms采样，完成12.52s、601条样本，qvel是实测六关节rad/s，非关节位置差分或策略输出。CPU教师的首尾断后0.5s非运动保持没有独立GPU阶段，采集时省略这26条样本，GPU状态机以目标6、阶段4的SETTLE_HEADTAIL模拟等待；原始报告及哈希在同名JSON。旧 `outputs/reconfiguration/full_cpu_student/teacher_reference.npz` 不覆盖；单环境CPU学生仍默认旧参考。

GPU实验根据参考标识启用首尾90°注册site与真实weld、分离同步旋转、2cm分离门槛、0.25s保持、0.5s首尾断后等待、1mm/2°捕获以及关闭卷曲镜像平均；速度限幅从3提高到7.2rad/s以覆盖本参考观测峰值约6.16rad/s。旧参考仍保留旧配置。GPU实验自身仍使用原80N连接断裂/其他面失败标准、1ms物理和30/2执行器设置；CPU教师采用100N物理阈值、其他面80N安全保护及不同增益，因此该参考不是物理完全一致的GPU教师演示，也不保证GPU能完成轨迹。

新GPU模式为 `experimental_full_gpu_refined90_velocity_torque_v9`，检查点记录参考SHA256，续训和回放校验模式、路径及哈希；旧v8用旧参考仍保留，不能将旧模型当作新参考模型续训。4096环境、2步一次PPO更新和4环境16步/一次更新、检查点加载0.1秒回放冒烟通过；只验证接线，尚无全流程成功或学习收敛证据。训练入口例子：

```bash
./scripts/python.sh scripts/train_full_student.py --env.scene.num-envs 4096 --iterations 500 --log-dir logs/full_gpu_reference_student_refined90
```

`train.sh`现有注释状态未更改，原日志与检查点保留。

## 零修正 GPU 参考与单环境查看器（2026-10-29）

`./scripts/python.sh scripts/view_headtail_gpu_refined.py` 运行一个 GPU 物理环境并打开窗口；Enter 重置，空格暂停。画面是精修正向的参考控制，策略修正为零，不是已训练的 RL 模型；可用 `--reference-rate` 调整参考时间尺度，默认 0.2 倍。物理步长始终 1ms，控制周期 20ms。默认参考方案采用 100N 物理断裂、80N 非目标保护、外层 80/4 与内层 400/4，并允许低速、近直链终态最大关节角 0.04rad。保护触发后的下一子步停机不能保证载荷不越过阈值。

`./scripts/python.sh scripts/validate_refined_gpu_baseline.py` 使用相同配置验收 4096 个 GPU 世界；该程序锁存每个世界首回合结果，完成的世界可被重置以便其余世界继续。只在实际全部完成且无失败时报告通过。当前零策略参考与旧的全流程 PPO 模式使用不同配置，不能将本查看器中的通过结果当作旧 PPO 已经健康或新策略已训练成功；具体有限测试和未完成事项参见 `docs/project-memory.md`。

## 小范围 GPU RL 试验（2026-10-29）

`bash train.sh` 启动 `scripts/train_refined_gpu_incremental.py`，默认 4096 个 GPU 世界、物理 1ms、控制 20ms、参考时钟 0.2 倍。成功参考状态机仍负责连接、断开及其他阶段；PPO 只在 CURL 叠加速度残差，最大 0.02rad/s，累计位置偏移限幅 0.02rad。初始策略均值为零，固定采样标准差 0.1；其他阶段的策略速度修正、时钟控制和加载力矩修正被禁止。状态机沿用100N物理断裂及80N非目标保护，不添加扰动。失败或非有限载荷会在当前采样步结束训练并跳过本轮更新；保护触发后由于惯性仍可能超过阈值，不能理解为连续时间硬保证。

默认 100 轮，日志和检查点位于 `logs/refined_gpu_curl_pilot`，与旧实验隔离；目录非空时需显式 `--resume` 或指定新的 `--log-dir`。如需中断后恢复：`bash train.sh --resume logs/refined_gpu_curl_pilot/latest.pt`。检查点标记为未经完整流程验证的候选，不可将保存当作成功或安全认证。按当前参考速度，完整回合约43秒仿真时间，128控制步/轮约2.56秒仿真时间，训练前若无失败，前约16轮没有完成回合属于预期。4096世界完整终态尚未验收，不声称保证稳定、学习收敛或提速。

## 起始闭环独立 PPO 试验（2026-10-29）

`bash train.sh` 默认启动 `scripts/train_refined_gpu_initial_closure.py`，默认4096个GPU环境。它只覆盖初始直链的卷曲、首尾捕获和0.25秒闭合稳定；全部六面连接且无保护失败后立即结束成功回合，不执行首个内部连接面加载或后续全流程。CURL阶段采用零均值初始化的小残差，速度与累计偏移分别限幅0.02rad/s与0.02rad；保持1ms物理、100N物理断裂、80N非目标保护、0.2倍参考速度及无扰动。

独立日志目录为 `logs/refined_gpu_initial_closure_ppo`，与先前全流程试验检查点不兼容。日志增加 `mean_success_time_s` 便于观察成功闭环耗时；时长和成功率需与同配置的零修正参考对比才能证明改善。失败或非有限载荷会停止且不更新该批次，原因保存为 `stop.json`。默认100轮，用户自行执行；只做小规模逻辑及图/重置回归，不自动启动训练或大规模验证。

起始闭环另有即时目标位姿奖励 `pose_nearness`：以实际90°首尾捕获位姿的距离 `d`（米）及姿态角误差 `θ`（弧度）计算 `exp(-d/0.15 - θ/0.75)`；只在 CURL 和 CAPTURE_HEADTAIL、目标为首尾时启用，权重1。奖励管理器按20ms控制周期积分，因而单步最大0.02。它是接近目标的密集奖励，不是安全约束；原成功/失败、时间和载荷规则不变。日志 `mean_pose_reward_rate` 为整个采样批次的平均加权奖励率（含非生效阶段的零）。为避免混用旧奖励目标的优化器/critic，新检查点模式为 `refined_gpu_initial_closure_pose_ppo_v2`，不兼容闭环v1检查点。

训练控制台采用 RSL 风格展示迭代、吞吐量、GPU流采样/学习时间、损失、完成回合奖励和长度、成功/失败及预计剩余时间；附带闭环平均耗时、目标位姿奖励率、CURL策略样本数和非目标载荷峰值。仅在有完成回合时显示其平均值，避免用未完成样本冒充回合统计；GPU流计时包含对应流上的调度间隙。原始每轮指标继续保存在 `metrics.jsonl`，TensorBoard 指标保留。

## 起始闭环：2ms RSL-RL 常规训练（当前入口）

用户要求更接近成功 walking 实验、允许失败重置并精简观测。`bash train.sh` 现在启动真正的 RSL-RL `MjlabOnPolicyRunner`，继承归档 walking 的配置：24 控制步/环境、ELU 256/128 actor 与 critic、观测归一化、初始动作标准差1.0、PPO 5 epochs/4 minibatches、自适应学习率初始0.001、γ=0.99、λ=0.95、entropy系数0.01、KL目标0.01及clip 0.2。默认4096环境、100轮、物理2ms与控制20ms；20ms参考采样按同一时间尺度使用，没有将1ms样本误当2ms回放。

观测精简为41维：六关节位置/速度、首尾目标位置误差与姿态误差、五个内部连接面载荷、四个闭环阶段、相位时间、六维残差积分状态与七维平滑动作状态。策略只在卷曲阶段起作用：额外关节速度幅度0.2rad/s、累计位置偏移限幅0.15rad、参考时间倍率在0.5～1.5之间；其余阶段由参考状态机执行。保留实际捕获位姿接近/进展奖励、成功奖励、时间及安全奖励，连接错断、载荷保护、碰撞或超时触发−10及对应环境GPU掩码自动重置，其他世界继续。物理断裂100N、非目标保护80N未放宽，但检测后惯性可能导致越限，不是硬上限。

日志独立于旧自定义PPO：`logs/refined_gpu_initial_closure_rsl_2ms`，重复启动请指定新目录或 `--resume` 本模式的RSL检查点。RSL标准日志包含真实回合奖励、长度和PPO信息，已结束回合还显示成功/失败、错断/载荷保护比例、闭环耗时、非目标载荷和位姿奖励；不再保留额外的零策略对照世界。当前仅完成小规模接线测试，2ms配置的训练效果及4096成功率未经验证。用户自行运行训练，未修改旧检查点。

当前 `bash run.sh` 回放起始闭环的**已训练RSL策略**：`scripts/view_initial_closure_rsl.py` 默认从 `logs/refined_gpu_initial_closure_rsl_2ms` 自动选迭代号最大的 `model_*.pt`，单GPU世界、2ms物理，确定性动作；Enter重置，空格暂停。可追加 `--checkpoint-file PATH` 指定检查点、`--headless` 无窗口运行。检查点模式和参考哈希必须匹配，不使用零策略参考冒充模型。旧全流程参考查看器仍保留为 `scripts/view_headtail_gpu_refined.py`。
