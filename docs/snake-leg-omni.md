# 蛇足全向：site坐标、键盘与计算效率

> 学生训练当前暂缓，先进行[全能特权教师训练](privileged-omni-teacher.md)。`train.sh` 已切到教师任务；本文保留学生实验与前期研究记录。

当前实验：`Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy`，日志目录 `logs/rsl_rl/zbot_universal_snake_leg_omni_deploy/`。这是独立新任务；旧四教师/世界X实验、walking实验及归档文件保留原语义。

## 任务约定

双足、蛇形都响应地面平面中的 `(vx, vy, wz)`：前进/后退、左右侧移、左右原地转动、混合运动和停车。同一个部署学生通过构型请求完成双向变形，变形时平滑停车，到位后恢复保留的速度请求。

用户确认的坐标约定：

| 模式 | 前向vx | 左向vy | 正偏航wz |
| --- | --- | --- | --- |
| 双足 | `foot_0_sole` 的局部+X在地面的投影 | 向上叉乘前向 | 绕世界+Z（本场景重力反方向） |
| 蛇形 | 同一site的局部−Z在地面的投影 | 向上叉乘前向 | 同上 |

坐标随site航向旋转，不锁定世界+X。主轴投影退化时保留最近有效航向；第一次就退化时从另一site轴取得水平参考，避免凭空引入固定世界方向。默认平躺姿态下site+X接近竖直，因此不能直接把它当作蛇形地面前向。

平移奖励测量整机质心速度在上述前/左轴的投影，避免仅摆头获得推进奖励。偏航测量site绕世界+Z的角速度。部署Actor保持185维五帧可测历史，Critic使用53维特权状态。

键盘：W/S调节vx，A/D调节vy，Q/E调节wz，X停车；按键累积形成混合指令，分量按配置上限裁剪。`[` 切蛇形，`]` 切双足。以上只控制当前选中的机器人，`,` / `.` 切换选择。指令变化只更新所选机器人的软件目标和观测，保留当前物理状态及其他机器人的观测历史。

## 教师和PPO

新实验显式设置 `enabled_teachers=('transform',)`、`teacher_warmup_enabled=False`。旧walking、omni walking网络及世界+X蛇网络均不加载，也不计入教师SHA来源。

保留的双向变形教师仅在适用的变形/静止构型域给标签。两种形态的全向运动从第0轮PPO学习；不会先要求未学习的全向能力通过BC门控，避免训练死锁。暂不建立旧教师ID分组的成功轨迹anchor。当前没有已验收的新全向运动教师，不能把世界+X检查点宣称为全向教师；后续若训练出新全向教师，应先验证其坐标、观测、动作和闭环能力，再增加适配器和蒸馏配置。

训练从小幅值开始，但两种形态从开始就采样停车、六个正负单轴和混合指令，按各模式表现提高幅值；初始世界yaw随机。旧四教师检查点不能完整恢复成此任务：教师集合及预热语义不匹配会被拒绝，推荐从头训练。

独立验收为20场景：每种模式的停车、6个正负单轴、2个混合指令，加起立和爬下。每场景覆盖0/90/180/270度初始世界yaw，通过真实命令API中途改指令。检查持续跟踪、未指令轴漂移、到位比例、有限状态和双足不跌倒；每个初始yaw都要通过，持续转向没有旧蛇教师的初始yaw限制。完整验收每100轮，鲁棒验收每500轮；部署和最终能力仍须经过完整训练及验收，冒烟测试不代表学会运动。

## CUDA Graph研究与实测

本机为RTX 5090、PyTorch `2.14.0+cu130`。源码和真实运行均确认：mjlab已为MuJoCo Warp的step、forward、reset捕获CUDA Graph，因此再次给物理单步包图没有凭空新增一层收益。MuJoCo Warp官方也推荐对大量短kernel使用图捕获。[官方性能说明](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#performance-tuning)

PyTorch CUDA Graph主要省去CPU逐kernel发射开销，要求静态尺寸/地址及可捕获控制流；CPU同步、动态长度索引、重置和日志不适合直接包进整段图。[PyTorch CUDA语义](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)；其中CUDA `nonzero` 会同步主机和设备。[nonzero文档](https://docs.pytorch.org/docs/stable/generated/torch.nonzero.html)

学术依据是选择性捕获和端到端GPU数据路径，而非无条件给整个Python环境包图：PyGraph讨论图编译的成本收益和选择性应用；mjlab论文说明GPU物理和可组合管理器的架构。[PyGraph论文](https://arxiv.org/abs/2503.19779)、[mjlab论文](https://arxiv.org/abs/2601.22074)。论文跨模型加速比例不能直接套到本机器人。

新任务8192环境×24步的真实CPU/CUDA profile：约67,876次kernel发射，CPU self约162.6ms；3232次流同步，CPU self约131.0ms；404次nonzero，inclusive约68.8ms。`env_step` 是主要CPU区段。GPU主要物理kernel为CCD约55.3ms、窄相碰撞约22.9ms、接触Jacobian约17.3ms。插桩有额外开销，不同层级的inclusive时间不可相加。

冻结变形教师MLP：eager约0.0928ms，固定全批CUDA Graph约0.0462ms，微测减半，但每24步仅有约1.1ms理论节省，不能据此声称整个采样快一倍。当前主瓶颈是管理器/观测/索引中的发射和同步，而非网络矩阵乘法。

已实现更合适的局部候选：只捕获site航向的固定形状纯张量数学；动态输入逐次复制，状态写回留在图外，输出有独立存储。真实配对对照（同进程交替6次、相同seed重置+24步起步、计时后续24步）结果：

| 后端 | 采样中位耗时 |
| --- | --- |
| eager | 0.63884秒 |
| site CUDA Graph | 0.62439秒 |

本轮下降约2.26%；这是未执行优化器更新的PPO采样基准，不含完整训练/验收用时。任务配置默认eager，`train.sh` 显式启用已测的局部图。使用 `--env.omni-frame-backend eager` 可回到原路径。未修改求解器迭代、接触物理或运行时依赖。

测量工具及结果：

```bash
./scripts/python.sh scripts/profile_omni_collection.py \
  --task Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy \
  --num-envs 8192 --output outputs/omni_collection_profile.json
./scripts/python.sh scripts/benchmark_omni_graph.py --num-envs 8192 --repeats 6
```

结果为 `outputs/omni_collection_profile.json` 和 `outputs/omni_graph_benchmark.json`，可选完整Chrome trace体积约215MB。先确认没有其他GPU训练/测试并发再测量。进一步优化应优先减少CPU同步和重复索引，再考虑编译融合纯张量观测/奖励；不能直接捕获带重置、随机课程和日志的整个collection。

## 启动和回放

```bash
# 学生实验的显式启动命令（当前暂缓，train.sh 已切到教师）：
./scripts/python.sh train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy \
  --env.scene.num-envs 8192 --env.omni-frame-backend cuda_graph \
  --agent.max-iterations 3000 --agent.run-name site_omni_v1

./scripts/python.sh play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy \
  --checkpoint-file <本任务的学生检查点> --env.scene.num-envs 1
```

终端继续采用精简格式，全向验收合并显示双足9项、蛇形9项和变形2项；各场景及各yaw的完整指标仍写入TensorBoard。`--agent.console-mode full` 恢复完整终端列表。
