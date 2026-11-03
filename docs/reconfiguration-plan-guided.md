# 关节规划教师模仿：首尾稳定贴合

独立任务为`Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided`，实验名`zbot_reconfiguration_head_tail_docking_plan_guided`，使用[独立任务卡](../src/zbot_rl_mjlab/experiments/cards/reconfiguration_head_tail_docking_plan_guided.json)。保留Plan-Guided名称，当前已按用户最新要求改为**纯监督模仿，不进入PPO**。Basic和PBT的任务配置、源文件与日志均独立保留。

## 工作流程

1. 用现有模型求首尾闭合逆运动学。关节目标限位±2rad，端面期望间隙1mm，固定合法注册角180°。当前得到两组镜像解，终值约交替±π/2。
2. 在关节空间检查整条路径的限位与实体自碰撞；直线插值可行时直接采用，否则使用双向采样连接。碰撞穿透容差为0.1mm，属于数值容差，不表示允许明显自穿透。
3. 在真实仿真动力学中执行规划教师，产生59维观测和对应动作标签；目标动作通过原速度积分接口执行，不改写qpos。教师加入小幅执行噪声，标签仍为反馈纠偏动作。示范必须在结束时保持贴合，否则拒绝训练。
4. 只训练确定性学生Actor的动作均方误差。没有PPO、Critic、回报估计、奖励优化、动作混合或引导退火。
5. 周期性从直链初态进行无教师学生回放，连续两次全部测试世界最终稳定贴合后停止。回放和导出策略只运行学生网络。

教师根据观测中已有的积分目标，在已选定路径上投影并前进，接近终点时减速、越过路径时纠偏。每个学生固定一个规划分支，避免同一直链初态得到相反动作标签；可用`--agent.algorithm.plan-branch 1`训练另一分支。学生仍使用原59维输入，不加入规划相位或参考轨迹。

规划文件为[head_tail_joint_plan.json](../assets/reconfiguration/head_tail_joint_plan.json)。训练读取已验证的规划，不在每个仿真步求逆运动学；需要重新生成时执行：

```bash
./scripts/python.sh scripts/plan_reconfiguration.py --starts 8
```

运动学路径只是第一步，训练示范和学生都必须通过动力学验证。当前验证保留原有地面、伺服、磁吸及weld物理；没有测试关闭磁吸后的表现。

## 参数与验收

物理沿用初始Basic：5ms×4子步、控制20ms、机械限位±π、积分目标±2rad。纯监督训练默认主场景16世界、教师示范16世界、4秒规划运动+2秒保持。网络为256/128 ELU、59维观测运行归一化、6维确定性动作。

Adam学习率0.001，每次监督更新采样512条示范，最多3000次更新；每100次保存，每250次做4世界、9秒的无教师验证。训练中的“iteration”指监督优化更新次数，不是PPO rollout。

贴合验收条件：原点距离≤4mm、两site的+Z夹角≤8°、相对线速度≤0.1m/s、法线垂直方向的相对角速度≤1rad/s，连续≥0.5秒。最终验证必须在回放结束时仍满足持续保持，`success_rate`表示最终仍稳定的比例；`ever_success_rate`仅表示过程中曾经达标。成功不要求weld状态，但物理磁吸仍存在。

## 训练与回放

当前可直接运行`./train.sh`和`./run.sh`。完整命令如下：

```bash
./scripts/python.sh train.py Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided \
  --log-root logs/rsl_rl_reconfiguration_plan_imitation

./scripts/python.sh play.py Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided \
  --log-root logs/rsl_rl_reconfiguration_plan_imitation \
  --env.scene.num-envs 1
```

`train.sh`/`run.sh`已启用24.c新实验并支持参数透传，Basic/PBT条目保留注释。训练脚本保留用户最新手动设置的100次监督更新上限（卡片默认3000）；严格验证的示例模型使用1250次更新，100次不保证教会。回放优先用独立日志目录中的模型，尚无模型时使用已验证的示例学生；显式`--checkpoint-file`或`--log-root`优先。新实验回放默认优先选`best.pt`，也可传`--checkpoint-file`。策略为确定性网络；无需`--stochastic`。Viewer显示当前是否稳定、曾否成功、保持时间和法线夹角。

完整训练检查点保存Actor、优化器、示范数据、规划哈希、训练进度和验证结果。完整恢复继续监督训练；只加载Actor时不读取规划或教师。已连续通过验收的检查点恢复训练会直接报告已教会。当前版本没有Critic参数或PPO优化器状态。

## 已验证结果与边界

实际短训练：4个教师世界，每条示范300个控制步；无教师验证4世界×9秒。日志在`outputs/reconfiguration/planning_guidance/pure_imitation_strict.log`。

| 监督更新次数 | 最终稳定比例 | 最短最终持续保持 | 最大最终距离 | 最大法线角差 |
|---|---:|---:|---:|---:|
| 1000 | 4/4 | 4.54s | 0.579mm | 1.10° |
| 1250 | 4/4 | 4.56s | 0.549mm | 1.02° |

连续两次达标后在1250次更新停止。已验证的学生模型：

```
outputs/reconfiguration/planning_guidance/pure_imitation_strict/zbot_reconfiguration_head_tail_docking_plan_guided/2026-10-26_20-10-22/best.pt
```

这些是单一规划分支、同一直链初态、当前仿真物理下的受控结果，不代表随机初态、外力扰动、不同摩擦或实机泛化。早期开发记录包含被用户取消的PPO原型及“曾成功”验收，均不作为当前纯教师方案的结果。

新测试覆盖IK和整段碰撞检查、采样规划绕障、分支与教师反馈、无weld的几何保持计数、监督训练/恢复和Actor独立回放；测试明确禁止调用PPO更新。Basic源文件与任务卡哈希、原43任务运行时配置均已逐项核对一致。
