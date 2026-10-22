# 实验导览与参数卡片

实验参数从下表的 **实验卡片** 进入。卡片是训练、回放共用的最终参数入口；装配代码负责选择任务机制，任务代码负责执行动作、计算观测和奖励。

## 怎么修改

- **共享参数**：[公共PPO默认卡片](src/zbot_rl_mjlab/experiments/cards/_defaults.json)。修改后，所有未覆盖该字段的子卡片同步继承。
- **单个实验**：点击下表卡片，在 `agent`、`env`、`rewards` 或 `addons` 中修改或添加覆盖。父卡片也有直达链接，可查看继承值。
- **生效时机**：保存后，下次启动训练/回放生效；Python内再次调用实验 `build_env()` / `build_ppo()` 也会重新读取。已有运行进程和已经创建的注册配置不热更新。
- **优先级**：公共默认 → 父卡片 → 当前卡片 → `train` / `play` 环境差异 → 命令行覆盖。修改卡片不会改写检查点或源代码。

| 想调整什么 | 卡片字段 |
|---|---|
| Actor / Critic层宽 | `agent.actor.hidden_dims`、`agent.critic.hidden_dims` |
| 激活与输入归一化 | `agent.actor.activation`、`agent.actor.obs_normalization` |
| 探索噪声等分布参数 | `agent.actor.distribution_cfg` |
| 学习率、熵、折扣与批次 | `agent.algorithm` |
| 训练轮数、rollout长度、保存频率 | `agent.max_iterations`、`agent.num_steps_per_env`、`agent.save_interval` |
| 外扰、域随机化、观测噪声与动作延迟 | `addons.<扩展类型>` |
| 当前site实验纯vx训练采样比例 | `train.env.commands.transform.motion_x_only_probability` |
| 动作范围与速度、指令采样范围 | `env.actions`、`env.commands` |
| 控制周期、回合长度、物理设置 | `env.decimation`、`env.episode_length_s`、`env.sim` |
| 奖励权重及函数参数 | `rewards.<奖励名>.weight`、`rewards.<奖励名>.params` |
| 仅训练 / 回放不同的设置 | `train.env`、`play.env`、`train.rewards`、`play.rewards` |

例如只修改[双足行走卡片](src/zbot_rl_mjlab/experiments/cards/walking.json)，可在现有JSON中合并以下字段：

```json
{
  "agent": {
    "actor": {"hidden_dims": [256, 128]},
    "algorithm": {"learning_rate": 0.0003}
  },
  "rewards": {"upright": {"weight": 0.3}}
}
```

这里展示覆盖写法，保留卡片已有的 `schema_version`、`task_id` 和 `inherits`。删除某个覆盖字段即可恢复父卡片值；`"$remove": ["字段名"]` 用于从继承结果移除字段，不代表删除任务中的奖励或动作。新增奖励函数/观测/动作类型应在任务实现和实验装配中定义，再在卡片中配置参数。`task_id`、Runner路径和回调函数不通过卡片替换。

网络结构、动作范围或观测相关参数改变后，需要匹配的检查点或重新训练；归档模型仍按原接口严格加载。自适应实验中的速度奖励权重是控制器初始预算，训练期间仍由对应调权算法更新。EMA系数、阶段判定等算法内部常量仍在控制器实现中，不属于当前卡片的普通配置字段。

已有 bests 对应卡片的父子差异、归档来源及历史配方边界见[归档实验卡片说明](docs/best-experiment-cards.md)。

`train.sh`、`run.sh` 当前分别显式运行22.d外部扰动蒸馏学生训练与回放；此前实验命令保留为注释。需要显式指定参数时使用：

```bash
./scripts/python.sh train.py <任务ID> --env.scene.num-envs 4096
./scripts/python.sh play.py <任务ID> --checkpoint-file <对应检查点>
```

## 全部实验

### 双足行走（12）

第0项 `Periodic-Stepping` 按指令步频交替抬脚，取消前进速度奖励，增加水平速度平方惩罚（权重−5），以原地踏步为训练目标。`Walking` 继续使用前进速度奖励，不继承该惩罚。旧版周期踏步模型按前进奖励训练，修改配置后必须重新训练；play 只执行已有策略，不会根据新奖励改变动作。原地踏步回放默认从 `logs/rsl_rl/zbot_periodic_stepping/` 选择模型，不能使用 `bests/zbot_walking/` 作为原地踏步效果的依据。

| 实验卡片 | 继承来源 | 注册任务ID |
|---|---|---|
| [周期踏步](src/zbot_rl_mjlab/experiments/cards/periodic_stepping.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Periodic-Stepping` |
| [双足平地行走](src/zbot_rl_mjlab/experiments/cards/walking.json) | [periodic_stepping](src/zbot_rl_mjlab/experiments/cards/periodic_stepping.json) | `Mjlab-Zbot-6dof-Walking` |
| [受扰双足行走](src/zbot_rl_mjlab/experiments/cards/disturbed_walking.json) | [walking](src/zbot_rl_mjlab/experiments/cards/walking.json) | `Mjlab-Zbot-6dof-Disturbed-Walking` |
| [速度与频率联合跟踪](src/zbot_rl_mjlab/experiments/cards/velocity_tracking.json) | [walking](src/zbot_rl_mjlab/experiments/cards/walking.json) | `Mjlab-Zbot-6dof-Velocity-Frequency-Tracking-Walking` |
| [前向节能行走](src/zbot_rl_mjlab/experiments/cards/velocity_energy_tracking.json) | [velocity_tracking](src/zbot_rl_mjlab/experiments/cards/velocity_tracking.json) | `Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking` |
| [受扰节能行走](src/zbot_rl_mjlab/experiments/cards/velocity_energy_disturbed.json) | [velocity_energy_tracking](src/zbot_rl_mjlab/experiments/cards/velocity_energy_tracking.json) | `Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking` |
| [双足全向节能行走](src/zbot_rl_mjlab/experiments/cards/omni_velocity_energy.json) | [velocity_energy_tracking](src/zbot_rl_mjlab/experiments/cards/velocity_energy_tracking.json) | `Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking` |
| [平足节能行走](src/zbot_rl_mjlab/experiments/cards/flat_foot_velocity_energy.json) | [velocity_energy_tracking](src/zbot_rl_mjlab/experiments/cards/velocity_energy_tracking.json) | `Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Walking` |
| [受扰平足行走](src/zbot_rl_mjlab/experiments/cards/flat_foot_velocity_energy_disturbed.json) | [flat_foot_velocity_energy](src/zbot_rl_mjlab/experiments/cards/flat_foot_velocity_energy.json) | `Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Disturbed-Walking` |
| [部署行走](src/zbot_rl_mjlab/experiments/cards/deploy_walking.json) | [velocity_energy_tracking](src/zbot_rl_mjlab/experiments/cards/velocity_energy_tracking.json) | `Mjlab-Zbot-6dof-Deploy-Walking` |
| [部署平足行走](src/zbot_rl_mjlab/experiments/cards/deploy_flat_foot.json) | [flat_foot_velocity_energy](src/zbot_rl_mjlab/experiments/cards/flat_foot_velocity_energy.json) | `Mjlab-Zbot-6dof-Deploy-Flat-Foot-Walking` |
| [步频奖励调参](src/zbot_rl_mjlab/experiments/cards/walking_tuning.json) | [walking](src/zbot_rl_mjlab/experiments/cards/walking.json) | `Mjlab-Zbot-6dof-Walking-Tuning` |

### 变形（7）

| 实验卡片 | 继承来源 | 注册任务ID |
|---|---|---|
| [双足变蛇形](src/zbot_rl_mjlab/experiments/cards/transform_biped_to_snake.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Transform-Biped-To-Snake` |
| [蛇形站起Gaussian](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Transform-Snake-To-Biped` |
| [蛇形站起Beta](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_beta.json) | [transform_snake_to_biped](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped.json) | `Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta` |
| [浅弧姿态站起](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_arc.json) | [transform_snake_to_biped_beta](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_beta.json) | `Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc` |
| [鲁棒镜像站起](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_robust.json) | [transform_snake_to_biped_beta](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_beta.json) | `Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust` |
| [站起特权教师](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_teacher.json) | [transform_snake_to_biped_beta](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_beta.json) | `Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher` |
| [双向变形教师](src/zbot_rl_mjlab/experiments/cards/transform_snake_biped_bidirectional.json) | [transform_snake_to_biped_beta](src/zbot_rl_mjlab/experiments/cards/transform_snake_to_biped_beta.json) | `Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional` |

### 独立蛇形推进（1）

| 实验卡片 | 继承来源 | 注册任务ID |
|---|---|---|
| [世界X蛇形推进教师](src/zbot_rl_mjlab/experiments/cards/snake_omni_teacher.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Snake-Omni-Teacher` |

### Site / Base蛇形与对照（5）

| 实验卡片 | 继承来源 | 注册任务ID |
|---|---|---|
| [site全向教师（iter599归档Beta）](src/zbot_rl_mjlab/experiments/cards/snake_site_omni_teacher.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher` |
| [site全向自适应教师](src/zbot_rl_mjlab/experiments/cards/snake_site_adaptive_teacher.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher` |
| [base全向两阶段自适应教师](src/zbot_rl_mjlab/experiments/cards/snake_base_adaptive_teacher.json) | [snake_site_adaptive_teacher](src/zbot_rl_mjlab/experiments/cards/snake_site_adaptive_teacher.json) | `Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher` |
| [固定前进对照A](src/zbot_rl_mjlab/experiments/cards/snake_site_forward_teacher.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher` |
| [固定前进放松对照B](src/zbot_rl_mjlab/experiments/cards/snake_site_forward_relaxed_teacher.json) | [snake_site_forward_teacher](src/zbot_rl_mjlab/experiments/cards/snake_site_forward_teacher.json) | `Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher` |

### 综合运动与教师 / 学生（15）

| 实验卡片 | 继承来源 | 注册任务ID |
|---|---|---|
| [综合运动多教师](src/zbot_rl_mjlab/experiments/cards/universal_locomotion.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Universal-Locomotion` |
| [综合运动部署学生](src/zbot_rl_mjlab/experiments/cards/universal_locomotion_deploy.json) | [universal_locomotion](src/zbot_rl_mjlab/experiments/cards/universal_locomotion.json) | `Mjlab-Zbot-6dof-Universal-Locomotion-Deploy` |
| [蛇足综合部署学生](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_deploy.json) | [universal_locomotion_deploy](src/zbot_rl_mjlab/experiments/cards/universal_locomotion_deploy.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy` |
| [蛇足全向部署学生](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_deploy.json) | [universal_locomotion_deploy](src/zbot_rl_mjlab/experiments/cards/universal_locomotion_deploy.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy` |
| [全能特权教师](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher.json) | [_defaults](src/zbot_rl_mjlab/experiments/cards/_defaults.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher` |
| [纯双足对照教师](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_wide_biped.json) | [universal_snake_leg_omni_teacher](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Wide-Biped` |
| [奖励分组Critic教师](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_multi_critic.json) | [universal_snake_leg_omni_teacher_wide_biped](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_wide_biped.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Multi-Critic` |
| [分任务Critic教师](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_task_critic.json) | [universal_snake_leg_omni_teacher_multi_critic](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_multi_critic.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Task-Critic` |
| [分任务网络教师](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_task_routed.json) | [universal_snake_leg_omni_teacher_task_critic](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_task_critic.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Task-Routed` |
| [归档指导全能教师](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_guided.json) | [universal_snake_leg_omni_teacher_task_routed](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_task_routed.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided` |

| [全能教师蒸馏学生](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student.json) | [universal_snake_leg_omni_teacher_guided](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_teacher_guided.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student` |
| [22.d外部扰动蒸馏学生](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_disturbed.json) | [universal_snake_leg_omni_student](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-Disturbed` |

| [监督学习后学生PPO](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_ppo_v2.json) | [universal_snake_leg_omni_student](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student.json) | `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO` |

当前默认22.d继承22.a纯蒸馏并加入从首轮生效的基座外扰，训练150轮，日志为`logs/rsl_rl_universal_student_disturbed/`。保留的PPO v2从150代DAgger学生重开，统一交接判据、补充变形收敛与速度代价、源学生运动参考和蛇形课程，每25次策略更新独立验收；日志为`logs/rsl_rl_universal_student_ppo_v2/`，教师保持冻结且不参与PPO。原DAgger实验及`logs/rsl_rl_universal_student_v2/`保留。阶段切换、初始化校验与续训命令见[实验说明](docs/universal-student.md)。

[学生 PPO v4](src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_ppo_v4.json) 注册为 `Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO-V4`，继承 v3 的 150 代学生初始化、300 次迭代、验证和模型选择设置，日志使用 `logs/rsl_rl_universal_student_ppo_v4/`。仅将速度奖励替换为质量加权全机身质心速度：世界系 EMA 后投影到平滑机身航向，避免头部摆动虚增平移速度；蛇形命令上限为 `(0.06, 0.06, 0.5)`，双足仍为原范围。命令保持现有定义（双足以重力方向 × BASE 局部 +Z 确定前向、蛇形头部 site 的 -Z 前向），先转到世界系再投影到同一平滑机身坐标系。验证指标仍沿用 v3 的 site 速度口径，因此应结合质心奖励判断进展；v2/v3 配方保持不变。可直接复制 `train.sh` 的 22.c 命令启动。

## 职责与源码位置

| 层次 | 职责 | 位置 |
|---|---|---|
| 实验卡片 | 可编辑参数、继承关系与训练/回放差异 | `experiments/cards/*.json` |
| 实验装配 | 选择任务、绑定奖励/观测/动作，叠加对照差异和卡片 | `experiments/walking.py`、`transformation.py`、`bidirectional.py`、`snake.py`、`site_snake.py`、`universal.py`、`omni_teacher.py` |
| 任务机制 | 可复用的场景、配置类型、动作/命令和计算函数 | `tasks/walking`、`tasks/transformation`、`tasks/snake`、`tasks/universal` |
| 训练运行 | PPO、蒸馏、分任务模型、调权控制器及检查点 | 根包的Runner/算法模块 |

卡片通过同一 `Experiment.build_env()` / `build_ppo()` 进入注册配置，没有另一套生成参数文件。完整架构与验证见[任务与实验分级](docs/task-experiment-architecture.md)。
