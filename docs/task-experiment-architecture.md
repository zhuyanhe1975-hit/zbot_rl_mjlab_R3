# 任务与实验两级架构

更新日期：2026-10-18。本轮已落实任务边界、统一构建与注册、继承复用和文件归并；当前注册 20 个实验。此前重构使实验目录 14→10 个、变形任务目录 16→13 个、全部源码 38→34 个。新增综合任务后，实验目录为 11 个，全部源码为 41 个；变形任务目录仍为 13 个。

## 职责边界

| 层次 | 职责 | 主要位置 |
|---|---|---|
| 任务 | 定义环境、动作、指令、观测、奖励、终止、重置、状态和课程 | `tasks/walking/`、`tasks/transformation/`、`tasks/universal/` |
| 实验 | 选择任务配置类型，声明任务 ID、训练参数、策略分布和可选扩展 | `experiments/` |
| 共享任务行为 | 提供不依赖具体实验的物理扰动与配置装配 | `tasks/common.py` |
| 训练支持 | 提供公共 PPO 默认值、热奖励控制及模型加载／迁移运行器 | `rl.py` |
| 多教师训练支持 | 冻结教师观测／动作适配、预热、联合 PPO 与进度恢复 | `teachers.py`、`distillation.py` |
| 机器人 | 定义唯一机器人资源、关节与执行器参数 | `robot.py`、`assets/zbot_6/ZBOT6.xml` |

任务代码不反向导入实验配方。行走的速度指令与奖励回调集中在 `tasks/walking/mdp.py`，不再随实验文件定义。纯物理测量与奖励仍使用 MJLab 函数回调，不为每个奖励增加对象包装。

## 唯一构建和注册流程

`experiments/base.py` 的 `Experiment` 提供唯一的 `build_env()` 与 `register()`。任务族通过 `create_env()` 创建基础环境，`runner_cls` 选择注册运行器。

构建顺序为：

1. 调用任务族工厂，直接创建最终配置类型。
2. 训练模式覆盖 `train_envs`；回放模式保留任务工厂的环境数。
3. 沿反向 MRO 从基类到子类，依次执行各类**自身声明**的 `reward_edit()`、`edit()`。
4. 收集并应用 `addons`；同类型扩展由子类替换父类，再按 `order` 排序。

每个类的配方钩子仅执行一次，`edit()` 内部不能调用 `super().edit()`。该约定只针对自动遍历的配方钩子；`build_ppo()` 仍采用普通的 `super()` 继承。

`RewardEdit` 保留删除、调权重、合并参数、添加／替换奖励的固定顺序，并校验被修改项存在。`register()` 统一注册训练环境、回放环境、PPO 配置和运行器，不再由具体实验复制注册流程。

## 实验继承与组合

```text
Experiment
├── UniversalLocomotion
├── WalkingExperiment
│   ├── PeriodicStepping
│   └── FlatWalking
│       └── VelocityTracking
│           └── VelocityEnergyTracking
│               └── OmniVelocityEnergy
└── TransformationExperiment
    ├── BipedToSnake
    └── SnakeToBiped
        └── SnakeToBipedBeta
            ├── SnakeToBipedArc
            └── MirroredStandingExperiment
                ├── SnakeToBipedRobust
                └── TeacherTrainingExperiment
                    ├── SnakeToBipedTeacher
                    └── SnakeBipedBidirectional
```

上图展示主干；行走调参与部署变体仍保留具名实验类和独立任务 ID。

`WalkingExperiment` 提供行走工厂及步频覆盖。`TransformationExperiment` 提供变形工厂与共同 Gaussian PPO 默认值。`BipedToSnake` 和 `SnakeToBiped` 是方向相反的兄弟，不再通过相反行为复用训练参数。

`MirroredStandingExperiment` 共用镜像动作、随机滚转、紧凑站立行为和鲁棒训练设置；任务层 `presets.py` 负责实际装配。`TeacherTrainingExperiment` 共用教师物理单位缩放所需的训练参数；单向教师与双向实验分别安装自己的观测和行为，双向实验不再继承完整单向教师。

扰动、观测噪声、延迟和域随机化继续由 `addons` 组合。`variants.py` 集中扰动及部署配方；`variant()` 仍支持命令行动态组合。变形任务不接受行走专用扩展。

Gaussian／Beta、六动作／三动作、普通／教师／指令观测对应实际模型接口差异，仍明确区分。

## 配置直接构造与共享字段

```text
TransformationEnvCfg
├── UniversalEnvCfg
├── BipedToSnakeEnvCfg
└── SnakeToBipedEnvCfg
    └── RobustSnakeToBipedEnvCfg
        └── BidirectionalEnvCfg
```

`transformation_scene_cfg(..., cfg_cls=...)` 直接构造最终 dataclass。方向工厂再安装各自行为，不再逐层创建配置并复制 dataclass 字段。起立工厂还接受动作配置类型与紧凑站立选项，以复用场景而避免先装配完整下降奖励再删除。

公共配置字段保留原默认值。`final_max_body_height` 不仅用于正向下降，也用于反向恢复和双向下降条件，因此保留在共享配置中。配置继承与实验继承分别表达字段复用和训练配方复用，不必机械保持相同树形。

## 状态更新时序

`TransformationState.reset(ids, step)` 统一清空保持时间并记录控制步；`FallRetryState.reset(ids, step)` 管理跌倒恢复相关字段，调用方不再逐项清空其内部状态。

`TransformationState.advance_hold(valid, step, dt)` 用 `last_step` 保证同一控制步仅累计一次：条件有效则累加，否则归零。下降、起立和鲁棒站立提供各自目标条件，保留原有保护与课程顺序。

双向任务调用 `advance_hold(..., commit=False)`，先累计目标保持时间，再用同一份“本步尚未更新”标记推进指令切换计时，最后提交 `last_step`。不能提前提交，否则会跳过切换计时。指令切换通过状态对象的重置方法清理旧目标计时和恢复状态。

## 归并后的模块

- `tasks/transformation/poses.py` 集中蛇形与弧形姿态常量，数值不变。
- `tasks/transformation/actions.py` 同时包含独立动作父类和镜像动作子类，保留各自动作维数和关节映射。
- 下降任务 ID 和配置工厂直接定义在 `tasks/transformation/env_cfg.py`，删除纯重导出包装。
- `experiments/walking.py` 集中基础、速度、能效和全向行走配方；`variants.py` 集中扰动及部署配方。
- `rl.py` 集中 `ppo_cfg`、`HotRewardController`、`SnakeToBipedHotRewardRunner` 和 `RobustSnakeToBipedRunner`，任务模块不再承担训练器职责。
- `reverse_observations.py` 与 `teacher_observations.py` 保持独立，避免为减少文件数引入循环依赖。

```text
src/zbot_rl_mjlab/
├── __init__.py
├── robot.py
├── cli.py
├── viewer.py
├── rl.py
├── teachers.py
├── distillation.py
├── tasks/
│   ├── __init__.py
│   ├── common.py
│   ├── walking/
│   │   ├── __init__.py
│   │   ├── env_cfg.py
│   │   ├── mdp.py
│   │   └── actions.py
│   ├── universal/
│   │   ├── __init__.py
│   │   ├── env_cfg.py
│   │   ├── commands.py
│   │   └── mdp.py
│   └── transformation/
│       ├── __init__.py
│       ├── env_cfg.py
│       ├── mdp.py
│       ├── actions.py
│       ├── poses.py
│       ├── presets.py
│       ├── snake_to_biped.py
│       ├── robust.py
│       ├── commanded.py
│       ├── roll_reset.py
│       ├── reverse_observations.py
│       ├── teacher_observations.py
│       └── std_projection.py
└── experiments/
    ├── __init__.py
    ├── base.py
    ├── common.py
    ├── addons.py
    ├── walking.py
    ├── variants.py
    ├── walking_tuning.py
    ├── flat_foot_velocity_energy.py
    ├── transformation.py
    ├── bidirectional.py
    └── universal.py
```

## 综合任务的组合复用

`UniversalLocomotion` 直接继承中性 `Experiment`，通过 `create_env()` 装配综合任务、`runner_cls` 选择 `MultiTeacherRunner`，不继承某个已有行为配方。`UniversalEnvCfg` 继承共享变形场景字段，复用全身接触场景和六独立目标速度动作，替换指令、观测、奖励和终止条件。`UniversalCommand` 管理目标构型、四阶段、平滑速度、保持时间、手动控制和速度课程。

`TeacherEnsemble` 位于训练支持层，按学生状态重建三个旧教师输入并转换输出；任务层不依赖教师。`MultiTeacherPPO` 管理监督缓存、Actor 预热和联合 PPO；`MultiTeacherRunner` 在训练时加载教师并核对续训来源，回放仅使用学生。现有 19 个实验配置保留不变。综合任务的 50／53 维观测、六维 Beta 动作构成新模型接口，旧检查点通过教师参与训练。使用方法和限制见[综合运动说明](universal-locomotion.md)。

## 兼容性与验证边界

重构保持任务 ID、实验名、观测项顺序及维度、策略分布、动作映射、奖励参数、传感器、控制周期和训练／回放差异。模型权重与归档原始 YAML 不变；历史配置中的项目回调模块路径由包内兼容别名解析，不保留旧转发文件。此兼容只覆盖项目符号，不意味着 MJLab 历史 YAML 中的所有标签均可直接反序列化。

配置基线、继承钩子、状态时序、历史回调和模型加载由回归测试覆盖；教师起立及双向切换另有 GPU 回放检查。测试与回放证明接口和所覆盖行为保持兼容，不证明策略收敛或实机可用。类型检查仍有既有诊断，具体运行结果见[清理与验证记录](cleanup-report.md)。
