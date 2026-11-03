# 闭环迁移断开力矩实验

## 当前：距离雅可比自动拉开＋MJLab内置PD（2026-10-27）

用户要求目标weld解除后直接执行确定性拉开动作。当前注册模式`jacobian_loading_deterministic_distance_pd_v3`：闭环加载仍为雅可比力矩前馈+PPO残差，weld解除后PPO输出不再参与控制，Actor损失、熵/KL与Actor统计仅选择加载步骤；Critic/GAE覆盖全部步骤。旧模式检查点保留并拒绝直接加载。

断后由GPU距离雅可比控制器生成关节速度，不依赖PPO探索。五个完整weld把六个独立模块约束成虚拟串联链，固定断开端一侧的根；用各关节世界轴、铰点及另一个端口位置构造相对位置Jacobian。公共刚体运动不改变距离。设delta为端口位置差、n=delta/|delta|，距离行Jacobian为nᵀJ_delta；距离不足1mm时使用源端口外法向启动，避免重合点的距离导数不定。基于实测几何每5ms重算，属于忽略weld柔度的局部运动学近似，不是避碰/力可行轨迹规划。

速度为阻尼最小二乘`dq=J_dᵀ*v/(J_d*J_dᵀ+lambda²)`。默认距离速度上限.02m/s、距离误差增益2/s、内部目标.051m（确保越过原严格>.05阈值）、lambda=.005m/rad；关节速度上限1rad/s、加速度上限.5rad/s²。积分目标角裁剪物理范围，目标角相对实际角误差最多.05rad，避免阻塞时积分累积。

**执行器使用MJLab自带`BuiltinPositionActuatorCfg`，不是显式IdealPd。** 加载时MuJoCo原生参数为gain=1、位置bias=0、速度bias=−1.5，ctrl为加载力矩；断后切换原生位置PD gain=15、位置bias=−15、速度bias=−1.5，ctrl为积分目标角。由原生implicitfast处理PD导数。每环境gain/bias模型参数通过MJLab expand_model_fields一次展开，逐子步GPU更新，无依赖修改。20Nm限制作用于原生净力矩，控制信号本身不作20Nm裁剪（PD目标角不是力矩）；总需求与实际14/20Nm/1秒保护保持。

目标weld解除后的下一物理子步捕获实际角初始化目标角，随即按控制器速度积分。>5cm锁存等待，关闭P与前馈，只保留原生D=1.5；2秒内原安全规则保持，等待结束reset，允许间距因撤P后略回落，沿用原等待语义。其他五面>64N仍失败reset。奖励和43/45维观测保持，增加分离指令速度与预测距离速度Metrics。

验证：固定IK初态64环境，PPO残差始终为0，64/64真实拉开超过5cm并完成2秒安全等待，失败0；物理子步峰值5.004–5.009cm，全部完成约5.28秒。证据`outputs/reconfiguration/disconnect_torque/distance_jacobian/verification_64.json`，详细回归与限制见project-memory。本方案不宣称对随机姿态、接触条件或不同目标面泛化。使用`bash train.sh`重新开始新模式；原模型和运行日志保留。

## 历史阶段：目标weld解除后切换速度积分PD（2026-10-27）

本节覆盖下文控制说明。用户确认切换事件为目标weld刚解除，非间距超过5cm。新模式`jacobian_loading_velocity_pd_unified_ppo_v2`：闭环仍雅可比前馈+PPO残差力矩，P0/D1.5；目标weld解除后下一5ms物理子步切换原生IdealPd，Kp=15Nm/rad、Kd=1.5Nm·s/rad。切换时q_target=q_actual，剩余子步速度命令为0，避免将加载力矩动作误当速度；下一20ms PPO步开始，action解释为目标角速度，最大1rad/s，各5ms子步积分q_target并裁剪物理关节范围。PD速度目标为0，保持项目原速度积分位置控制语义。

同一个Actor负责两种动作语义。断后tau_J和残差力矩均为0，力矩来自PD；14Nm连续、20Nm短时1秒、总需求>20Nm失败与实际力矩保护均覆盖PD。等待/安全失败关闭P和前馈，仅保留原D1.5。>5cm清零、2秒安全等待/reset语义保留。每世界PD模式、目标角与速度缓存支持独立reset。

为了显示积分内部状态，原37维Actor追加(q_target−q)/π六维，共43维；Critic45维。新增separation_pd_active与target_max_gap_m（回合最大）Metrics，原target_gap_m仍为回合均值。新模式拒绝旧37维纯力矩PPO检查点，旧文件保留，需重新开始。训练超参数暂不改：原std=.00025在PD阶段对应.00025rad/s，仍是很小的速度探索；本次没有宣称速度积分PD本身已解决探索效率或分离学习。

## 历史阶段：雅可比前馈＋统一残差PPO（2026-10-27）

本节为当前配置，下文保留历史。Task注册`jacobian_ppo.JacobianPpoRunner`，模式`jacobian_feedforward_unified_residual_ppo_v1`。同一个Actor贯穿加载、断开与分离，不再冻结教师接管加载，不依赖旧静态模型。旧检查点保留，不能直接作为新模式续训或回放。

原生执行器控制为`tau = clamp(tau_J + 20*action - 1.5*dq, -20, 20)`Nm，P=0。固定IK闭环下利用MuJoCo约束Jacobian、质量矩阵M和约束正则R，解`(J M^-1 J^T + R) dλ/dτ = -J M^-1 B`，转为六端面局部wrench响应。CPU初始化求一次参考力矩；运行时Torch GPU根据实测载荷调整增益，没有每环境每步CPU优化。

该模型是固定接触活跃集下的局部瞬时动力学线性化，并非任意姿态的精确静力逆解。目标Mx/My方向135°，瞬时倾转载荷投影50N、其他面预测≤60N，最小化力矩得到约`[2.373,-1.108,-.525,.525,1.109,-2.371]`Nm。独立GPU闭环保持诊断中稳态目标约84N、其他面峰值约60.9N；瞬时50N与稳态84N不能混作同一预测。

仅六面全连接、关节偏离参考<.05rad且无安全失败时启用前馈；任意weld解除后下一5ms物理子步撤前馈，同一个PPO继续控制。增益依据目标84N与其他面60N软上限修正，范围[.1,1.5]。其他面>64N失败/reset、48N起接近惩罚、14Nm连续/20Nm最多1秒及>20Nm需求失败保持。超过5cm后全部前馈清零、保留阻尼、安全等待2秒。

Actor37维：原关节角6、角速度6、六面等效剥离力6、实际关节力矩6、连接状态6、间距1，再加当前雅可比前馈/20Nm六维。实际力矩仍除14Nm。Critic再加等待标志和时间比例，共39维。

`ResidualMaskedPPO`保留RSL-RL PPO的GAE、裁剪策略/价值损失与Adam。加载和分离都训练Actor，仅强制等待屏蔽Actor损失/熵/KL及Actor统计；Critic与GAE使用全部步骤。新Actor输出头归零，std=.00025（.005Nm），范围[.0002,.02]。24步/轮、5epochs×4minibatches、gamma=.995、lambda=.95、entropy=.005。初始lr=1e-6，自适应范围[1e-7,1e-4]。64环境单轮平均KL约.0147；初试lr1e-4产生KL约284，已纠正，短测不能保证长期稳定。

奖励、物理和六面载荷Metrics保持。`bash train.sh`保留4096环境。专项检查覆盖局部导数、37/39维观测、加载参与Actor训练、断后撤前馈、等待mask及新模型保存恢复。独立两环境回放单轮模型，目标均在首个20ms断开；其中一个最大间距约8.4mm后其他面超载失败，另一个在检查停止时约.245mm。尚未证明学会完整5cm分离和2秒安全等待。

## 历史阶段：冻结加载教师＋MJLab/RSL-RL PPO分离（2026-10-27）

本节覆盖下文历史训练器说明。同一Task已改为`teacher_separation.TeacherSeparationRunner`，算法为RSL-RL `PPO`子类`TeacherMaskedPPO`，保留标准Actor/Critic、GAE、clipped surrogate/value loss、Adam和自适应KL；共享MJLab/RSL依赖不改。旧高斯均值训练器保留，但不再是本任务当前注册runner。

`agent.teacher_checkpoint`指向已验证84.4N静态六维均值模型。教师冻结且不加入优化器：目标面仍连接时输出教师动作；weld关闭后，学生Actor根据31维观测输出六关节前馈；超过5cm后Action逐物理子步强制零前馈，保留D=1.5、等待2秒，原成功/失败/reset规则不改。教师实际上是已学到的六维固定均值，不引入额外教师网络或监督模仿。

每条transition按动作前真实控制阶段记录mask。教师与强制等待步骤只用于Critic和完整GAE；Actor的surrogate、熵、KL、自适应学习率判断、优势归一化及观测归一化仅使用学生实际控制的步骤。所有mask为false时Actor连同Adam动量、std与归一化统计保持不变。等待结束的成功/失败仍沿GAE传给之前分离动作，不能删掉这段回报。

Actor观测仍31维；Critic额外得到等待标志和已等待比例，总33维，以区分等待倒计时。Actor新建256/128 ELU网络，输出头归零作为安全初始均值；高斯log-std初始.00025、范围[.0002,.02]。当前24步/轮、5epochs×4minibatches、lr1e-4 adaptive、gamma .995、lambda .95、entropy .005。当前实现限定单GPU、前馈MLP，无RND/symmetry扩展；本机RTX5090工作流满足此范围。

新模式`frozen_loading_teacher_masked_separation_ppo_v1`，模型内保存教师六维动作及源SHA，回放/续训不再依赖原教师文件。旧静态/两常量分离阶段模型不能直接load新阶段，仍保留。使用`bash train.sh`启动；`run.sh`需选新PPO检查点。

真实16世界单轮24步PPO链路通过；mask专项验证教师/等待不改Actor（含已有Adam动量）、学生阶段可更新Actor、等待末奖励能通过GAE影响之前学生动作、缺失原教师文件仍能加载回放。尚未验证完整5cm安全分离学习成功。

## 历史阶段：两段常量均值分离（2026-10-27）

本节覆盖下文静态阶段的历史说明。当前`enable_weld_failure=true`，六面完整等效载荷`max(Fz,0)+sqrt(Mx²+My²)/.05m`达到80N立即关闭对应weld，保持原生动力学，不改运动学树、不写qpos、不加解锁动作。

两段事件切换策略：目标面连接时使用已验证84.4N静态模型的固定六维加载均值；目标面关闭后，从下一控制步开始使用独立六维分离均值。只训练后者，加载点不再探索/更新。分离初值为0；`initial_checkpoint`明确指向`outputs/reconfiguration/disconnect_torque/static_gaussian/penalty02/verification_model.pt`。策略根据31维观测中的目标连接标志切换，无状态反馈网络，分离阶段目前仍是常量力矩策略，尚不能据此宣称普适动态重构。

目标端口间距>5cm后，原Action锁存零前馈（D=1.5阻尼保留），安全等待满2秒后success +10并reset。其他面>64N、错误面断开、力矩超限或其他异常随时failure −10/reset；4秒主动回合超时在等待期屏蔽。六面恢复连接后进入下一试验。

分离奖励改为距离进展：potential=clip(gap/.05m,0,1)，控制步内正增量给`target_separation`（+5），负增量给`penalty_separation_regression`（−5）；除以step_dt后由奖励管理器积分，因此完整到5cm净进展奖5，闭合往返净奖0。进入等待后potential冻结，不重复发分离奖励。目标加载奖仍只看真实倾转力矩，不保留断裂历史发奖。其他惩罚权重不改（安全−.2、力矩−.01、失败−10）。

高斯采样/候选验证改为完整试验回报，最多300控制步/6秒，全部样本终止则提前结束；保留每世界终止掩码，不把自动reset后的回合计入原样本。每10次更新及训练末长验证同为300步。分离探索std=.00025（20Nm尺度下每关节.005Nm）、下限.0002，均值向量步长.0025（总力矩向量.05Nm），8次回溯。相比静态即时加载，动态分离需要运行整段过程，采样耗时会增加；日志给出各部分耗时。

新checkpoint模式`gaussian_load_then_separate_v1`，保存冻结加载均值和可训练分离均值；旧静态模型只允许作初值，不允许误作新阶段续训/回放。Metrics恢复target_broken/success/waiting/wait_elapsed_s；训练日志增加成功率、断开率、进入等待率、最大分离间距、真实断裂载荷及峰值倾转载荷。

实际诊断：原84.4N均值在首个20ms控制步可断开目标；断后继续原力矩约80ms会错断其他面，断后零前馈则安全但4秒仅约.3–.4mm间距。简单缩放旧向量也未验证安全5cm，故不宣称静态解已经解决分离。旧分离实现及安全保护已通过受控状态机测试；新阶段学习成功需另行评估。

## 历史训练器：六维高斯均值即时策略梯度（2026-10-27）

当前Task保持不变，运行器已切换为`static_gaussian.StaticGaussianRunner`，新模型模式`static_gaussian_tilt_mean_v1`。训练只更新六维高斯均值，不构建PPO、GAE、Critic或观测网络，不使用教师/规划/监督数据。31维环境观测继续用于显示与物理诊断，当前固定构型策略输出不依赖这些观测，不能据此宣称获得扰动反馈或泛化能力。

卡片`agent.algorithm.initial_checkpoint`明确指向上一轮`2026-10-27_08-28-39/model_400.pt`：仅在复位观测上读取该已训练PPO策略的初始输出作为六维均值，记录源SHA；旧网络不参与后续更新。清空该参数可从seed42的小随机均值开始。新模型不能用旧PPO方式续训/回放；旧文件保留。

每批从同一个真实IK复位状态执行成对高斯力矩样本，默认固定执行10控制步（200ms），平均实际即时奖励作为该静态试验回报，无折扣/价值自举。以`mean(noise*(reward-baseline)/std²)`估计高斯得分梯度，按向量范数.02（20Nm尺度下总力矩步长.4Nm）提出均值更新；候选用同样10步物理试验验收，回报增加且无失败才接受，否则最多8次折半。初始化std=.02，出现样本安全失败或所有候选被拒绝时减小探索半径，下限.002；均值步长不受std或KL绑定。

每10次更新及训练结束，用200控制步/4秒验证均值。`best.pt`只保存无失败的最佳长验证结果；长验证失败回退到已验证均值。其他载荷/电机保护完全由原环境执行。失败后的试验不再把自动reset新回合计入原样本回报。保存六维均值/std、局部随机生成器、更新数、已验证最佳值和源模型信息，可恢复；显式CLI新探索配置优先。继承配置中的Actor/Critic设为null，未分配神经网络或rollout storage。

日志`StaticGaussian/*`给出回报/载荷/采样失败率和采样、候选验证、长验证耗时；`StaticReward/*`保留`penalty_`项名称，记录试验内平均加权奖励率（与原PPO按4秒回合归一化的Episode_Reward口径不同）。训练脚本仍显式4096环境；相同入口`train.sh`和`run.sh`，回放请选择新模型。

## 历史阶段：闭环静态加载（2026-10-27）

用户要求暂时去掉weld失效、断开运动和呈现流程，只学习前面的闭环内力加载。当前卡片`enable_weld_failure=false`：所有六条weld在整个回合保持激活，目标面超过80N也不关闭约束；不记录断裂峰值、不进入5cm前馈清零/2秒等待流程。不是冻结qpos或使用新的静力学求解器，仍由原MuJoCo动力学/约束求解计算反力；不宣称真实姿态绝对不动。

活动奖励仅`target_loading`（weight1，仅奖励倾转力矩等效拉力，上限1.2，仍有其他面安全门控）、`penalty_safety_margin`（−0.2）、`penalty_torque`（−.01）、`penalty_failure`（−10）。删除当前配置中的`target_separation`与`success`，移除等待终止及断开/成功/等待显示；保留六面载荷并新增`target_load_utilization`。其他面>64N安全reset、14/20Nm与1秒保护、4秒超时reset保留。

控制P0/D1.5、前馈20Nm、31维观测及训练脚本当前4096环境不改；连接标志在本阶段为常量1。原PPO静态阶段模式为`torque_feedforward_p0_static_tilt_loading_obs31_v10`，已由上节六维高斯训练器替代；旧断开策略文件保留且拒绝误加载。旧断开实现保留供后续恢复，以下有关分离/等待/成功的叙述是历史流程，当前不启用。

独立任务 `Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque`。不修改 Basic 或规划教师实验。第一阶段固定首尾连接面（编号5），内部五面编号0–4；尚未推广随机目标面。

## 当前目标奖励：只奖励力矩生成的拉力

`F_tilt=sqrt(Mx²+My²)/.05m`；`target_loading=clip(控制步F_tilt峰值/80N,0,1.2)*其他面安全门控`。Mx/My为目标源端口局部倾转力矩，Mz扭转不计；直接拉力max(Fz,0)完全不计入目标奖励，历史断裂值也不计。80N只作归一化基准，等价于倾转力矩4Nm的归一化；奖励上限1.2对应4.8Nm/96N。不要求目标面真正断开。

六面载荷观测、其他面48N接近惩罚/>64N reset仍使用完整`max(Fz,0)+F_tilt`，不改变安全物理含义。Metrics新增`target_tilt_load_N`与`target_direct_tension_N`分别显示目标面力矩等效拉力、直接正拉力；原`connector_5_load_N`和`target_load_utilization`仍显示总载荷，日志为回合均值。31维观测不改。

## 入口

```bash
bash train.sh
bash run.sh
```

两脚本当前启用24.d，支持追加CLI参数。训练日志独立保存至 `logs/rsl_rl_disconnect_torque`，每10轮保存；回放需要该实验自己的检查点。代码交付时未运行力矩训练，也没有已学会断开的模型。

重新生成静止闭环逆运动学初态（不训练）：

```bash
./scripts/python.sh scripts/prepare_disconnect_start.py
```

以已保存闭环规划分支终点为逆解初值，重新求解零间隙、180°注册的首尾闭合。六模块根位姿由同一原串联模型正运动学生成，按网格最低点放置地面；所有根速度、关节速度与初始电机力矩均为0。复位激活全部六条weld，包括首尾连接面5。状态文件 `assets/reconfiguration/closed_loop_start.json` 保存该逆解；源学生检查点SHA仅用于Actor特征迁移，不再决定初始姿态或力矩。

## 控制与安全

六个独立自由根模块、六个电机关节、六条可断开的site weld。当前使用MJLab原生`IdealPdActuatorCfg`，其控制律为`Kp*(q_target-q)+Kd*(dq_target-dq)+tau_ff`。本实验Kp=0、Kd=1.5、dq_target=0，策略输出`tau_ff=20*clip(action,-1,1)`Nm，原生执行器将前馈与阻尼之和限幅±20Nm。不再积分目标角或保持位置，不在Action内重写PD；每物理子步由执行器统一计算。卡片可调torque_action_scale/joint_damping/torque_limit；joint_stiffness必须0。penalty_torque代价读取真实qfrc_actuator。

当前Actor/Critic均为31维观测，依次为：关节角/π（0–5）、关节角速度rad/s（6–11）、六面当前等效拉力/80N（12–17）、实际电机力矩/14Nm（18–23）、六面连接状态0/1（24–29）、目标面间距m（30）。断开面当前等效拉力为0，不以历史断裂值代替；断裂值仍保留在内部用于奖励。已删除完整wrench、裕度、相对姿态与速度、重力、上一动作及目标one-hot观测。源学生只迁移q/dq特征，角度除π后相应变换归一化统计和首层权重，新的力矩头归零，Critic新建。

`target_loading`使用目标面当前控制步峰值与历史真实断裂载荷的最大值，上限1.2；乘安全系数`clip(其他五面最小裕度/0.2,0,1)`，其他面误断时归零。低于安全裕度要求逐渐减弱加载奖；该项本身非负，不奖励牺牲其他面加载。断裂仍在真实利用率1处被动触发，1～1.2来自断裂子步真实超调，不能凭空把已断开面加载到1.2。首次断裂载荷保留避免下个控制步加载奖突然归零，分离与成功奖励继续保留。

新模型标记`torque_feedforward_p0_peel_obs31_penalty_safety_v8`；旧31维无力矩时限/无等待流程、94维前馈、直接力矩、偏移PD和速度积分PD均拒绝按新语义回放/续训，文件保留。

每个5ms物理子步由约束Jacobian和约束力测量源模块port_out局部端面wrench。六个端口+Z均为源侧外法向，受拉时约束对源侧的反作用为+Z。按用户给定结构：中心磁铁80N，连接凸台限制面内滑移与绕法向旋转，有效半径0.05m。当前等效剥离力和利用率为：

`F_peel = max(Fz,0) + sqrt(Mx²+My²)/connector_radius`

`u = F_peel / break_force`

达到80N（u≥1）即关闭对应weld，恰好80N也触发。剪切Fx/Fy、法向扭矩Mz和压紧Fz<0不计入剥离；倾转Mx/My即使在压紧状态仍按其大小计入，不用压紧力抵扣倾转项。凸台限滑限扭目前由连接状态下的weld共同约束表示，未新增独立凸台接触几何。

卡片connector_radius=.05m，break_force=80N；删除独立break_torque参数。纯倾转的等效极限仍是80*.05=4Nm，但法向拉力与倾转按相加判据共同消耗容量：例如40N+2Nm/.05m=80N，不再分别用两个范数取max。31维观测中的六面等效拉力除以break_force；加载奖、安全裕度和断裂逻辑统一使用u。

其余五面安全目标为min(1-u)。目标weld断开且目标端口间距严格超过5cm，立即锁定该世界全部六关节策略前馈为0，从下一物理子步起生效；P=0/D=1.5原生速度阻尼仍保留，因此实际电机力矩不保证为0。新动作或间距再次缩小都不能恢复前馈，直到reset。

从首次越过5cm的子步之后计满2秒，再在控制步边界自动reset（训练和回放一致），恢复静止IK初态及六条激活weld，继续后续回合。原4秒超时在等待期间屏蔽，防止提前reset；错误面断开/异常状态仍立即失败。取消原2cm/0.2秒保持成功终止；目标间距>5cm且其余面全连接、最小裕度≥0.2时进入安全等待。整个回合逐物理子步检查其他面最小裕度，严格低于0.2立即锁存失败，使用penalty_failure权重发一次−10并在控制步边界立即reset；恢复到阈值之上也不会撤销本步失败。完整等待2秒且无违规才报告成功并发一次成功奖，然后reset。加载/分离等公式保留，等待期间仍统计奖励。加载奖励1.2上限仍只反映真实子步断裂超调。

卡片`separation_distance=.05`、`post_separation_wait_s=2`；Metrics另有`waiting`与`wait_elapsed_s`显示等待状态及已等待秒数。

拆分模块后显式恢复原串联树五组父子端面的碰撞排除，避免原本重叠的端面产生人工冲击。本阶段内部面断开即终止，因此这五组静态排除不适用于未来任意连接面分离后继续运动的实验，推广时需调整。

## 验证边界

任务注册/模型结构/力矩变换测试通过；GPU两环境初始化得到有限94维观测，控制力矩可下发。历史回放快照初态曾在恒定源力矩下约60ms内部超载；该初态已被静止零间隙IK替代。不能据此宣称已能安全保持或选择性断开。这里交付训练代码和入口，未调阈值、未训练或验证学习成功。

## 2026-10-26：修复连接面的角向柔度

旧weld即使ON、site间距很小，仍可能在零力矩下转开28.675°，因此此前只用连接开关和位置间距判定贴合的验证不充分。已将weld长度尺度由0.05改为1，solref改为(0.005,1)，solimp改为(0.999,0.9999,0.001,0.5,2)，提高连接约束刚度；六面破坏容量80N/4Nm不变。

静止IK初态零动作4秒：六面全连接，全过程最大角差0.008°，末步约0.001～0.003°；全过程最大利用率0.2633。新增GPU两世界1秒回归，同时检查每个端面位置误差<0.1mm、角差<1°及无断裂；5项专用测试通过。原生查看器状态栏新增六面间距与角差。

旧model_90在新初态/刚性约束下首个控制步就错误断面超载（最大利用率1.203），属于真实失败，不能通过仅更改显示修复。旧模型和日志保留，不能视为适配新物理，未启动新训练。诊断位于outputs/reconfiguration/disconnect_torque/weld_rigidity/。

## PD闭环验证（2026-10-26）

专用5项回归通过，包含GPU两世界零偏移保持、逐步端面间距/角差检查、固定参考偏移不累积、真实q/dq解析PD输出、子步反馈刷新与局部动作复位。未启动长期训练，尚未验证位置微偏移能学会安全断开。

当前速度积分PD版本已通过积分/限位反向/目标观测/反馈回归；49项专用与配置合同通过（排除Basic既有参数差异），viewer4项通过。两世界新建Actor零头回放4秒，六面全连接，末步最大面角差约2.04e-5°。接触容量nconmax=128，无overflow。此为控制和初态验证，不是学习成功证据。

## 5ms物理步长（2026-10-26）

按用户要求，本实验物理步长改为0.005s，decimation=4，控制周期仍0.02s。每轮物理子步数减半；实际迭代耗时尚未测量，不保证整体训练时间减半。目标速度上限0.1rad/s、每控制步最大目标增量0.002rad以及PD增益、weld容量、训练参数均不变。运行中进程不会热加载，重启后生效。

## EPA碰撞边界缓冲（2026-10-26）

`Warning: EPA horizon = 24 isn't large enough.`来自MuJoCo Warp凸网格穿透检测，表示EPA边界工作区不足，代码会终止该次EPA扩展并设置EPA_HORIZON溢出位。它不同于接触数量nconmax，也不是PD或weld容量。

本实验SimulationCfg新增epa_horizon_capacity=64。当前库没有公开参数，项目内collision.py适配器在本模型的凸碰撞临时数组分配期间应用该容量，然后立即恢复上游24；CUDA图重建/非图执行也适用。其他模型无标记则保持上游容量。未修改共享依赖文件、碰撞网格、接触排除或求解精度。适配依赖当前MuJoCo Warp私有接口，未来升级需重新验证。

1024环境、随机速度动作std=.1、200控制步（4秒）检查GPU overflow位，未出现EPA_HORIZON。只覆盖短测轨迹，不保证所有后续训练构型。重启训练/回放加载；再次溢出时应保留触发状态继续诊断，而非屏蔽警告。

## 原生前馈PD验证（2026-10-26）

7项专用测试通过：IK/模型/力矩坐标、原生IdealPd的P0阻尼与前馈叠加、总力矩限幅、GPU静态连接、面载荷观测、安全门控、1.2奖励封顶及真实断裂载荷保留。未验证主动选择性断开或训练收敛，未启动长期训练。此前速度积分/固定偏移章节为历史记录。

## 剥离判据验证（2026-10-26）

历史剥离判据版本（严格>80N）验证：10项专用检查通过，包含正拉/压紧符号、六端口外法向、拉力与双轴倾转组合、剪切/扭转排除、半径缩放、GPU被动开关（80N保持、81N仅目标断开）、原生前馈PD与安全奖励。控制模式标记更新为torque_feedforward_p0_peel_v4，旧范数判据模型因载荷观测语义变化被拒绝误加载。未启动训练或验证学习成功。

### 连接面载荷指标

Metrics新增`connector_0_load_N`至`connector_5_load_N`，显示六面当前等效剥离拉力（N），与断开判据同公式；0–4为内部面，5为首尾目标面，80N为容量。断开后当前载荷显示0。Viewer显示当前控制步采样值，训练日志`Episode_Metrics/connector_i_load_N`为回合内控制步均值，不是峰值或历史断裂载荷。

### 其他连接面纯惩罚与安全reset

原`other_safety_margin`正向奖励与越界混合项已替换为纯`penalty_safety_margin`，weight=-0.2，函数非负。最小安全裕度阈值仍为.2，新增`safety_penalty_buffer=.2`；用控制步内其他五面峰值利用率得到最小裕度m_peak，原始代价`max((.4-m_peak)/.2,0)^2`。裕度≥.4（等效拉力≤48N）无惩罚，.4至.2逐渐增大惩罚；m_peak=.2时原始代价1。

整个回合任一其他面裕度严格低于.2（等效拉力>64N）立即锁存`safety_margin_violation`，发`penalty_failure`一次−10并在控制步边界reset，包含加载、分离和等待阶段。恰好64N仍允许，不因float32的1-u舍入误判。安全裕度奖励不再产生正分。

所有本实验负权重奖励统一以`penalty_`命名：`penalty_safety_margin`、`penalty_torque`、`penalty_failure`，因此Viewer奖励面板和Episode_Reward日志均带penalty标签；其他实验不改。成功奖励仍为`success`，终止条件仍名`failure`。

### 移除动作变化代价

断开实验已删除`action_rate`奖励项，不再惩罚相邻控制步的策略动作变化；当前前馈尺度20Nm、总硬限幅±20Nm、连续阈值14Nm与原生D=1.5；详见力矩工作范围。

### 力矩工作范围：连续14Nm、短时20Nm/1秒

每关节按实际电机力矩绝对值逐5ms子步独立计时：≤14Nm无penalty_torque代价、清除连续超载计数；14<|τ|≤20Nm允许短时使用，连续达到1秒时触发failure −10并reset。计时是连续时长，不是热模型或滑动窗口累计；回到≤14Nm重新计时。等待阶段的阻尼力矩也纳入检查。

前馈尺度20Nm使策略可以主动请求20Nm。原生执行器硬限幅±20Nm，另外检查限幅前总需求`20*clip(action,±1)-1.5*dq`，严格超过20Nm或实际力矩严格超过20Nm均锁存失败，避免硬裁剪隐藏超限。需求违规时立即清零策略前馈，实际输出仍由原生阻尼及±20Nm硬限幅计算；reset在控制步边界执行。

`penalty_torque`函数为每物理子步`mean((max(|τ_actual|-14,0)/6)^2)`的控制步峰值，配置权重仍−0.01；惩罚函数非负、负权重。不再惩罚14Nm以内正常输出。实际力矩观测仍除以14Nm（超过连续阈值时可大于1），31维不变；超载计时暂不进入观测。Metrics新增`torque_violation`与`max_torque_overload_s`。卡片`continuous_torque_limit=14`、`torque_limit=20`、`torque_overload_duration_s=1`、`torque_action_scale=20`。

### 8192环境显存不足诊断（2026-10-27）

当前RTX 5090 32GB上，8192环境可创建场景但首轮训练的`sim.forward()` Warp图执行报CUDA error 2 out of memory。4096环境完成一轮24步PPO（约1.59秒）；核对时训练脚本已是显式4096，保留，卡片默认1024不变。此为短训练链路验证，不保证所有后续接触构型的显存峰值或学习收敛。诊断日志`/tmp/zbot-disconnect-{8192,4096}-diagnostic.log`，未更改共享环境、碰撞/约束容量或算法设置。

## GPU与性能诊断

原生weld/接触/约束求解由MuJoCo Warp CUDA图在GPU执行；端面反力提取为GPU Torch算子，Python负责组织发射。已将六面efc筛选/Jacobian提取/坐标变换合并批处理，物理配置不变。固定安全载荷短基准：64环境控制步约21.97→18.82ms，4096环境32.24→30.60ms（约5%整步改善）；4096读数与原六次公式一致，64最大力误差约9.54e-7N、力矩2.38e-7Nm。

4096环境诊断中，去掉Python载荷读数但保留真实weld物理解算约25.13ms/步，去掉所有子步指标约23.67ms；后者仍包含物理步进、状态刷新和普通环境流程，不可全归因于weld。完整reset约5.77ms。批量读数仍约5.5ms，有进一步融合空间；200步长验证本身约6秒量级，是新训练器可见的周期性耗时。此为单一构型短测，不承诺所有训练状态相同速度或8192不再OOM。证据`outputs/reconfiguration/disconnect_torque/performance/{baseline,report}.json`。

## 70N平台后的配置调整（2026-10-27）

用户同意把`penalty_safety_margin.weight`从−2减为−.2，并将`backtrack_attempts`从4增为8。在默认动作步长.02下，最小回溯候选由.0025降到.00015625。48N起接近惩罚、其他面>64N立即失败reset、20Nm/1秒电机保护都保留，不放宽硬安全约束。

原六维高斯模型可用新任务卡继续训练；恢复时保留均值/RNG而重新验证现行回报，旧惩罚权重不会从checkpoint覆盖当前配置。要接着原运行，可在train.sh后追加`--agent.resume True --agent.load-run 2026-10-27_09-00-11 --agent.load-checkpoint model_49.pt`。

该调整短对照：从正式高斯model_49继续5次更新，4096环境4秒确定性验证目标倾转拉力平均84.39N、其他面最小裕度.2398、无失败。诊断报告与验证模型位于`outputs/reconfiguration/disconnect_torque/static_gaussian/penalty02/`；仅固定初态短验证，不代表随机探索或扰动泛化。正式长训练尚未启动。

新阶段入口仍为`bash train.sh`，自动使用84.4N静态点初始化加载段；不要把旧静态checkpoint当成新阶段resume。短调试模型可用`bash run.sh --checkpoint-file outputs/reconfiguration/disconnect_torque/separation/verification_model.pt`回放。该模型仅短调试1次更新：目标约20ms关闭，其他面保持，4秒间距约0.9mm，尚未达到5cm成功标准；不要据此宣称分离已学会。
