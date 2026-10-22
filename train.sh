source /home/yhzhu/AI/mjlab/.venv/bin/activate

# 0、按照指定频率原地踏步实验
# python train.py Mjlab-Zbot-6dof-Periodic-Stepping \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 51

# 1、按照指定频率迈步+基本平地行走
# python train.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 600

# 2、按照指定频率迈步+平地行走+扰动力和力矩
# python train.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 600 \

# 2.5、跟踪频率 + 速度指令联合跟踪
# python train.py Mjlab-Zbot-6dof-Velocity-Frequency-Tracking-Walking \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500 \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     # --env.test-frequency 1.0 \

#新建一个实验，跟踪速度指令，但频率不跟踪，增加节省能量的奖励
# python train.py Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500

#继承 Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking，只增加外力扰动和物理随机化，保留速度跟踪、无频率跟踪和能量奖励配置。
# python train.py Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)" \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500

#全方位行走实验
# python train.py Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500 \
#     "$@"

#实机部署实验
# --disturbance	基座受推
# --domain-randomization	足底摩擦、编码器零偏、基座质心
# --mass-randomization	质量和惯量
# --gain-randomization	PD 增益
# --observation-noise	actor 观测噪声和偏置
# --action-delay [N]	动作延迟 0 到 N 步,默认 2,只在训练时生效
# --blind-actor	actor 看不到 base_lin_vel、yaw_error
# --deployment	以上全部
# python train.py Mjlab-Zbot-6dof-Deploy-Flat-Foot-Walking \
#     --deployment \
#     --env.scene.num-envs 8192 \
#     --agent.max-iterations 500 \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,0.5)"

# 7、双足到蛇形的变形实验
# python train.py Mjlab-Zbot-6dof-Transform-Biped-To-Snake \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 200

# 8、蛇形到双足的变形实验
# python train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 1000

# 9、站立目标同侧的浅弧起点 → 双脚底着地 → 双足站立
# python train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 4000

# 10、从头训练：3维镜像动作 + 随机roll，暂时关闭base扰动力/力矩。
# python train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 2000

# 11、全状态教师：36维Actor / 44维Critic，3维镜像动作，从头训练。
# python train.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 200

# 12、蛇足双向变形：随机蛇形/双足初态，+1站立 / -1爬下，37/45维观测。
# python train.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 600

# 13、综合运动：多教师蒸馏预热，随后联合强化学习。
# python train.py Mjlab-Zbot-6dof-Universal-Locomotion \
#   --agent.resume True \
#   --agent.load-run 2026-10-18_11-17-41_final_objective_check \
#   --agent.load-checkpoint model_698.pt \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 1200

# 14、综合运动：多教师蒸馏预热，随后联合强化学习。
# python train.py Mjlab-Zbot-6dof-Universal-Locomotion-Deploy \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 3000

# python train.py Mjlab-Zbot-6dof-Universal-Locomotion-Deploy \
#   --agent.resume True \
#   --agent.load-run 2026-10-18_12-45-51_final_student_straighten_relearn \
#   --agent.load-checkpoint model_500.pt \
#   --agent.run-name speed_course \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 3000

# python train.py Mjlab-Zbot-6dof-Universal-Locomotion-Deploy \
#   --agent.resume True \
#   --agent.load-run 2026-10-18_13-23-48_speed_course \
#   --agent.load-checkpoint model_1000.pt \
#   --agent.run-name directional_course_v2 \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 300

# 15、蛇形运动：覆盖前进、后退、左右侧移、双向转动及组合运动
# python train.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 500 \
#   --agent.run-name translation_stall_v2

# python train.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 1500 \
#   --agent.run-name beta_body_v3

# python train.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 1500 \
#   --agent.run-name beta_body_v3
  # --agent.resume True \
  # --agent.load-run 2026-10-19_16-45-28_beta_body_v3 \
  # --agent.load-checkpoint model_200.pt \
  # --env.allow-resume-joint-limit-change True \
  # --env.actions.joint-pos.target-limit 3.141592653589793 \

# 16、蛇足全能部署学生：腿走 + 蛇前爬 + 双向变形，四教师从头蒸馏。
# python train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy \
#   --env.scene.num-envs 16 \

# 17、纯双足全向训练。
# python train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Wide-Biped \
#   --log-root logs/ablations/biped_scalar_only \
#   --env.commands.transform.mode stand \
#   --env.scene.num-envs 8192 \
#   --env.omni-frame-backend cuda_graph \
#   --agent.max-iterations 6000 \
#   --agent.run-name biped_scalar_only_v1

# 18、新蛇形全向训练：前后、侧移、转向及组合运动。
# python train.py Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 6000

# 19、新蛇形全向训练：前后、侧移、转向及组合运动。自适应权重
# python train.py Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 6000

# 20、base靶标蛇形全向：继承19的奖励和自适应调权，独立模型与日志。
# python train.py Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher \
#   --env.scene.num-envs 8192 \
#   --agent.max-iterations 6000

# 特权全能教师：按任务归档指导 + PPO。
# python train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided \
#   --log-root logs/rsl_rl_guided \
#   --env.commands.transform.mode source \
#   --env.scene.num-envs 8192 \
#   --env.omni-frame-backend cuda_graph \
#   --agent.max-iterations 6000 \
#   --agent.run-name task_guidance_ppo_v2 \
#   "$@"