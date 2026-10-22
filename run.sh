source /home/yhzhu/AI/mjlab/.venv/bin/activate

# 0、按照指定频率原地踏步实验（使用本任务独立日志目录中的模型）
# python play.py Mjlab-Zbot-6dof-Periodic-Stepping \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8
#     --checkpoint-file bests/periodic_stepping_2026-10-22_iter50/model_50.pt

# 1、按照指定频率迈步+基本平地行走
# python play.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8 \
#     --checkpoint-file bests/zbot_walking/model_599.pt

# 2、按照指定频率迈步+平地行走+扰动力和力矩
# python play.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.25 \
#     --env.step-frequency-max 0.25 \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/zbot_disturbed_walking/model_599.pt

# 2.5、跟踪频率 + 速度指令联合跟踪
# python play.py Mjlab-Zbot-6dof-Velocity-Frequency-Tracking-Walking \
#     --env.step-frequency-min 0.5 \
#     --env.step-frequency-max 0.5 \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.05,0.05)" \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/velocity_frequency_tracking_2026-10-22_iter499/model_499.pt

# 3、跟踪速度指令，但频率不跟踪，增加节省能量的奖励
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.0,1.0)" \
#     --env.scene.num-envs 8 \
#     --checkpoint-file bests/velocity_energy_tracking_2026-10-10_iter499/model_499.pt

# 4、继承 Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking，只增加外力扰动和物理随机化
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/velocity_energy_disturbed_2026-10-10_iter499/model_499.pt

# 5、全方位行走实验
# python play.py Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/omni_velocity_energy_2026-10-10_iter499/model_499.pt \
#     "$@"

# 6、实机部署实验-1 （扰动 + 域随机化）
# python play.py Mjlab-Zbot-6dof-Deploy-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.4)" \
#     --env.scene.num-envs 16 \
#     --checkpoint-file bests/deploy_walking_2026-10-11_iter1499/model_1499.pt

# 7、双足到蛇形的变形实验
# python play.py Mjlab-Zbot-6dof-Transform-Biped-To-Snake \
#     --env.scene.num-envs 16 \
#     --checkpoint-file bests/transform_biped_to_snake_2026-10-16_iter199/model_199.pt

# 10、model_2550预训练输入 + base随机扰动力/力矩，默认8192环境、2000轮。
# python play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher \
#   --env.scene.num-envs 16 \
#   --checkpoint-file bests/transform_snake_to_biped_teacher_2026-10-17_iter200/model_200.pt

# 12、蛇足双向变形：随机蛇形/双足初态，+1站立 / -1爬下，37/45维观测。
# python play.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/transform_snake_biped_bidirectional_2026-10-17_iter599/model_599.pt

# 15、蛇形运动：覆盖前进、后退、左右侧移、双向转动及组合运动
#   python play.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/snake_world_x_2026-10-19_iter1499/model_1499.pt \
  # --stochastic True

# 18、新蛇形全向回放：前后、侧移、转向及组合运动。
# python play.py Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/snake_site_omni_2026-10-20_iter599/model_599.pt

# 21、 三教师指导的新全能教师；默认选择独立训练目录下的最新检查点。
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/universal_three_teacher_gaussian_mirror_2026-10-22_iter499/model_499.pt

# 22、归档全能教师 → 历史观测学生；由用户启动训练。
# 22.a 学生继承全能教师 - 150代即可
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/universal_student_clean_2026-10-22_iter150/model_150.pt \
#   "$@"

# 22.b 学生PPO自学微调
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO \
#   --log-root logs/rsl_rl_universal_student_ppo_v2 \
#   --env.scene.num-envs 1 \
#   "$@"

# 22.c 学生PPO v4：默认回放 v4 日志目录下最新检查点
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO-V4 \
#   --log-root logs/rsl_rl_universal_student_ppo_v4 \
#   --env.scene.num-envs 1 \
#   "$@"

# 22.d 外部扰动学生：独立日志目录的最新模型；回放环境全部受扰，避免单环境被抽为静默。
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-Disturbed \
#   --log-root logs/rsl_rl_universal_student_disturbed \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/universal_student_disturbed_2026-10-22_iter499/model_499.pt \
# #   "$@"
