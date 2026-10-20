source /home/yhzhu/AI/mjlab/.venv/bin/activate

# 特权全能教师回放。
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided \
#   --log-root logs/rsl_rl_guided \
#   --env.commands.transform.mode source \
#   --env.scene.num-envs 16 \
#   --viewer native \
#   --follow-smoothing-time 0.2 \
#   "$@"

# 1、按照指定频率迈步+基本平地行走
# python play.py Mjlab-Zbot-6dof-Walking \
#     --env.step-frequency-min 0.3 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 8 \
#     --checkpoint-file bests/zbot_walking/model_599.pt

# 2、按照指定频率迈步+平地行走+扰动力和力矩
# python play.py Mjlab-Zbot-6dof-Disturbed-Walking \
#     --env.step-frequency-min 0.5 \
#     --env.step-frequency-max 1.0 \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/zbot_disturbed_walking/model_599.pt

# 3、跟踪速度指令，但频率不跟踪，增加节省能量的奖励
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/velocity_energy_tracking_2026-10-10_iter499/model_499.pt

# 4、继承 Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking，只增加外力扰动和物理随机化
# python play.py Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.1)" \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/velocity_energy_disturbed_2026-10-10_iter499/model_499.pt

# 5、全方位行走实验
# python play.py Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking \
#     --env.scene.num-envs 1 \
#     "$@"
    # --checkpoint-file bests/omni_velocity_energy_2026-10-10_iter499/model_499.pt

# 6、实机部署实验-1 （扰动 + 域随机化）
# python play.py Mjlab-Zbot-6dof-Deploy-Walking \
#     --env.commands.target-speed.ranges.lin-vel-x "(0.1,0.4)" \
#     --env.scene.num-envs 16 \
#     --checkpoint-file bests/deploy_walking_2026-10-11_iter1499/model_1499.pt

# 7、双足到蛇形的变形实验
# python play.py Mjlab-Zbot-6dof-Transform-Biped-To-Snake \
#     --env.scene.num-envs 16

# 8、蛇形到双足的变形实验
# python play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped \
#     --env.scene.num-envs 16 \
#     # --checkpoint-file logs/rsl_rl/zbot_transform_snake_to_biped/2026-10-17_10-36-10/model_400.pt

# python play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Arc \
#     --env.scene.num-envs 16 \

# 10、model_2550预训练输入 + base随机扰动力/力矩，默认8192环境、2000轮。
# python play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher \
#   --env.scene.num-envs 16 \

# 12、蛇足双向变形：随机蛇形/双足初态，+1站立 / -1爬下，37/45维观测。
# python play.py Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional \
#   --env.scene.num-envs 1 \
#   --checkpoint-file bests/transform_snake_biped_bidirectional_2026-10-17_iter599/model_599.pt

# 13、综合运动：多教师蒸馏预热，随后联合强化学习。
# python play.py Mjlab-Zbot-6dof-Universal-Locomotion \
#   --checkpoint-file logs/rsl_rl/zbot_universal_locomotion/2026-10-18_11-34-33/model_897.pt \
#   --env.scene.num-envs 1

# 14、综合运动：多教师蒸馏预热，随后联合强化学习。
# python play.py Mjlab-Zbot-6dof-Universal-Locomotion-Deploy \
#   --env.scene.num-envs 1

# 15、蛇形运动：覆盖前进、后退、左右侧移、双向转动及组合运动
  # python play.py Mjlab-Zbot-6dof-Snake-Omni-Teacher \
  # --env.scene.num-envs 1 \
  # --checkpoint-file bests/snake_world_x_2026-10-19_iter1499/model_1499.pt \
  # # --stochastic True

# 16、蛇足全能部署学生：腿走 + 蛇前爬 + 双向变形，四教师从头蒸馏。
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy \
#   --env.scene.num-envs 16 \

# 17、纯双足对照：只读取本次对照的检查点。
# python play.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Wide-Biped \
#   --log-root logs/ablations/biped_scalar_only \
#   --env.commands.transform.mode stand \
#   --env.scene.num-envs 16 \
#   --viewer native \
#   --follow-smoothing-time 0.2

# 18、新蛇形全向回放：前后、侧移、转向及组合运动。
python play.py Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher \
  --env.scene.num-envs 1
