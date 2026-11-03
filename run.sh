#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/activate.sh" || exit 1

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


# 23、RZ270独立蛇形：趴地重置，蛇头朝世界+X，关节零位不变。
# ./scripts/python.sh play.py Mjlab-Zbot-6dof-Snake-RZ270-Teacher \
#     --env.scene.num-envs 1 \
#     --checkpoint-file bests/snake_rz270_2026-10-26_iter599/model_599.pt

# 24.a、初始Basic：13:11配方，59维、速度积分、原物理；使用独立日志根目录重新开始。
# python play.py Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Basic \
#     --log-root logs/rsl_rl_reconfiguration_initial \
#     --env.scene.num-envs 1

# 24.b、PBT重构回放：默认优先加载PBT独立日志目录中的best.pt。
# 切换实验时只启用一组命令；指定模型可加 --checkpoint-file /path/to/model.pt。
# python play.py Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-PBT \
#     --env.scene.num-envs 1 \
#     --env.commands.dock.target-yaw-deg 180

# 24.c、规划教师模仿学生：回放只执行学生，默认优先选best.pt。
# python play.py Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided \
#     --log-root logs/rsl_rl_reconfiguration_plan_imitation \
#     --env.scene.num-envs 1

# 24.d、历史断开PPO回放（暂停学习，当前使用独立状态机）。
# python play.py Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque \
#     --log-root logs/rsl_rl_disconnect_torque \
#     --env.scene.num-envs 1 \
#     "$@"

# 25、独立重构状态机：加载→断开→拉开5cm→端面相对旋转90°→接近贴合→重建weld。
# 默认MJLab原生窗口；bash run.sh --headless 可无窗口执行并保存JSON。
# python scripts/reconfigure_state_machine.py --viewer "$@"


# 26、独立运动学重构：初始闭环→首尾辅助边重构→五处内部180°到270°→最后开首尾并展开。
# 不使用力矩/动力学；忽略关节限位。始终一条连通链，IK失败停止，不移动独立模块凑终态。
# python scripts/reconfigure_to_rz270_kinematic.py --viewer "$@"


# 27、独立连通闭环运动学：全180→[270,270,180,90,90]，处理第1、2、4、5连接。
# 首尾辅助连接及第3连接保持180；忽略限位/力矩/动力学，每次只断开一处。
# python scripts/reconfigure_mixed_closed_kinematic.py --viewer "$@"

# 28、闭环混合角动力学状态机：真实载荷断开、原生PD运动、碰撞穿透保护。
# 尚未跑通全程；运行时实时显示阶段和保护触发。
# # # # ./scripts/python.sh scripts/reconfigure_mixed_closed_dynamic.py --viewer "$@"

# 29、回放混合连接角蛇形教师；先执行 train.sh 生成独立检查点。
# ./scripts/python.sh play.py Mjlab-Zbot-6dof-Snake-Mixed-270-270-180-90-90-Teacher \
#     --env.scene.num-envs 1 \
#     --checkpoint-file logs/rsl_rl/zbot_snake_mixed_270_270_180_90_90_teacher/2026-10-28_11-05-14/model_599.pt \
#     "$@"

# 30、混合连接角双足行走回放；先执行 train.sh 生成检查点。
# ./scripts/python.sh play.py Mjlab-Zbot-6dof-Walking-Mixed-270-270-180-90-90 \
#     --log-root logs/rsl_rl_walking_mixed_mirror_025 \
#     --env.scene.num-envs 1 \
#     "$@"

# 31、回到混合角重构动力学基线；当前为确定性状态机，尚未接入全流程PPO。
# # ./scripts/python.sh scripts/reconfigure_mixed_closed_dynamic.py --viewer "$@"

# 32、回放运动学引导的混合角重构PPO；先运行train.sh生成该任务的检查点。
# ./scripts/python.sh play.py Mjlab-Zbot-6dof-Reconfigure-Mixed-270-270-180-90-90-PPO \
#     --log-root logs/rsl_rl_reconfigure_mixed_guided_batched_v4 \
#     --env.scene.num-envs 1 \
#     "$@"

# 33、六模块重构v3：雅可比加载断开；七模块路径策略 + 固定原生PD动力学执行。
# 1ms物理步长；故障/完成自动暂停。bash run.sh --headless 可无窗口执行。
# ./scripts/python.sh scripts/reconfigure_mixed_closed_reference.py --viewer "$@"

# 34、CPU完整重构：1ms、100N，一次性优化与预演，断开后稳定并重连。
# ./scripts/python.sh scripts/reconfigure_mixed_full_cpu.py --viewer "$@"
# ./scripts/python.sh scripts/view_headtail_search.py "$@"
# ./scripts/python.sh scripts/view_headtail_bidirectional.py
# ./scripts/python.sh scripts/view_headtail_mjbatch_refined.py
# ./scripts/python.sh scripts/view_headtail_mjbatch_2ms.py

# 35、CPU 全流程教师轨迹引导的 PPO 学生；默认回放当前训练的最新检查点。
# 完成或失败后保持窗口；--headless 可无窗口运行，其他参数传给评估脚本。
# ./scripts/python.sh scripts/eval_full_cpu_student.py \
#     logs/full_cpu_reference_student/latest.pt --viewer "$@"

# 36、GPU 全流程重构学生：使用当前训练最新检查点与状态机教师，确定性策略回放。
# 成功或失败后保持窗口；bash run.sh --headless 可无窗口运行。
# ./scripts/python.sh scripts/eval_full_gpu_student.py \
#     --logs/full_gpu_reference_student/latest.pt \
#     --viewer "$@"

# 37、静态闭环内力优化评估：各连接面独立加载，达到断开条件即停止，不执行后续分离。
# ./scripts/python.sh scripts/eval_static_loading.py "$@"

# 38、GPU 精修正向参考可视化：单环境、1ms 物理步长；Enter 重置，空格暂停。
# 当前为零策略修正参考回放，尚不是训练好的 RL 策略。
# ./scripts/python.sh scripts/view_headtail_gpu_refined.py "$@"

# 39、实时关节空间雅可比规划v11：单 GPU 环境、2ms；优先验证best。
# Enter 重置、空格暂停；传 --checkpoint-file 可指定检查点。
# python play.py Mjlab-Zbot-6dof-Reconfigure-Initial-Closure \
#     --env.scene.num-envs 1 \
#     "$@"
