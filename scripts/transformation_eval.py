"""评估任一变形方向的一个完整回合。

未指定 --checkpoint 时，使用零关节目标速度进行评估（保持初始姿态）。
使用已注册的 MJLab 任务，包括其目标与失败判定。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.tasks.transformation import mdp
from zbot_rl_mjlab.tasks.transformation.env_cfg import TASK_ID
from zbot_rl_mjlab.tasks.transformation.robust import (
    ROBUST_TASK_ID,
    RampedBaseImpulse,
    RobustSnakeToBipedEnvCfg,
    disturbance_scale,
)
from zbot_rl_mjlab.tasks.transformation.snake_to_biped import (
    ARC_TASK_ID,
    BETA_TASK_ID,
    SnakeToBipedEnvCfg,
    base_height,
    base_vertical_velocity,
    both_feet_ground_contact,
    sole_angles_degrees,
    sole_support_valid,
)
from zbot_rl_mjlab.tasks.transformation.snake_to_biped import TASK_ID as REVERSE_TASK_ID
from zbot_rl_mjlab.tasks.transformation.teacher_observations import TEACHER_TASK_ID


def evaluate(num_envs=4, device="cuda:0", checkpoint=None, seed=42, task_id=TASK_ID, curriculum_stage=2, support_observations=False, curriculum_joint_error_threshold=None):
    cfg = load_env_cfg(task_id, play=True)
    is_reverse = isinstance(cfg, SnakeToBipedEnvCfg)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    if is_reverse:
        assert isinstance(cfg, SnakeToBipedEnvCfg)
        cfg.curriculum_start_stage = curriculum_stage
        cfg.support_observations = support_observations
        if curriculum_joint_error_threshold is not None:
            cfg.curriculum_joint_error_threshold = curriculum_joint_error_threshold
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        agent = load_rl_cfg(task_id)
        joint_error_fn = cfg.metrics["joint_error"].func
        success_fn = cfg.metrics["success"].func
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        policy = None
        if checkpoint is not None:
            runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
            runner = runner_cls(wrapped, asdict(agent), device=device)
            runner.load(
                str(checkpoint),
                load_cfg={"actor": True},
                strict=True,
                map_location=device,
            )
            policy = runner.get_inference_policy(device=device)
        obs = wrapped.get_observations()
        base_id = env.scene["robot"].find_bodies("base")[0][0]
        disturbed_mask = None
        disturbance_config = None
        if isinstance(cfg, RobustSnakeToBipedEnvCfg):
            event = env.event_manager.get_term_cfg("base_disturbance")
            assert isinstance(event.func, RampedBaseImpulse)
            disturbed_mask = event.func.disturbed_mask.tolist()
            disturbance_config = {
                key: event.params[key]
                for key in ("force_range", "torque_range", "duration_s", "cooldown_s", "disturbed_fraction")
            }
            disturbance_config["scale"] = disturbance_scale(env)
        initial_state = {
            "joint_pos_rad_env_0": env.scene["robot"].data.joint_pos[0].tolist(),
            "max_abs_joint_pos_rad_all_envs": env.scene["robot"].data.joint_pos.abs().amax().item(),
            "root_quat_wxyz_env_0": env.scene["robot"].data.root_link_quat_w[0].tolist(),
            "root_position_relative_origin_env_0": (
                env.scene["robot"].data.root_link_pos_w[0] - env.scene.env_origins[0]
            ).tolist(),
        }
        roll_offsets = getattr(env, "_initial_snake_roll_offset", None)
        if roll_offsets is not None:
            initial_state["roll_offsets_deg_all_envs"] = torch.rad2deg(roll_offsets).tolist()
        standing_target = None
        sole_site_ids = None
        if is_reverse:
            assert isinstance(cfg, SnakeToBipedEnvCfg)
            standing_target = torch.as_tensor(cfg.standing_joint_pos, device=device)
            sole_site_ids, _ = env.scene["robot"].find_sites(
                ("foot_0_sole", "foot_1_sole"), preserve_order=True
            )
        finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        results = []
        trace = []
        sole_angle_sums = torch.zeros(num_envs, 2, device=device)
        sole_aligned_steps = torch.zeros(num_envs, device=device)
        episode_steps = torch.zeros(num_envs, device=device)
        contact_steps = torch.zeros(num_envs, device=device)
        support_steps = torch.zeros(num_envs, device=device)
        best_joint_error = torch.full((num_envs,), float("inf"), device=device)
        ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        peak_external_force = torch.zeros(num_envs, device=device)
        peak_external_torque = torch.zeros(num_envs, device=device)
        with torch.inference_mode():
            for step in range(env.max_episode_length + 1):
                actions = (
                    policy(obs)
                    if policy is not None
                    else torch.zeros(num_envs, wrapped.num_actions, device=device)
                )
                obs, reward, dones, _ = wrapped.step(actions)
                if not torch.isfinite(reward).all() or any(
                    not torch.isfinite(value).all()
                    for value in obs.to_dict().values()
                    if isinstance(value, torch.Tensor)
                ):
                    raise RuntimeError("non-finite observation or reward")
                com = mdp.com_state(env)
                joint_error = joint_error_fn(env)
                task = mdp.state(env)
                angles = None
                joint_errors = None
                sole_heights = None
                if is_reverse:
                    assert standing_target is not None and sole_site_ids is not None
                    angles = sole_angles_degrees(env)
                    joint_errors = (
                        env.scene["robot"].data.joint_pos
                        - standing_target
                    ).abs()
                    sole_heights = (
                        env.scene["robot"].data.site_pos_w[:, sole_site_ids, 2]
                        - env.scene.env_origins[:, 2, None]
                    )
                    active = ~finished
                    if disturbed_mask is not None:
                        wrench = env.scene["robot"].data.body_external_wrench[:, base_id]
                        peak_external_force[active] = torch.maximum(peak_external_force[active], wrench[active, :3].abs().amax(-1))
                        peak_external_torque[active] = torch.maximum(peak_external_torque[active], wrench[active, 3:].abs().amax(-1))
                    sole_angle_sums[active] += angles[active]
                    sole_aligned_steps[active] += (angles[active].amax(-1) < 10.0).float()
                    episode_steps[active] += 1
                    contact_steps[active] += both_feet_ground_contact(env)[active].float()
                    support_steps[active] += sole_support_valid(env)[active].float()
                    best_joint_error[active] = torch.minimum(best_joint_error[active], joint_error[active])
                    ever_success[active] |= success_fn(env)[active]
                # 为首个环境记录精简的完整时间分辨率轨迹。
                if not finished[0]:
                    trace.append(
                        {
                            "wall_time_s": (step + 1) * env.step_dt,
                            "com_height_m": com[0, 0].item(),
                            "com_vertical_velocity_m_s": com[0, 1].item(),
                            "joint_error_rad": joint_error[0].item(),
                            "hold_time_s": task.hold_time[0].item(),
                            "joint_pos_rad": env.scene["robot"]
                            .data.joint_pos[0]
                            .tolist(),
                        }
                    )
                    if angles is not None:
                        assert joint_errors is not None and sole_heights is not None
                        trace[-1]["joint_errors_rad"] = joint_errors[0].tolist()
                        trace[-1]["max_joint_error_rad"] = joint_errors[0].amax().item()
                        trace[-1]["sole_heights_m"] = sole_heights[0].tolist()
                        trace[-1]["joint_vel_rad_s"] = env.scene["robot"].data.joint_vel[0].tolist()
                        trace[-1]["joint_speed_rms_rad_s"] = env.scene["robot"].data.joint_vel[0].square().mean().sqrt().item()
                        trace[-1]["base_speed_m_s"] = env.scene["robot"].data.body_link_lin_vel_w[0, base_id].norm().item()
                        forces = env.scene["feet_ground_contact"].data.force
                        trace[-1]["feet_ground_force_w_n"] = None if forces is None else forces[0].tolist()
                        trace[-1]["sole_angles_deg"] = angles[0].tolist()
                        trace[-1]["base_height_m"] = base_height(env)[0].item()
                        trace[-1]["base_vertical_velocity_m_s"] = base_vertical_velocity(env)[0].item()
                        trace[-1]["commanded_target_rad"] = mdp.commanded_target(env)[0].tolist()
                        trace[-1]["policy_action"] = actions[0].tolist()
                        trace[-1]["root_quat_w"] = env.scene["robot"].data.root_link_quat_w[0].tolist()
                        trace[-1]["feet_ground_contacts"] = env.scene["feet_ground_contact"].data.found[0].tolist()
                        trace[-1]["sole_support_valid"] = bool(sole_support_valid(env)[0])
                        if disturbed_mask is not None:
                            trace[-1]["base_external_force_torque_w"] = env.scene["robot"].data.body_external_wrench[0, base_id].tolist()
                completed = dones.bool() & ~finished
                for i in completed.nonzero(as_tuple=False).flatten().tolist():
                    reasons = [
                        name
                        for name in cfg.terminations
                        if env.termination_manager.get_term(name)[i].item()
                    ]
                    results.append(
                        {
                            "environment": i,
                            "success": bool(success_fn(env)[i])
                            and not bool(env.termination_manager.terminated[i]),
                            "reasons": reasons,
                            "wall_time_s": (step + 1) * env.step_dt,
                            "hold_time_s": task.hold_time[i].item(),
                            "com_height_m": com[i, 0].item(),
                            "joint_error_rad": joint_error[i].item(),
                        }
                    )
                    if angles is not None:
                        assert joint_errors is not None and sole_heights is not None
                        results[-1].update(
                            joint_errors_rad=joint_errors[i].tolist(),
                            max_joint_error_rad=joint_errors[i].amax().item(),
                            sole_heights_m=sole_heights[i].tolist(),
                            joint_speed_rms_rad_s=env.scene["robot"].data.joint_vel[i].square().mean().sqrt().item(),
                            base_speed_m_s=env.scene["robot"].data.body_link_lin_vel_w[i, base_id].norm().item(),
                            feet_ground_force_w_n=None if env.scene["feet_ground_contact"].data.force is None else env.scene["feet_ground_contact"].data.force[i].tolist(),
                            base_height_m=base_height(env)[i].item(),
                            base_vertical_velocity_m_s=base_vertical_velocity(env)[i].item(),
                            joint_pos_rad=env.scene["robot"].data.joint_pos[i].tolist(),
                            commanded_target_rad=mdp.commanded_target(env)[i].tolist(),
                            best_joint_error_rad=best_joint_error[i].item(),
                            ever_success=bool(ever_success[i]),
                            both_feet_contact_fraction=(contact_steps[i] / episode_steps[i]).item(),
                            sole_support_valid_fraction=(support_steps[i] / episode_steps[i]).item(),
                            sole_angles_deg=angles[i].tolist(),
                            mean_sole_angles_deg=(
                                sole_angle_sums[i] / episode_steps[i]
                            ).tolist(),
                            both_soles_within_10deg_fraction=(
                                sole_aligned_steps[i] / episode_steps[i]
                            ).item(),
                        )
                finished |= completed
                if finished.all():
                    break
                # MJLab 要求先显式重置已结束的环境，才能继续推进仿真。
                # 每个环境仅将第一个回合计入结果。
                reset_ids = dones.nonzero(as_tuple=False).flatten()
                if reset_ids.numel():
                    env.reset(env_ids=reset_ids)
                    obs = wrapped.get_observations()
        if not finished.all():
            raise RuntimeError("evaluation ended with unfinished environments")
        results.sort(key=lambda row: row["environment"])
        disturbance_summary = {}
        if disturbed_mask is not None:
            for row in results:
                row["disturbance_enabled"] = bool(disturbed_mask[row["environment"]])
                row["peak_external_force_component_N"] = peak_external_force[row["environment"]].item()
                row["peak_external_torque_component_Nm"] = peak_external_torque[row["environment"]].item()
            disturbance_summary["disturbance_config"] = disturbance_config
            for group, enabled in (("disturbed", True), ("quiet", False)):
                selected = [row for row in results if row["disturbance_enabled"] == enabled]
                disturbance_summary[f"{group}_episode_count"] = len(selected)
                disturbance_summary[f"{group}_success_rate"] = sum(row["success"] for row in selected) / len(selected) if selected else None
        return {
            **disturbance_summary,
            "task": task_id,
            "seed": seed,
            "initial_state": initial_state,
            "curriculum_stage": curriculum_stage if is_reverse else None,
            "support_observations": support_observations if is_reverse else False,
            "curriculum_joint_error_threshold": cfg.curriculum_joint_error_threshold if isinstance(cfg, SnakeToBipedEnvCfg) else None,
            "controller": str(checkpoint)
            if checkpoint
            else "zero_joint_target_velocity",
            "success_rate": sum(row["success"] for row in results) / num_envs,
            "episodes": results,
            "trace_env_0": trace,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(TASK_ID, REVERSE_TASK_ID, BETA_TASK_ID, ARC_TASK_ID, ROBUST_TASK_ID, TEACHER_TASK_ID), default=TASK_ID)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-trace", action="store_true", help="同时保存首个环境的逐步轨迹；默认仅保存评估摘要")
    parser.add_argument("--curriculum-stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--support-observations", action="store_true", help="使用反向任务的 43 维观测（需要匹配的模型）")
    parser.add_argument("--curriculum-joint-error-threshold", type=float, default=None, help="反向任务从阶段 1 晋级时允许的最大关节误差（弧度）")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/transformation/evaluation.json")
    )
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    result = evaluate(args.num_envs, args.device, args.checkpoint, args.seed, args.task, args.curriculum_stage, args.support_observations, args.curriculum_joint_error_threshold)
    if not args.save_trace:
        result.pop("trace_env_0", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "trace_env_0"},
            indent=2,
        )
    )
    print(f"评估结果已写入 {args.output}")


if __name__ == "__main__":
    main()
