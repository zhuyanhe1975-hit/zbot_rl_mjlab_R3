"""Bounded, training-free parallel snake gait scan (angles in radians).

Run with the mjlab Python environment, for example:
timeout --signal=TERM --kill-after=10s 580s python scripts/open_loop_gait_speed.py
The only persistent diagnostic artifact is --output. No checkpoints are used.
"""

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parameter_grid():
    rows = [
        {
            "family": "stationary",
            "amplitude": 0.0,
            "frequency_hz": 0.0,
            "phase_step": 0.0,
            "group_phase": 0.0,
            "even_ratio": 1.0,
            "start_phase": 0.0,
        }
    ]
    for amplitude, frequency, delta, start in itertools.product(
        (0.3, 0.6, 1.0, 1.4),
        (0.5, 1.0, 1.5),
        (
            -3 * math.pi / 4,
            -math.pi / 2,
            -math.pi / 4,
            math.pi / 4,
            math.pi / 2,
            3 * math.pi / 4,
        ),
        (0.0, math.pi / 2),
    ):
        # Leave room for the startup ramp under the 2*pi rad/s slew limit.
        if amplitude * frequency > 0.85:
            continue
        common = {
            "amplitude": amplitude,
            "frequency_hz": frequency,
            "phase_step": delta,
            "start_phase": start,
        }
        rows.append(dict(family="serpenoid", group_phase=0.0, even_ratio=1.0, **common))
        for offset, ratio in itertools.product((-math.pi / 2, math.pi / 2), (0.5, 1.0)):
            rows.append(
                dict(
                    family="sidewinding", group_phase=offset, even_ratio=ratio, **common
                )
            )
    return rows


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize(rows):
    valid = [
        row
        for row in rows
        if row["valid"] and row["parameters"]["family"] != "stationary"
    ]
    best = {}
    for family in ("serpenoid", "sidewinding", "all"):
        selected = [
            r for r in valid if family == "all" or r["parameters"]["family"] == family
        ]
        best[family] = {
            axis: max(selected, key=lambda r: abs(r[axis]), default=None)
            for axis in ("vx", "vy")
        }
    straight = [
        r for r in valid if abs(r["yaw_rate"]) <= 0.1 and r["body_yaw_peak_deg"] <= 30.0
    ]
    posture_safe = [r for r in straight if r["within_original_posture_limits"]]
    return {
        "best_by_family": best,
        "best_posture_safe": {
            axis: max(posture_safe, key=lambda r: abs(r[axis]), default=None)
            for axis in ("vx", "vy")
        },
        "best_straight": {
            axis: max(straight, key=lambda r: abs(r[axis]), default=None)
            for axis in ("vx", "vy")
        },
        "top10": sorted(valid, key=lambda r: r["speed"], reverse=True)[:10],
        "ranking": "absolute component for best vx/vy; hypot(vx,vy) for top10; valid only",
        "valid_count": len(valid),
        "flipped_count": sum(r["flipped"] for r in rows),
        "abnormal_count": sum(r["abnormal"] for r in rows),
    }


def run(args):
    import torch
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

    import zbot_rl_mjlab  # noqa: F401
    from zbot_rl_mjlab.robot import JOINT_NAMES, RS00_RATED_TORQUE
    from zbot_rl_mjlab.tasks.snake import mdp
    from zbot_rl_mjlab.tasks.snake.actions import (
        DirectJointPositionAction,
        DirectJointPositionActionCfg,
    )
    from zbot_rl_mjlab.tasks.snake.env_cfg import (
        SNAKE_JOINT_LIMIT,
        SNAKE_MAX_TARGET_SPEED,
        TASK_ID,
        SnakeEnvCfg,
    )
    from zbot_rl_mjlab.tasks.transformation.mdp import body_masses

    started = time.monotonic()
    grid = parameter_grid()
    if args.max_cases:
        grid = grid[: args.max_cases]
    cfg = cast(SnakeEnvCfg, load_env_cfg(TASK_ID, play=True))
    cfg.scene.num_envs = min(args.num_envs, len(grid))
    cfg.auto_reset = False
    cfg.seed = args.seed
    # Retain model, ground, solver, contact sensors and canonical snake reset.
    # No reward/observation/termination computation or training-policy dependency.
    cfg.observations = {}
    cfg.rewards = {}
    cfg.metrics = {}
    cfg.terminations = {}
    cfg.commands["motion"].debug_vis = False
    cfg.actions["joint_pos"] = DirectJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        use_default_offset=False,
        scale=SNAKE_JOINT_LIMIT,
        target_limit=SNAKE_JOINT_LIMIT,
    )
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    for actuator in articulation.actuators:
        assert isinstance(actuator, BuiltinPositionActuatorCfg)
        actuator.effort_limit = RS00_RATED_TORQUE
    env = ManagerBasedRlEnv(cfg, device=args.device)
    rows = []
    result: dict[str, Any] = {
        "status": "running",
        "task": TASK_ID,
        "cases_planned": len(grid),
        "seed": args.seed,
        "method": "mass-weighted COM displacement over measurement_window_s projected on initial site -Z heading; no resets",
        "tracking_difference": "Guided tracking_velocity uses instantaneous foot_0_sole point velocity and instantaneous site -Z heading, not COM displacement",
        "gait_equations": {
            "serpenoid": "q[j]=ramp*A*sin(2*pi*f*t+j*delta+start)",
            "sidewinding": "q[2*k]=ramp*A*sin(2*pi*f*t+k*delta+start); q[2*k+1]=ramp*A*ratio*sin(2*pi*f*t+k*delta+start+group_phase)",
        },
        "limitations": [
            "Oblique hinges: odd/even groups are not pure horizontal/vertical hinges.",
            "Finite grid gives a demonstrated speed, not a global physical maximum.",
            "Flat isotropic friction 0.6; rated 5 Nm torque; no thermal or battery model.",
            "Single deterministic reset per start phase; no terrain/randomization robustness test.",
            "Safety sampled at 50 Hz control boundaries; substep contact history retained.",
            "Independent snake full-body friction differs from some universal task contact settings.",
        ],
        "constraints": {
            "target_limit_rad": SNAKE_JOINT_LIMIT,
            "target_slew_limit_rad_s": SNAKE_MAX_TARGET_SPEED,
            "actuator_effort_limit_nm": RS00_RATED_TORQUE,
            "actual_speed_valid_limit_rad_s": 10.47197551197,
        },
        "validity": "finite, no head/body inversion, no self-contact, no unsafe impact/workspace violation, actual joints within target envelope (+0.05 rad), actual speed <=100 rpm",
        "flip_definition": "head local +X below horizontal OR mass-weighted initial body-up directions below horizontal",
        "straight_definition": "abs(mean head heading yaw rate)<=0.1 rad/s and peak tail-to-head yaw deviation<=30 degrees",
        "original_posture_limits_deg": {
            "head_tilt": math.degrees(cfg.head_tilt_limit),
            "head_yaw": math.degrees(cfg.head_yaw_limit),
            "body_yaw": math.degrees(cfg.body_yaw_limit),
        },
        "results": rows,
    }

    def save(status):
        result.update(
            status=status,
            elapsed_wall_s=time.monotonic() - started,
            cases_completed=len(rows),
            **summarize(rows),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                clean_json(result), indent=2, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )

    try:
        robot = env.scene["robot"]
        assert tuple(robot.joint_names) == JOINT_NAMES, robot.joint_names
        action = env.action_manager.get_term("joint_pos")
        assert isinstance(action, DirectJointPositionAction)
        assert action.target_ids.tolist() == list(range(6))
        result["joint_names"] = list(robot.joint_names)
        result["physical_joint_limits_rad"] = (
            robot.data.joint_pos_limits[0].cpu().tolist()
        )
        result["num_parallel_envs"] = env.num_envs
        result["physics_dt"] = env.physics_dt
        result["control_dt"] = env.step_dt
        steps, warm = (
            round(args.seconds / env.step_dt),
            round(args.discard / env.step_dt),
        )
        if not 0 < warm < steps:
            raise ValueError(
                "discard and measurement windows must each contain a control step"
            )
        window = (steps - warm) * env.step_dt
        result["measurement_window_s"] = [warm * env.step_dt, steps * env.step_dt]
        masses = body_masses(env)
        weights = masses / masses.sum(-1, keepdim=True)
        site = mdp.motion_site_index(env)
        tail = robot.find_sites(cfg.tail_site)[0][0]
        j = torch.arange(6, device=args.device)[None, :]

        def com():
            return (robot.data.body_com_pos_w * weights[..., None]).sum(-2)

        def heading():
            direction = quat_apply(
                robot.data.site_quat_w[:, site],
                robot.data.joint_pos.new_tensor((0.0, 0.0, -1.0)).expand(
                    env.num_envs, -1
                ),
            )[:, :2]
            return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        with torch.inference_mode():
            for offset in range(0, len(grid), env.num_envs):
                batch = grid[offset : offset + env.num_envs]
                padded = batch + [grid[0]] * (env.num_envs - len(batch))
                env.reset()
                initial_heading = heading().clone()
                initial_left = torch.stack(
                    (-initial_heading[:, 1], initial_heading[:, 0]), -1
                )
                initial_yaw = torch.atan2(initial_heading[:, 1], initial_heading[:, 0])
                up = torch.zeros_like(robot.data.body_link_pos_w)
                up[..., 2] = 1
                local_up = quat_apply_inverse(robot.data.body_link_quat_w, up)
                parameters = {
                    key: torch.tensor([r[key] for r in padded], device=args.device)[
                        :, None
                    ]
                    for key in (
                        "amplitude",
                        "frequency_hz",
                        "phase_step",
                        "group_phase",
                        "even_ratio",
                        "start_phase",
                    )
                }
                side = torch.tensor(
                    [r["family"] == "sidewinding" for r in padded], device=args.device
                )[:, None]
                index = torch.where(side, j // 2, j)
                phase = index * parameters["phase_step"] + parameters["start_phase"]
                phase += side * (j % 2) * parameters["group_phase"]
                amplitude = parameters["amplitude"] * torch.where(
                    side & (j % 2 == 1), parameters["even_ratio"], 1.0
                )
                target = torch.zeros((env.num_envs, 6), device=args.device)
                flags = torch.zeros(
                    (env.num_envs, 5), dtype=torch.bool, device=args.device
                )
                peaks = torch.zeros((env.num_envs, 8), device=args.device)
                yaw_sum = torch.zeros(env.num_envs, device=args.device)
                angular_sum = torch.zeros_like(yaw_sum)
                tracking_sum = torch.zeros((env.num_envs, 2), device=args.device)
                previous_yaw = initial_yaw.clone()
                start_com = com().clone()
                current_com = start_com
                for step in range(1, steps + 1):
                    t = (step - 1) * env.step_dt
                    ramp = 0.5 - 0.5 * math.cos(math.pi * min(t, 1.0))
                    desired = (
                        ramp
                        * amplitude
                        * torch.sin(
                            2 * math.pi * parameters["frequency_hz"] * t + phase
                        )
                    )
                    change = (desired - target).clamp(
                        -SNAKE_MAX_TARGET_SPEED * env.step_dt,
                        SNAKE_MAX_TARGET_SPEED * env.step_dt,
                    )
                    target += change
                    peaks[:, 7] = torch.maximum(
                        peaks[:, 7], change.abs().amax(-1) / env.step_dt
                    )
                    env.step(target / SNAKE_JOINT_LIMIT)
                    data = robot.data
                    current_com = com()
                    current_heading = heading()
                    yaw = torch.atan2(current_heading[:, 1], current_heading[:, 0])
                    yaw_delta = torch.atan2(
                        (yaw - previous_yaw).sin(), (yaw - previous_yaw).cos()
                    )
                    previous_yaw = yaw
                    tilt = mdp.head_tilt(env)
                    body_up = (
                        quat_apply(data.body_link_quat_w, local_up)[..., 2] * weights
                    ).sum(-1)
                    flags[:, 0] |= (tilt > math.pi / 2) | (body_up < 0.0)
                    flags[:, 1] |= mdp.self_collision(env)
                    finite = (
                        torch.isfinite(current_com).all(-1)
                        & torch.isfinite(data.joint_pos).all(-1)
                        & torch.isfinite(data.joint_vel).all(-1)
                    )
                    flags[:, 2] |= ~finite | mdp.unsafe(env)
                    flags[:, 3] |= data.joint_vel.abs().amax(-1) > 10.47197551197
                    flags[:, 4] |= (
                        data.joint_pos.abs().amax(-1) > SNAKE_JOINT_LIMIT + 0.05
                    )
                    axis = data.site_pos_w[:, site, :2] - data.site_pos_w[:, tail, :2]
                    body_yaw = torch.atan2(axis[:, 1], axis[:, 0]) - initial_yaw
                    body_yaw = torch.atan2(body_yaw.sin(), body_yaw.cos()).abs()
                    head_yaw = yaw - initial_yaw
                    head_yaw = torch.atan2(head_yaw.sin(), head_yaw.cos()).abs()
                    values = torch.stack(
                        (
                            tilt,
                            body_yaw,
                            data.actuator_force.abs().amax(-1),
                            data.joint_vel.abs().amax(-1),
                            mdp.normalized_ground_peak(env),
                            data.joint_pos.abs().amax(-1),
                            head_yaw,
                        ),
                        -1,
                    )
                    peaks[:, :7] = torch.maximum(peaks[:, :7], values)
                    if step == warm:
                        start_com = current_com.clone()
                    if step > warm:
                        yaw_sum += yaw_delta
                        angular_sum += data.site_ang_vel_w[:, site, 2] * env.step_dt
                        linear = data.site_lin_vel_w[:, site, :2]
                        left = torch.stack(
                            (-current_heading[:, 1], current_heading[:, 0]), -1
                        )
                        tracking_sum += (
                            torch.stack(
                                (
                                    (linear * current_heading).sum(-1),
                                    (linear * left).sum(-1),
                                ),
                                -1,
                            )
                            * env.step_dt
                        )
                    if step % 100 == 0:
                        print(
                            f"batch {offset // env.num_envs + 1}: {step * env.step_dt:.1f}/{args.seconds}s simulated, wall={time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                displacement = current_com - start_com
                velocity = (
                    torch.stack(
                        (
                            (displacement[:, :2] * initial_heading).sum(-1),
                            (displacement[:, :2] * initial_left).sum(-1),
                        ),
                        -1,
                    )
                    / window
                )
                packed = (
                    torch.cat(
                        (
                            velocity,
                            (yaw_sum / window)[:, None],
                            (angular_sum / window)[:, None],
                            tracking_sum / window,
                            peaks,
                        ),
                        -1,
                    )
                    .cpu()
                    .tolist()
                )
                for n, (parameters_row, numbers, status) in enumerate(
                    zip(batch, packed, flags.cpu().tolist())
                ):
                    vx, vy, yaw_rate, angular_rate, tracking_x, tracking_y, *peak = (
                        numbers
                    )
                    rows.append(
                        {
                            "id": offset + n,
                            "parameters": parameters_row,
                            "vx": vx,
                            "vy": vy,
                            "speed": math.hypot(vx, vy),
                            "yaw_rate": yaw_rate,
                            "mean_site_angular_z": angular_rate,
                            "mean_guided_style_site_velocity": [tracking_x, tracking_y],
                            "head_tilt_peak_deg": math.degrees(peak[0]),
                            "body_yaw_peak_deg": math.degrees(peak[1]),
                            "torque_peak_nm": peak[2],
                            "joint_speed_peak_rad_s": peak[3],
                            "contact_peak_bodyweights": peak[4],
                            "joint_angle_peak_rad": peak[5],
                            "head_yaw_peak_deg": math.degrees(peak[6]),
                            "within_original_posture_limits": (
                                peak[0] < cfg.head_tilt_limit
                                and peak[1] <= cfg.body_yaw_limit
                                and peak[6] <= cfg.head_yaw_limit
                            ),
                            "target_slew_peak_rad_s": peak[7],
                            "flipped": status[0],
                            "self_collision": status[1],
                            "abnormal": any(status[1:]),
                            "unsafe": status[2],
                            "motor_speed_exceeded": status[3],
                            "joint_envelope_exceeded": status[4],
                            "valid": not any(status),
                        }
                    )
                save("running")
        save("complete")
    finally:
        env.close()
    print("rank id family A f delta group ratio start vx vy yaw flipped abnormal")
    for rank, row in enumerate(result["top10"], 1):
        p = row["parameters"]
        print(
            f"{rank:2} {row['id']:3} {p['family']:12} {p['amplitude']:.2f} {p['frequency_hz']:.2f} "
            f"{p['phase_step']:.3f} {p['group_phase']:.3f} {p['even_ratio']:.1f} {p['start_phase']:.3f} "
            f"{row['vx']:+.5f} {row['vy']:+.5f} {row['yaw_rate']:+.4f} {row['flipped']} {row['abnormal']}"
        )
    print(json.dumps(result["best_by_family"], ensure_ascii=False, indent=2))
    print(f"Results: {args.output}; wall seconds: {result['elapsed_wall_s']:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--discard", type=float, default=2.0)
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Only for a bounded smoke test"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/diagnostics/open_loop_gait_speed/results.json"),
    )
    args = parser.parse_args()
    if (
        args.num_envs <= 0
        or args.max_cases < 0
        or not all(math.isfinite(x) for x in (args.seconds, args.discard))
        or not (0 < args.discard < args.seconds)
    ):
        parser.error("require num-envs>0, max-cases>=0 and 0<discard<seconds")
    run(args)


if __name__ == "__main__":
    main()
