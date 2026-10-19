"""独立蛇形教师的无窗口验收；必须明确指定检查点，不自动选择最新模型。"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TASK_ID = "Mjlab-Zbot-6dof-Snake-Omni-Teacher"


@dataclass(frozen=True)
class Criteria:
    linear_error: float = .02
    yaw_error: float = .12
    head_tilt_mean_deg: float = 15.
    head_tilt_peak_deg: float = 30.
    contact_peak_bodyweights: float = 3.
    head_acceleration_cost_mean: float = 1.


DEFAULT_CRITERIA = Criteria()


def scenarios(speed: float, yaw_speed: float):
    """正负三个自由度、停车，以及平移和转弯的组合。"""
    return {
        "forward": (speed, 0., 0.),
        "backward": (-speed, 0., 0.),
        "left": (0., speed, 0.),
        "right": (0., -speed, 0.),
        "turn_left": (0., 0., yaw_speed),
        "turn_right": (0., 0., -yaw_speed),
        "stop": (0., 0., 0.),
        "forward_turn": (speed, 0., yaw_speed),
    }


def summarize(velocity, tilt, contact, acceleration, failed, collided, target,
              criteria=DEFAULT_CRITERIA, *, trial_tilt_peak=None, trial_contact_peak=None):
    """输入为计量窗的时间×环境数据，失败锁存则覆盖完整场景。

    采用逐帧绝对速度误差，避免正负摆动平均为零而误判停车成功。
    非有限数据直接判失败，并使用 JSON 的 null 表示无法统计的数值。
    """
    if velocity.ndim != 3 or velocity.shape[-1] != 3 or not velocity.shape[0]:
        raise ValueError("速度必须为非空的时间×环境×3张量")
    expected = velocity.shape[:2]
    if any(value.shape != expected for value in (tilt, contact, acceleration)):
        raise ValueError("指标必须与速度的时间、环境维数相同")
    finite = torch.isfinite(velocity).all(dim=(0, 2))
    for value in (tilt, contact, acceleration):
        finite &= torch.isfinite(value).all(dim=0)
    failed = failed.bool() | ~finite
    requested = velocity.new_tensor(target)
    error = (velocity - requested).abs().mean(dim=0)
    tilt_deg = torch.rad2deg(tilt)
    mean_tilt = tilt_deg.mean(dim=0)
    peak_tilt = tilt_deg.amax(dim=0)
    peak_contact = contact.amax(dim=0)
    # 无量纲抖动代价：各轴 (线加速度/15)^2 与 (角加速度/30)^2 之和。
    mean_acceleration_cost = acceleration.mean(dim=0)
    if trial_tilt_peak is not None:
        peak_tilt = torch.maximum(peak_tilt, torch.rad2deg(trial_tilt_peak))
    if trial_contact_peak is not None:
        peak_contact = torch.maximum(peak_contact, trial_contact_peak)
    tolerance = velocity.new_tensor((criteria.linear_error, criteria.linear_error,
                                     criteria.yaw_error))
    motion_passed = (error <= tolerance).all(-1) & ~failed & ~collided
    camera_stable = ((mean_tilt <= criteria.head_tilt_mean_deg)
                     & (peak_tilt <= criteria.head_tilt_peak_deg)
                     & (mean_acceleration_cost <= criteria.head_acceleration_cost_mean)
                     & finite)
    contact_safe = (peak_contact <= criteria.contact_peak_bodyweights) & finite & ~collided
    passed = motion_passed & camera_stable & contact_safe

    def clean(value):
        result = value.detach().cpu().tolist()
        if isinstance(result, list):
            return [number if math.isfinite(number) else None for number in result]
        return result if math.isfinite(result) else None

    return {
        "target_velocity_m_s_m_s_rad_s": list(target),
        "actual_velocity_m_s_m_s_rad_s": clean(velocity.mean(dim=(0, 1))),
        "velocity_mae_m_s_m_s_rad_s": clean(error.mean(dim=0)),
        "head_tilt_mean_deg": clean(mean_tilt.mean()),
        "head_tilt_peak_deg": clean(peak_tilt.amax()),
        "contact_peak_bodyweights": clean(peak_contact.amax()),
        "head_acceleration_cost_mean": clean(acceleration.mean()),
        "self_collision_rate": clean(collided.float().mean()),
        "failure_rate": clean(failed.float().mean()),
        "success_rate": clean(passed.float().mean()),
        "motion_success_rate": clean(motion_passed.float().mean()),
        "camera_stable_rate": clean(camera_stable.float().mean()),
        "contact_safe_rate": clean(contact_safe.float().mean()),
        "per_env_success": passed.cpu().tolist(),
    }


def evaluate(args):
    # 延迟导入仿真组件，--help 和纯统计单测不初始化 GPU 或 MuJoCo。
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from tensordict import TensorDict

    import zbot_rl_mjlab  # noqa: F401
    from zbot_rl_mjlab.experiments.snake import SnakeTeacherRunner
    from zbot_rl_mjlab.tasks.snake import mdp
    from zbot_rl_mjlab.tasks.snake.env_cfg import SnakeEnvCfg

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"检查点不存在：{checkpoint}")
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.auto_reset = False
    # 结束标记专用于安全失败，避免较长评估被训练回合时限截断。
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 3.2)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        agent = load_rl_cfg(TASK_ID)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        train_cfg = asdict(agent)
        if issubclass(runner_cls, SnakeTeacherRunner):
            SnakeTeacherRunner.prepare_playback_cfg(train_cfg, str(checkpoint))
        runner = runner_cls(wrapped, train_cfg, device=args.device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        if getattr(env.cfg, "observation_reference", "command") == "world_x":
            raise ValueError("当前脚本验收旧速度指令任务，不能用八方向指令场景评价世界X推进模型；请查看世界X位移、速度与复位指标。")
        policy = runner.get_inference_policy(device=args.device)
        command = mdp.command_term(env)
        if args.seconds < 2 * env.step_dt:
            raise ValueError("评估目标段至少需要两个控制步")
        limits = command.cfg.velocity_limits
        if args.speed > min(limits[:2]) or args.yaw_speed > limits[2]:
            raise ValueError("评估速度超过任务上限，指令会被截断；请减小速度")
        criteria = Criteria(args.linear_error, args.yaw_error,
                            args.head_tilt_mean_deg, args.head_tilt_peak_deg,
                            args.contact_peak_bodyweights,
                            args.head_acceleration_cost_mean)
        results = {}
        with torch.inference_mode():
            for name, target in scenarios(args.speed, args.yaw_speed).items():
                obs, _ = wrapped.reset()
                command.set_velocity(args.speed, 0., 0.)
                failed = torch.zeros(args.num_envs, device=args.device, dtype=torch.bool)
                collided = torch.zeros_like(failed)
                trial_tilt_peak = torch.zeros(args.num_envs, device=args.device)
                trial_contact_peak = torch.zeros_like(trial_tilt_peak)
                velocity, tilt, contact, acceleration = [], [], [], []
                total_steps = math.ceil((2.2 + args.seconds) / env.step_dt)
                previous_phase = None
                for step in range(total_steps):
                    time = step * env.step_dt
                    phase = 0 if time < 2. else 1 if time < 2.2 else 2
                    current_target = ((args.speed, 0., 0.) if phase == 0
                                      else (0., 0., 0.) if phase == 1 else target)
                    if phase != previous_phase:
                        command.set_velocity(*current_target)
                        previous_phase = phase
                    obs, reward, dones, _ = wrapped.step(policy(obs))
                    actual = mdp.planar_velocity(env)
                    head_tilt = mdp.head_tilt(env)
                    ground_peak = mdp.normalized_ground_peak(env)
                    head_acceleration = mdp.head_acceleration_cost(env)
                    trial_tilt_peak = torch.maximum(trial_tilt_peak, head_tilt)
                    trial_contact_peak = torch.maximum(trial_contact_peak, ground_peak)
                    # 必须先读终态，再显式重置；锁存标记贯穿准备段与计量段。
                    collided |= mdp.self_collision(env)
                    failed |= (dones.bool() | mdp.unsafe(env)
                               | ~torch.isfinite(reward)
                               | ~torch.isfinite(actual).all(-1)
                               | ~torch.isfinite(head_tilt)
                               | ~torch.isfinite(ground_peak)
                               | ~torch.isfinite(head_acceleration))
                    if time >= 2.2 + args.seconds / 2:
                        velocity.append(actual.clone())
                        tilt.append(head_tilt.clone())
                        contact.append(ground_peak.clone())
                        acceleration.append(head_acceleration.clone())
                    reset_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(reset_ids):
                        reset_obs, _ = env.reset(env_ids=reset_ids)
                        obs = TensorDict(reset_obs, batch_size=[args.num_envs])
                        command.set_velocity(*current_target, env_ids=reset_ids)
                results[name] = summarize(
                    torch.stack(velocity), torch.stack(tilt), torch.stack(contact),
                    torch.stack(acceleration), failed, collided, target, criteria,
                    trial_tilt_peak=trial_tilt_peak, trial_contact_peak=trial_contact_peak)
                print(f"{name}: 成功率 {results[name]['success_rate']:.1%}")
        return {
            "task": TASK_ID, "checkpoint": str(checkpoint), "seed": args.seed,
            "motion_frame": cast(SnakeEnvCfg, env.cfg).motion_frame,
            "head_gyro_frame": cast(SnakeEnvCfg, env.cfg).head_gyro_frame,
            "num_envs": args.num_envs, "seconds": args.seconds,
            "measurement_window": "先前进2秒、停车0.2秒，再切换目标；目标段后半段计量",
            "criteria": asdict(criteria),
            "success_definition": "逐帧速度误差、头倾斜、无量纲加速度抖动代价均值和接触峰值均达标，且全程无失败、无自碰撞；停车同样检查三轴漂移",
            "head_acceleration_cost_definition": "各轴(线加速度/15)^2与(角加速度/30)^2之和，再对计量窗取时间均值；无量纲，非SI加速度",
            "failure_latching": True, "scenarios": results,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="必须显式指定模型文件")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=.05)
    parser.add_argument("--yaw-speed", type=float, default=.3)
    parser.add_argument("--linear-error", type=float, default=.02)
    parser.add_argument("--yaw-error", type=float, default=.12)
    parser.add_argument("--head-tilt-mean-deg", type=float, default=15.)
    parser.add_argument("--head-tilt-peak-deg", type=float, default=30.)
    parser.add_argument("--contact-peak-bodyweights", type=float, default=3.)
    parser.add_argument("--head-acceleration-cost-mean", type=float, default=1.,
                        help="无量纲抖动代价均值上限：各轴(线加速度/15)^2与(角加速度/30)^2之和")
    parser.add_argument("--output", type=Path, default=Path("outputs/snake-evaluation/result.json"))
    args = parser.parse_args()
    positive = (args.seconds, args.speed, args.yaw_speed, args.linear_error,
                args.yaw_error, args.head_tilt_mean_deg, args.head_tilt_peak_deg,
                args.contact_peak_bodyweights, args.head_acceleration_cost_mean)
    if args.num_envs <= 0 or any(not math.isfinite(v) or v <= 0 for v in positive):
        parser.error("环境数量及时间、速度、验收阈值必须为正有限值")
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")
    print(f"评估结果：{args.output}")


if __name__ == "__main__":
    main()
