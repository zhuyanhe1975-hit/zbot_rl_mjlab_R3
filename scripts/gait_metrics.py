"""评估检查点的步态：足底平整度、交替性、周期稳定性和镜像对称性。

    python scripts/gait_metrics.py <TASK> <checkpoint.pt> [--speed 0.25]

以固定前进指令运行 16 个环境、1000 个控制步（20 s），并丢弃前 250 步。
参考数值见 docs/gait-metrics.md。
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.tasks.walking import mdp

NUM_ENVS, STEPS, WARMUP = 16, 1000, 250


def rollout(task, checkpoint, speed, device):
    """返回逐步的 (contact, pitch, roll, joint_pos) 数组和步长 dt。"""
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = NUM_ENVS
    cfg.commands["target_speed"].ranges.lin_vel_x = (speed, speed)
    env = ManagerBasedRlEnv(cfg, device=device)
    agent = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    robot = env.scene["robot"]
    sites = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)[0]
    base = mdp._base(env)[1]
    contact, pitch, roll, joints = [], [], [], []
    obs = wrapped.reset()[0]
    with torch.inference_mode():
        for step in range(STEPS):
            obs, *_ = wrapped.step(policy(obs))
            if step < WARMUP:
                continue
            quat = robot.data.site_quat_w[:, sites]
            normal = torch.zeros_like(quat[..., :3])
            normal[..., 2] = 1
            normal = mdp.quat_apply(quat.reshape(-1, 4), normal.reshape(-1, 3))
            normal = normal.reshape(NUM_ENVS, 2, 3)
            normal = normal * torch.sign(normal[..., 2:3])  # 足底法向量朝上
            body_quat = robot.data.body_link_quat_w[:, base]
            fwd_b, left_b, _ = mdp.locomotion_frame(body_quat)
            fwd = mdp.quat_apply(body_quat, fwd_b)[:, None]
            left = mdp.quat_apply(body_quat, left_b)[:, None]
            pitch.append(torch.rad2deg(torch.asin((normal * fwd).sum(-1).clamp(-1, 1))))
            roll.append(torch.rad2deg(torch.asin((normal * left).sum(-1).clamp(-1, 1))))
            contact.append(env.scene["feet_ground_contact"].data.found > 0)
            joints.append(robot.data.joint_pos.clone())
    dt = env.step_dt
    env.close()
    stack = lambda xs: torch.stack(xs).cpu().numpy()
    return stack(contact), stack(pitch), stack(roll), stack(joints), dt


def touchdown_stats(contact, dt):
    """计算步周期（合并双足）及足 0 落地后足 1 落地的相位。"""
    periods, offsets = [], []
    for env_i in range(contact.shape[1]):
        landings = [
            np.flatnonzero(contact[1:, env_i, f] & ~contact[:-1, env_i, f]) * dt
            for f in range(2)
        ]
        for times in landings:
            if len(times) > 2:
                periods.append(np.diff(times))
        if len(landings[0]) > 2 and len(landings[1]) > 2:
            period = np.median(np.diff(landings[0]))
            for t0 in landings[0][:-1]:
                later = landings[1][landings[1] > t0]
                if len(later):
                    offsets.append((later[0] - t0) / period)
    return (np.concatenate(periods) if periods else np.array([np.nan])), np.array(
        offsets
    )


def mirror_errors(joints, period_steps):
    """计算 q_i(t) + q_(7-i)(t + T/2) 的均方根与 std(q_i) 的比值；0 表示完全镜像对称。"""
    half = max(period_steps // 2, 1)
    errors = []
    for i in range(3):
        a, b = joints[:-half, :, i], -joints[half:, :, 5 - i]
        errors.append(np.sqrt(((a - b) ** 2).mean()) / (joints[:, :, i].std() + 1e-6))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    contact, pitch, roll, joints, dt = rollout(
        args.task, args.checkpoint, args.speed, args.device
    )
    for foot in range(2):
        loaded = contact[:, :, foot]
        for name, angle in (("pitch", pitch), ("roll", roll)):
            tilt = np.abs(angle[:, :, foot][loaded])
            print(
                f"RESULT foot{foot} {name:5s} mean={tilt.mean():.1f} "
                f"p90={np.percentile(tilt, 90):.1f} deg (loaded samples: {loaded.sum()})"
            )
    both = (contact[:, :, 0] & contact[:, :, 1]).mean()
    neither = (~contact[:, :, 0] & ~contact[:, :, 1]).mean()
    print(f"RESULT double-support={both:.2f} flight={neither:.2f}")
    periods, offsets = touchdown_stats(contact, dt)
    print(
        f"RESULT period={periods.mean():.2f}s cv={periods.std() / periods.mean():.2f} | "
        f"alternation phase={offsets.mean():.2f} std={offsets.std():.2f} (0.50 is ideal)"
    )
    errors = mirror_errors(joints, round(np.nanmedian(periods) / dt))
    e16, e25, e34 = errors
    print(
        f"RESULT mirror error joint1/6={e16:.2f} joint2/5={e25:.2f} "
        f"joint3/4={e34:.2f} (0 ideal, ~1.4 unrelated)"
    )


if __name__ == "__main__":
    main()
