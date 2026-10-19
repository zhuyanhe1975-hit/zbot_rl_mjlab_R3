"""策略对 RS00 关节的负荷有多大？

    python scripts/actuator_usage.py <TASK> <checkpoint.pt> [--speed 0.25]

使用 robot.py 中的执行器模型，在固定前进指令下运行 16 个环境、1000 个控制步
（丢弃前 250 步，保留 15 s），并逐关节报告：

* 扭矩：将 |torque| 的峰值、99 百分位和均方根与 14 N*m 峰值及
  5 N*m 额定（连续）扭矩比较；
* 速度：将 |joint velocity| 的 99 百分位和峰值与数据手册 T-N 曲线开始下降的
  100 rpm 拐点及 315 rpm 空载转速比较；
* 仿真器仅施加固定的 14 N*m 上限，未施加数据手册中随速度下降的 T-N 曲线，
  因而可能要求电机输出超出硬件能力的扭矩。“over curve” 表示沿运动方向的扭矩
  超过该速度下曲线允许值（robot.available_torque）的样本比例。
  该比例大于 0，意味着仿真步态使用了真实电机在该速度下无法提供的扭矩；
* 按手册的旋转过载表（robot.RS00_OVERLOAD）进行热负荷检查：
  计算任意 3、7、10、15 s 窗口内的最大扭矩均方根，与电机可持续该时长的负载比较。
  该表适用于安装在 90x85 mm 散热器上的冷态电机，仅能作为实际模块的粗略参考。

同时统计跌倒次数，以判断采用其他执行器限制训练的策略能否在 RS00 限制下继续行走。
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
from zbot_rl_mjlab import robot

NUM_ENVS, STEPS, WARMUP = 16, 1000, 250
WINDOWS_S = (3.0, 7.0, 10.0, 15.0)


def curve_limit(velocity):
    """按数组元素计算各关节转速对应的数据手册 T-N 扭矩上限。"""
    return np.vectorize(robot.available_torque)(velocity)


def hold_limit(seconds):
    """手册规定电机在旋转状态下可持续指定时长的负载（N*m）。"""
    times, loads = zip(*robot.RS00_OVERLOAD)
    if seconds >= times[-1]:
        return robot.RS00_RATED_TORQUE
    return float(np.interp(seconds, times, loads))


def window_rms(torque, steps):
    """逐关节计算任意 `steps` 个控制步窗口内的最大扭矩均方根。

    torque 的形状为 (time, envs, joints)；最大值在窗口和环境两个维度上取。
    """
    square = np.cumsum(np.square(torque), axis=0)
    square = np.concatenate([np.zeros_like(square[:1]), square])
    mean = (square[steps:] - square[:-steps]) / steps
    return np.sqrt(mean).max(axis=(0, 1))


def collect(task, checkpoint, speed, device="cuda:0"):
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
    data = env.scene["robot"].data
    torque, velocity = [], []
    falls = 0
    obs = wrapped.reset()[0]
    with torch.inference_mode():
        for step in range(STEPS):
            obs, _, dones, extras = wrapped.step(policy(obs))
            timed_out = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls += int((dones.bool() & ~timed_out).sum())
            if step >= WARMUP:
                torque.append(data.actuator_force.cpu().numpy())
                velocity.append(data.joint_vel.cpu().numpy())
    dt = env.step_dt
    env.close()
    return np.stack(torque), np.stack(velocity), falls, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()
    torque, velocity, falls, dt = collect(args.task, args.checkpoint, args.speed)

    peak, rated = robot.RS00_PEAK_TORQUE, robot.RS00_RATED_TORQUE
    corner, top = robot.RS00_CORNER_SPEED, robot.RS00_NO_LOAD_SPEED
    print(
        f"RESULT falls={falls}  (peak {peak} N*m, rated {rated} N*m, T-N curve falls "
        f"from {corner:.1f} rad/s, no-load {top:.1f} rad/s)"
    )
    # T-N 曲线限制沿运动方向输出的驱动扭矩，不限制制动扭矩。
    driving = torque * velocity > 0
    over_curve = driving & (np.abs(torque) > curve_limit(velocity) + 1e-6)
    flat = lambda a: a.reshape(-1, a.shape[-1])
    t_all, v_all, over = flat(torque), flat(velocity), flat(over_curve)
    limits = {w: hold_limit(w) for w in WINDOWS_S}
    heat = {w: window_rms(torque, round(w / dt)) for w in WINDOWS_S}
    for joint in range(torque.shape[-1]):
        t, v = np.abs(t_all[:, joint]), np.abs(v_all[:, joint])
        rms = float(np.sqrt(np.mean(t_all[:, joint] ** 2)))
        print(
            f"RESULT joint{joint + 1}: torque peak {t.max():5.2f} p99 {np.percentile(t, 99):5.2f}"
            f" rms {rms:4.2f}{' >rated' if rms > rated else ''} | speed p99 "
            f"{np.percentile(v, 99):5.1f} peak {v.max():5.1f} rad/s"
            f"{' >100rpm' if v.max() > corner else ''}{' >no-load' if v.max() > top else ''}"
            f" | over curve {100 * over[:, joint].mean():5.2f}% of samples"
        )
    for window in WINDOWS_S:
        worst = heat[window]
        flag = " OVER" if (worst > limits[window]).any() else ""
        print(
            f"RESULT thermal {window:4.0f} s window: max rms per joint "
            + " ".join(f"{x:5.2f}" for x in worst)
            + f" | allowed {limits[window]:5.2f}{flag}"
        )


if __name__ == "__main__":
    main()
