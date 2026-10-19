"""足部高度扫描射线从哪里发出，又击中了什么？

    python scripts/height_scan_check.py <TASK> <checkpoint.pt>

逐足检查：射线原点（刚体原点）相对足底站点的高度、击中平地以外物体的射线数量，
以及射线原点位于几何体内部的频率。
"""

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

task, checkpoint = sys.argv[1], sys.argv[2]
cfg = load_env_cfg(task, play=True)
cfg.scene.num_envs = 8
env = ManagerBasedRlEnv(cfg, device="cuda:0")
agent = load_rl_cfg(task)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")
robot = env.scene["robot"]
sites = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)[0]
sensor = env.scene["foot_height_scan"]
num_rays = sensor.num_rays_per_frame

origin_above_sole, hit_z, backface, miss = [], [], [], []
obs = wrapped.reset()[0]
with torch.inference_mode():
    for step in range(600):
        obs, *_ = wrapped.step(policy(obs))
        if step < 200:
            continue
        data = sensor.data
        frame_z = data.frame_pos_w[:, :, 2]  # [B, F]
        sole_z = robot.data.site_pos_w[:, sites, 2]  # [B, 2]
        origin_above_sole.append((frame_z - sole_z).cpu().numpy())
        batch = frame_z.shape[0]
        hz = data.hit_pos_w[:, :, 2].view(batch, 2, num_rays)
        dist = data.distances.view(batch, 2, num_rays)
        normal_z = data.normals_w[:, :, 2].view(batch, 2, num_rays)
        hit_z.append(hz.cpu().numpy())
        backface.append(((dist >= 0) & (normal_z < 0)).cpu().numpy())
        miss.append((dist < 0).cpu().numpy())
oas = np.concatenate(origin_above_sole)  # [S*B, 2]
hz = np.concatenate(hit_z)  # [S*B, 2, N]
bf = np.concatenate(backface)
ms = np.concatenate(miss)
for foot in range(2):
    on_robot = (hz[:, foot] > 0.003) & ~ms[:, foot]  # 击中位置高于地平面
    print(
        f"RESULT foot{foot}: ray origin {oas[:, foot].mean() * 1000:+.1f} mm "
        f"(std {oas[:, foot].std() * 1000:.1f}) above the sole site | "
        f"rays hitting above ground: {on_robot.mean() * 100:.1f}% "
        f"(mean hit z {hz[:, foot][on_robot].mean() * 1000 if on_robot.any() else 0:.0f} mm) | "
        f"backface (origin inside geometry): {bf[:, foot].mean() * 100:.1f}% | "
        f"misses: {ms[:, foot].mean() * 100:.1f}%"
    )
env.close()
