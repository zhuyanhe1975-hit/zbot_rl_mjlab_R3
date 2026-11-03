"""Validate the shared runtime, CUDA computation, EGL and a tiny Zbot simulation."""

import importlib.metadata
import json
import sys

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg

import zbot_rl_mjlab  # noqa: F401


def main():
    versions = {name: importlib.metadata.version(name) for name in (
        "mjlab", "torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "tensordict",
    )}
    assert torch.cuda.is_available(), "CUDA is unavailable"
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    torch.cuda.synchronize()
    wp.init()
    assert wp.get_device("cuda:0").is_cuda
    tasks = [task for task in list_tasks() if task.startswith("Mjlab-Zbot")]
    assert len(tasks) == 45, tasks

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/>'
        '<geom type="sphere" size="0.1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
        assert frame.shape == (64, 64, 3) and np.isfinite(frame).all()

    cfg = load_env_cfg("Mjlab-Zbot-6dof-Walking")
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env)
        obs = wrapped.get_observations()
        for _ in range(8):
            obs, rewards, _, _ = wrapped.step(
                torch.zeros(4, wrapped.num_actions, device="cuda:0")
            )
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
        torch.cuda.synchronize()
    finally:
        env.close()
    print(json.dumps({
        "python": sys.version, "executable": sys.executable, "versions": versions,
        "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
        "tasks": len(tasks), "cuda_backward": "passed", "egl": "passed",
        "simulation": "4 environments, 8 steps passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
