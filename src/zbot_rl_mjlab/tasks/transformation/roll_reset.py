"""重置时绕蛇形纵轴（根部局部 Z 轴）随机滚转。"""

import math
from functools import lru_cache

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

from ...robot import get_spec


@lru_cache(maxsize=4)
def collision_radial_envelope(joints):
    """原生碰撞顶点的径向凸包及纵向范围。

    沿纵向拉伸径向凸包可保守包围所有顶点。在规定的水平蛇形轴姿态下，
    除浮点误差外，其相对平面的净空计算精确，无需额外几何依赖。
    """
    model = get_spec().compile()
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[3:7] = (1, 0, 0, 0)
    data.qpos[7:] = joints
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    points = []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        mesh = model.geom_dataid[geom]
        if mesh < 0:
            raise ValueError("Snake roll reset requires native mesh collision geometry")
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        points.extend(vertices.tolist())
    radial = sorted({(p[0], p[1]) for p in points})

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in radial:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(radial):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1], min(p[2] for p in points), max(p[2] for p in points)


def rolled_root_states(default, origins, angles, radial, z_min, z_max, clearance):
    """仅构造根状态；角度为弧度制偏移量。"""
    root = default.clone()
    delta = torch.zeros_like(root[:, 3:7])
    delta[:, 0] = torch.cos(angles / 2)
    delta[:, 3] = torch.sin(angles / 2)
    root[:, 3:7] = quat_mul(root[:, 3:7], delta)
    root[:, :3] += origins
    up = torch.zeros_like(root[:, :3])
    up[:, 2] = 1
    local_up = quat_apply_inverse(root[:, 3:7], up)
    lowest = (local_up[:, :2] @ radial.T).amin(-1)
    lowest += torch.minimum(local_up[:, 2] * z_min, local_up[:, 2] * z_max)
    root[:, 2] = origins[:, 2] - lowest + clearance
    root[:, 7:] = 0
    return root


def randomize_initial_snake_roll(env, env_ids):
    low, high = env.cfg.initial_roll_range_deg
    clearance = env.cfg.initial_roll_clearance
    if not (math.isfinite(low) and math.isfinite(high) and -180 <= low <= high <= 180):
        raise ValueError("initial_roll_range_deg must satisfy -180 <= low <= high <= 180")
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("initial_roll_clearance must be finite and nonnegative")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not len(ids):
        return
    robot = env.scene["robot"]
    if not hasattr(env, "_snake_roll_envelope"):
        radial, z_min, z_max = collision_radial_envelope(tuple(robot.data.default_joint_pos[0].tolist()))
        env._snake_roll_envelope = (torch.tensor(radial, device=env.device, dtype=robot.data.default_root_state.dtype), z_min, z_max)
        env._initial_snake_roll_offset = torch.zeros(env.num_envs, device=env.device)
    radial, z_min, z_max = env._snake_roll_envelope
    angles = (torch.rand(len(ids), device=env.device) * (high - low) + low) * math.pi / 180
    states = rolled_root_states(robot.data.default_root_state[ids], env.scene.env_origins[ids], angles, radial, z_min, z_max, clearance)
    robot.write_root_state_to_sim(states, env_ids=ids)
    env._initial_snake_roll_offset[ids] = angles
    env._observation_generation = getattr(env, "_observation_generation", 0) + 1


def initial_roll_degrees_metric(env):
    return torch.rad2deg(env._initial_snake_roll_offset)
