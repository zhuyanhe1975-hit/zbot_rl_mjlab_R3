"""Zbot MJLab 任务的双机器人仿真对照查看器。"""

import argparse
import pathlib
import sys
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.cli import latest_checkpoint
from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS, robot_cfg
from zbot_rl_mjlab.viewer import FollowCameraViewer, _AsyncOverlayHandle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    choices=[t for t in list_tasks() if t.startswith("Mjlab-Zbot")],
    default="Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking",
)
parser.add_argument("--checkpoint", type=pathlib.Path)
parser.add_argument("--frequency", type=float, default=0.5)
parser.add_argument("--target-speed", type=float, default=0.25)
parser.add_argument("--lateral-speed", type=float, default=0.0)
parser.add_argument("--yaw-rate", type=float, default=0.0)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--force-scale", type=float, default=1.0)
parser.add_argument("--torque-scale", type=float, default=1.0)
args = parser.parse_args()
TASK_ID = args.task
if args.force_scale < 0 or args.torque_scale < 0:
    raise ValueError("force-scale and torque-scale must be non-negative")
CHECKPOINT = args.checkpoint or latest_checkpoint(
    load_rl_cfg(TASK_ID).experiment_name, ROOT / "logs" / "rsl_rl"
)
print(f"Task: {TASK_ID}\nCheckpoint: {CHECKPOINT}\nFrequency: {args.frequency} Hz")

TASK_ENV_CFG = load_env_cfg(TASK_ID, play=True)
OBS_TERMS = tuple(TASK_ENV_CFG.observations["actor"].terms)
ACTION_SCALE = float(TASK_ENV_CFG.actions["joint_pos"].scale)
print(f"Action scale: {ACTION_SCALE}")

# 同一场景中放置两个机器人副本："ref"（MJLab，仅供显示）和 "native"。
scene_cfg = deepcopy(TASK_ENV_CFG.scene)
scene_cfg.entities = {"ref": robot_cfg(), "native": robot_cfg()}
scene_cfg.sensors = ()
m = Scene(scene_cfg, device="cpu").compile()
TASK_ENV_CFG.sim.mujoco.apply(m)
d = mujoco.MjData(m)
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["actor_state_dict"]


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", ck["obs_normalizer._mean"])
        self.register_buffer("std", ck["obs_normalizer._std"])
        weights = {
            k: v
            for k, v in ck.items()
            if k.startswith("mlp.") and k.endswith(".weight")
        }
        layers = []
        for index, name in enumerate(
            sorted(weights, key=lambda x: int(x.split(".")[1]))
        ):
            out_dim, in_dim = weights[name].shape
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if index < len(weights) - 1:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)
        self.net.load_state_dict(
            {k[4:]: v for k, v in ck.items() if k.startswith("mlp.")}
        )

    def forward(self, x):
        return self.net((x - self.mean) / (self.std + 1e-2))


actor = Actor().eval()
default = np.array(DEFAULT_JOINT_POS, dtype=np.float64)
STEP_DT = TASK_ENV_CFG.sim.mujoco.timestep * TASK_ENV_CFG.decimation
FREQ_MIN = TASK_ENV_CFG.step_frequency_min
FREQ_MAX = TASK_ENV_CFG.step_frequency_max


def adr(prefix):
    return [
        m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/joint{i}")
        ]
        for i in range(1, 7)
    ]


def dof(prefix):
    return [
        m.jnt_dofadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/joint{i}")
        ]
        for i in range(1, 7)
    ]


def body(prefix):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/base")


rq, nq = adr("ref"), adr("native")
rd, nd = dof("ref"), dof("native")
rb, nb = body("ref"), body("native")


def native_ctrl_slots():
    """按名称查找原生机器人六个位置目标对应的控制索引。

    机器人每个关节使用一个以关节名命名的 <position> 伺服执行器，因此按名称查找，
    避免假设原生机器人在 ``ctrl`` 中排在第二位。
    """
    slots = []
    for i in range(1, 7):
        name = f"native/joint{i}"
        index = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if index < 0:
            raise KeyError(f"actuator {name} not found; the robot's actuator changed")
        slots.append(index)
    return np.array(slots)


native_ctrl = native_ctrl_slots()


def locomotion_frame(r):
    """复现 mdp.locomotion_frame：forward = gravity x body_z_h，left = forward x gravity。"""
    gravity = np.array([0.0, 0.0, -1.0])
    z = r[:, 2].copy()
    z[2] = 0
    z /= max(np.linalg.norm(z), 1e-6)
    forward = np.cross(gravity, z)
    forward /= max(np.linalg.norm(forward), 1e-6)
    left = np.cross(forward, gravity)
    left /= max(np.linalg.norm(left), 1e-6)
    return forward, left, gravity


obs_sizes = []  # 各观测项的维数，由 obs() 填充


def obs(b, qadr, dadr, last, step):
    """按 MJLab 任务中相同的观测项顺序重建策略网络观测。"""
    r = d.xmat[b].reshape(3, 3)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    lin_w = vel[3:] + np.cross(vel[:3], d.xpos[b] - d.xipos[b])
    axes = locomotion_frame(r)
    forward = axes[0]
    initial = initial_forward
    phase = step * STEP_DT * args.frequency * 2 * np.pi
    values = {
        "base_lin_vel": [lin_w @ axis for axis in axes],
        "base_ang_vel": [vel[:3] @ axis for axis in axes],
        "projected_gravity": r.T @ np.array([0.0, 0.0, -1.0]),
        "yaw_error": [
            np.arctan2(
                forward[0] * initial[1] - forward[1] * initial[0],
                forward[:2] @ initial[:2],
            )
        ],
        "joint_pos": d.qpos[qadr] - default,
        "joint_vel": d.qvel[dadr],
        "actions": last,
        "phase": [np.sin(phase), np.cos(phase)],
        "frequency": [(args.frequency - FREQ_MIN) / max(FREQ_MAX - FREQ_MIN, 1e-6)],
        "target_speed": [args.target_speed],
        "velocity_command": [args.target_speed, args.lateral_speed, args.yaw_rate],
    }
    missing = [t for t in OBS_TERMS if t not in values]
    if missing:
        raise KeyError(f"Native observation has no builder for terms: {missing}")
    parts = [np.ravel(values[t]) for t in OBS_TERMS]
    obs_sizes[:] = [p.size for p in parts]
    value = np.concatenate(parts).astype(np.float32)
    if value.size != int(ck["obs_normalizer._mean"].numel()):
        raise ValueError(
            "Native observation dimension "
            f"{value.size} does not match training config/checkpoint "
            f"{ck['obs_normalizer._mean'].numel()}"
        )
    return value


# 外部 MJLab 环境实例
a = TASK_ENV_CFG
a.test_frequency = args.frequency
a.scene.num_envs = 1
if "target_speed" in a.commands:
    ranges = a.commands["target_speed"].ranges
    ranges.lin_vel_x = (args.target_speed,) * 2
    ranges.lin_vel_y = (args.lateral_speed,) * 2
    ranges.ang_vel_z = (args.yaw_rate,) * 2
    print(
        f"Command: vx {args.target_speed} m/s, vy {args.lateral_speed} m/s, "
        f"wz {args.yaw_rate} rad/s"
    )
if "base_disturbance" in a.events:
    params = a.events["base_disturbance"].params
    params["disturbed_fraction"] = 1.0
    params["force_range"] = tuple(v * args.force_scale for v in params["force_range"])
    params["torque_range"] = tuple(
        v * args.torque_scale for v in params["torque_range"]
    )
    print(
        f"Disturbance force scale: {args.force_scale}; torque scale: {args.torque_scale}"
    )
env = ManagerBasedRlEnv(a, device=args.device)
agent = load_rl_cfg(TASK_ID)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
runner.load(
    str(CHECKPOINT), load_cfg={"actor": True}, strict=True, map_location=args.device
)
policy = runner.get_inference_policy(device=args.device)


def root_adr(prefix):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/root")
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]


NATIVE_ROOT_Q, _ = root_adr("native")
REF_ROOT_Q, REF_ROOT_D = root_adr("ref")
DISTURBED = "base_disturbance" in a.events
ENV_BASE = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/base")
initial_forward = np.zeros(3)
last = np.zeros(6)
native_step = 0


def place_ref():
    """在原生机器人旁显示 MJLab 机器人（仅供显示）。"""
    src = env.sim.data.qpos[0].detach().cpu().numpy()
    d.qpos[REF_ROOT_Q : REF_ROOT_Q + 7] = src[:7] + np.array([0, 0.5, 0, 0, 0, 0, 0])
    d.qpos[rq] = src[7:13]
    # 禁止 MuJoCo 自行推进参考机器人的刚体状态。
    d.qvel[REF_ROOT_D : REF_ROOT_D + 6] = 0
    d.qvel[rd] = 0


def reset_native():
    """用 MJLab 重置姿态初始化原生机器人，并在侧向错开 1 m。"""
    global initial_forward, last, native_step
    mujoco.mj_resetData(m, d)
    reset_qpos = env.sim.data.qpos[0].detach().cpu().numpy()
    d.qpos[NATIVE_ROOT_Q : NATIVE_ROOT_Q + 7] = reset_qpos[:7] + np.array(
        [0, -0.5, 0, 0, 0, 0, 0]
    )
    d.qpos[nq] = default
    place_ref()
    mujoco.mj_forward(m, d)
    initial_forward = locomotion_frame(d.xmat[nb].reshape(3, 3))[0]
    last = np.zeros(6)
    native_step = 0


def check_obs(native, reference, atol=1e-3):
    """两个机器人从相同状态开始，因此首次观测应当一致。"""
    start = 0
    for name, size in zip(OBS_TERMS, obs_sizes):
        err = np.abs(native[start : start + size] - reference[start : start + size])
        if err.max() > atol:
            print(f"[WARN] native obs '{name}' differs from MJLab by {err.max():.4f}")
        start += size


def paired_policy(mj_obs):
    """为环境生成 MJLab 动作，同时在旁边执行一个原生控制步。"""
    global last, native_step
    ref_action = policy(mj_obs)
    native_obs = obs(nb, nq, nd, last, native_step)
    if native_step == 0:
        check_obs(native_obs, mj_obs["actor"][0].cpu().numpy())
    native_action = actor(torch.from_numpy(native_obs)[None]).numpy()[0]
    last = native_action.copy()
    native_step += 1
    place_ref()
    d.ctrl[:] = 0
    d.ctrl[native_ctrl] = default + ACTION_SCALE * native_action
    for _ in range(a.decimation):
        mujoco.mj_step(m, d)
    return ref_action


class CompareViewer(FollowCameraViewer):
    """在双机器人场景中使用 mjlab 原生查看器（含实时叠加信息、暂停和速度快捷键）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, enable_perturbations=False, **kwargs)

    def setup(self):
        self.mjm, self.mjd = m, d
        self.vopt = mujoco.MjvOption()
        self.viewer = mujoco.viewer.launch_passive(
            m,
            d,
            key_callback=self._safe_key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer = _AsyncOverlayHandle(self.viewer)
        v = self.viewer
        # 用户场景中的标签几何体固定在世界坐标系内，用于区分控制器。
        v.user_scn.ngeom = 2
        for geom, label, color in zip(
            v.user_scn.geoms[:2],
            ("pytorch+mujoco", "mjlab"),
            ((0.2, 0.7, 1.0, 1.0), (0.2, 1.0, 0.4, 1.0)),
        ):
            geom.type = mujoco.mjtGeom.mjGEOM_LABEL
            geom.size[:] = (0.12, 0.12, 0.12)
            geom.rgba[:] = color
            geom.label = label
        v.cam.distance = 2.2
        v.cam.azimuth = 135
        v.cam.elevation = -20
        v.cam.lookat[:] = [0, -0.06, 0.2]
        reset_native()

    def reset_environment(self):
        super().reset_environment()
        reset_native()

    def _follow_point(self):
        # 跟随两个机器人的中点，使它们同时保持在视野内。
        return (d.xpos[nb] + d.xpos[rb]) / 2

    def sync_viewer_to_env(self):
        pass  # 合并场景不使用鼠标扰动。

    def sync_env_to_viewer(self):
        v = self.viewer
        if not v.is_running():
            return
        with self._mj_lock:
            place_ref()
            mujoco.mj_forward(m, d)
            self._apply_follow()
            if not v.is_running():
                return
            self._set_status_overlay(v)
            v.user_scn.ngeom = 2
            if DISTURBED:
                # 在参考机器人上用粉色绘制当前施加的 MJLab 扰动。
                force = (
                    env.sim.data.xfrc_applied[0, ENV_BASE, :3].detach().cpu().numpy()
                )
                if np.linalg.norm(force) > 1e-5:
                    start = d.xpos[rb].copy()
                    geom = v.user_scn.geoms[v.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_ARROW.value,
                        np.zeros(3),
                        np.zeros(3),
                        np.zeros(9),
                        np.array((0.95, 0.05, 0.75, 1.0), dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_ARROW.value,
                        0.025,
                        start,
                        start + force * 0.04,
                    )
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    v.user_scn.ngeom += 1
            for geom, b in zip(v.user_scn.geoms[:2], (nb, rb)):
                geom.pos[:] = [d.xpos[b, 0], d.xpos[b, 1], d.xpos[b, 2] + 0.24]
            v.sync()


wrapped.reset()
try:
    CompareViewer(wrapped, paired_policy).run()
finally:
    env.close()
