"""MJLab 原生查看器，支持按键切换机器人跟随相机。"""

import math
import time

import mujoco
import numpy as np
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import KEY_BACKSLASH, KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET

from .tasks.universal.env_cfg import UniversalEnvCfg
from .tasks.universal.task_fsm import Morphology, TaskState
from .tasks.walking import mdp

# MuJoCo 和 MJLab 已绑定大部分字母，因此使用标点键。
FOLLOW_KEY = KEY_BACKSLASH
_FOLLOW_PAYLOAD = "zbot_follow_camera"
# 按键逐次调整目标速度，X 清零；由仿真线程执行并平滑到实际指令。
_UNIVERSAL_KEYS = {
    ord("W"): "motion_forward", ord("S"): "motion_backward",
    ord("A"): "motion_left", ord("D"): "motion_right",
    ord("Q"): "motion_turn_left", ord("E"): "motion_turn_right",
    ord("X"): "motion_stop",
}
_UNIVERSAL_DELTAS = {
    "motion_forward": (.05, 0., 0.), "motion_backward": (-.05, 0., 0.),
    "motion_left": (0., .05, 0.), "motion_right": (0., -.05, 0.),
    "motion_turn_left": (0., 0., .1), "motion_turn_right": (0., 0., -.1),
}
_SNAKE_DELTAS = {
    **_UNIVERSAL_DELTAS,
    "motion_forward": (.025, 0., 0.), "motion_backward": (-.025, 0., 0.),
    "motion_left": (0., .025, 0.), "motion_right": (0., -.025, 0.),
}


def task_fsm_overlay_rows(term, env_idx):
    """Return the selected robot's latched task and velocity, if available."""
    fsm = getattr(term, "task_fsm", None)
    if fsm is None:
        return []
    morphology = Morphology(int(fsm.current_morphology[env_idx]))
    rows = [("Morphology", morphology.name)]
    if morphology == Morphology.TRANSITION:
        rows.append(("Source", Morphology(int(fsm.source_morphology[env_idx])).name))
    rows.extend([
        ("Target", Morphology(int(fsm.target_morphology[env_idx])).name),
        ("Task", TaskState(int(fsm.task_state[env_idx])).name),
        ("Requested vx/vy/wz", " / ".join(f"{value:+.3f}" for value in term.requested_velocity[env_idx].tolist())),
        ("Effective vx/vy/wz", " / ".join(f"{value:+.3f}" for value in term.velocity_command[env_idx].tolist())),
    ])
    if hasattr(term, 'retry_attempts') and int(fsm.task_state[env_idx]) == TaskState.ASCENDING:
        stage = 'STRAIGHTEN' if bool(term.straightening_mask[env_idx]) else 'RISE'
        rows.extend([('Stage', stage), ('Retries', str(int(term.retry_attempts[env_idx])))])
    return rows


def follow_lookat(lookat, target, *, dt=0., smoothing_time=0.):
    """按时间常数低通跟踪XY；默认零时间常数保持直接跟踪的接口。"""
    if not math.isfinite(smoothing_time) or smoothing_time < 0:
        raise ValueError('follow smoothing time must be finite and nonnegative')
    if not math.isfinite(dt) or dt < 0:
        raise ValueError('follow dt must be finite and nonnegative')
    moved = np.array(lookat, dtype=float)
    desired = np.asarray(target, dtype=float)[:2]
    if np.isfinite(desired).all():
        alpha = -math.expm1(-dt / smoothing_time) if smoothing_time > 0 else 1.
        moved[:2] += alpha * (desired - moved[:2])
    return moved


class FollowCameraViewer(NativeMujocoViewer):
    """按 FOLLOW_KEY 让相机跟随机器人，再按一次解除跟随。

    跟随期间保留鼠标设置的方位角、仰角和距离，仅让注视点跟踪机器人
    （解除跟随前，拖动平移会被覆盖）。重写 sync_env_to_viewer 的子类
    必须自行调用 _apply_follow()。
    """

    def __init__(self, env, policy, *, follow_smoothing_time=.2, **kwargs):
        if not math.isfinite(follow_smoothing_time) or follow_smoothing_time < 0:
            raise ValueError('follow smoothing time must be finite and nonnegative')
        self.follow_smoothing_time = follow_smoothing_time
        self._follow_last_time = None
        self._universal_controls = isinstance(env.unwrapped.cfg, UniversalEnvCfg)
        self._site_snake_controls = getattr(env.unwrapped.cfg, 'site_snake_omni', False)
        self._snake_controls = ("motion" in env.unwrapped.cfg.commands
                                and getattr(env.unwrapped.cfg, "observation_reference", "command") == "command")
        super().__init__(env, policy, key_callback=self._on_key, **kwargs)
        self._follow = False

    def _safe_key_callback(self, key):
        # 避免 A 同时触发 MJLab 的“显示全部环境”；此线程只入队。
        if self._motion_controls() and key in _UNIVERSAL_KEYS:
            self._on_key(key)
            return
        return super()._safe_key_callback(key)

    def _on_key(self, key):
        # 运行于 MuJoCo 查看器线程：仅将请求加入队列。
        if self._motion_controls() and key in _UNIVERSAL_KEYS:
            self.request_action("CUSTOM", _UNIVERSAL_KEYS[key])
        elif key == FOLLOW_KEY:
            self.request_action("CUSTOM", _FOLLOW_PAYLOAD)
        elif (key in (KEY_RIGHT_BRACKET, KEY_LEFT_BRACKET)
              and not getattr(self, "_snake_controls", False)
              and not getattr(self, '_site_snake_controls', False)):
            self.request_action("CUSTOM", "transform_stand" if key == KEY_RIGHT_BRACKET else "transform_snake")

    def _motion_controls(self):
        return getattr(self, "_universal_controls", False) or getattr(self, "_snake_controls", False)

    def _set_status_overlay(self, viewer):
        env = self.env.unwrapped
        term = env.command_manager.get_term("transform") if "transform" in env.cfg.commands else None
        rows = task_fsm_overlay_rows(term, self.env_idx)
        if not rows:
            return super()._set_status_overlay(viewer)
        status = self.get_status()
        capped = " [CAPPED]" if status.capped else ""
        text_1 = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT"
        text_2 = (
            f"{self.env_idx + 1}/{self.env.num_envs}\n"
            f"{status.step_count}\n"
            f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
            f"{status.speed_label}\n"
            f"{status.target_realtime:.2f}x\n"
            f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)"
        )
        text_1 += "\n" + "\n".join(label for label, _ in rows)
        text_2 += "\n" + "\n".join(value for _, value in rows)
        viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_150.value,  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,  # pyright: ignore[reportAttributeAccessIssue]
            text_1,
            text_2,
        ))

    def _handle_custom_action(self, action, payload):
        if action == ViewerAction.CUSTOM and isinstance(payload, str) and payload in (*_UNIVERSAL_DELTAS, "motion_stop"):
            if not self._motion_controls():
                return False
            env = self.env.unwrapped
            snake = getattr(self, "_snake_controls", False)
            term = env.command_manager.get_term("motion" if snake else "transform")
            env_ids = torch.tensor([self.env_idx], device=env.device, dtype=torch.long)
            if payload == "motion_stop":
                term.stop(env_ids)
            else:
                current = term.requested_velocity[self.env_idx].tolist()
                delta = (_SNAKE_DELTAS if snake or getattr(self, '_site_snake_controls', False) else _UNIVERSAL_DELTAS)[payload]
                term.set_velocity(*(value + step for value, step in zip(current, delta)), env_ids=env_ids)
            env.observation_manager.compute(update_history=True, env_ids=env_ids)
            print(f"机器人 {self.env_idx + 1}/{env.num_envs} 目标速度 vx/vy/wz：{term.requested_velocity[self.env_idx].tolist()}")
            return True
        if action == ViewerAction.CUSTOM and payload == _FOLLOW_PAYLOAD:
            return self._toggle_follow()
        if action == ViewerAction.CUSTOM and payload in ("transform_stand", "transform_snake"):
            env = self.env.unwrapped
            if getattr(env.cfg, 'site_snake_omni', False):
                return True
            if "transform" not in env.cfg.commands:
                return False
            target = "stand" if payload == "transform_stand" else "snake"
            env_ids = torch.tensor([self.env_idx], device=env.device, dtype=torch.long)
            env.command_manager.get_term("transform").set_target(target, env_ids)
            env.observation_manager.compute(update_history=True, env_ids=env_ids)
            print(f"[INFO] Robot {self.env_idx + 1}/{env.num_envs}: {target}; manual control, automatic resets disabled")
            return True
        return super()._handle_custom_action(action, payload)

    def _toggle_follow(self):
        self._follow = not self._follow
        self._follow_last_time = None
        print(f"[INFO] Camera follow {'ON' if self._follow else 'OFF'}")
        return True

    def _follow_point(self):
        """相机跟随的世界坐标位置，即所选环境的基座位置。"""
        robot, i = mdp._base(self.env.unwrapped)
        return robot.data.body_link_pos_w[self.env_idx, i].cpu().numpy()

    def _apply_follow(self):
        v = self.viewer
        if not self._follow or v is None or not v.is_running():
            return
        now = time.perf_counter()
        dt = self.frame_time if self._follow_last_time is None else max(0., now - self._follow_last_time)
        self._follow_last_time = now
        target = self._follow_point()
        with v.lock():
            v.cam.lookat[:] = follow_lookat(v.cam.lookat, target, dt=dt, smoothing_time=self.follow_smoothing_time)

    def sync_env_to_viewer(self):
        self._apply_follow()
        super().sync_env_to_viewer()
