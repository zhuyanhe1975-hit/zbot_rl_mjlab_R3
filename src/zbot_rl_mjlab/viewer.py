"""MJLab 原生查看器，支持按键切换机器人跟随相机。"""

import numpy as np
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import KEY_BACKSLASH, KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET

from .tasks.universal.env_cfg import UniversalEnvCfg
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


def follow_lookat(lookat, target):
    """将相机注视点的 x/y 移到目标位置，保持高度不变。"""
    moved = np.array(lookat, dtype=float)
    moved[:2] = np.asarray(target, dtype=float)[:2]
    return moved


class FollowCameraViewer(NativeMujocoViewer):
    """按 FOLLOW_KEY 让相机跟随机器人，再按一次解除跟随。

    跟随期间保留鼠标设置的方位角、仰角和距离，仅让注视点跟踪机器人
    （解除跟随前，拖动平移会被覆盖）。重写 sync_env_to_viewer 的子类
    必须自行调用 _apply_follow()。
    """

    def __init__(self, env, policy, **kwargs):
        self._universal_controls = isinstance(env.unwrapped.cfg, UniversalEnvCfg)
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
        elif key in (KEY_RIGHT_BRACKET, KEY_LEFT_BRACKET) and not getattr(self, "_snake_controls", False):
            self.request_action("CUSTOM", "transform_stand" if key == KEY_RIGHT_BRACKET else "transform_snake")

    def _motion_controls(self):
        return getattr(self, "_universal_controls", False) or getattr(self, "_snake_controls", False)

    def _handle_custom_action(self, action, payload):
        if action == ViewerAction.CUSTOM and isinstance(payload, str) and payload in (*_UNIVERSAL_DELTAS, "motion_stop"):
            if not self._motion_controls():
                return False
            env = self.env.unwrapped
            snake = getattr(self, "_snake_controls", False)
            term = env.command_manager.get_term("motion" if snake else "transform")
            if payload == "motion_stop":
                term.stop()
            else:
                current = term.requested_velocity[0].tolist()
                delta = (_SNAKE_DELTAS if snake else _UNIVERSAL_DELTAS)[payload]
                term.set_velocity(*(value + step for value, step in zip(current, delta)))
            env.observation_manager.compute(update_history=True)
            print(f"目标速度 vx/vy/wz：{term.requested_velocity[0].tolist()}")
            return True
        if action == ViewerAction.CUSTOM and payload == _FOLLOW_PAYLOAD:
            return self._toggle_follow()
        if action == ViewerAction.CUSTOM and payload in ("transform_stand", "transform_snake"):
            env = self.env.unwrapped
            if "transform" not in env.cfg.commands:
                return False
            target = "stand" if payload == "transform_stand" else "snake"
            env.command_manager.get_term("transform").set_target(target)
            env.observation_manager.compute(update_history=True)
            print(f"[INFO] Transform command: {target}; manual control, automatic resets disabled")
            return True
        return super()._handle_custom_action(action, payload)

    def _toggle_follow(self):
        self._follow = not self._follow
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
        with v.lock():
            v.cam.lookat[:] = follow_lookat(v.cam.lookat, self._follow_point())

    def sync_env_to_viewer(self):
        self._apply_follow()
        super().sync_env_to_viewer()
