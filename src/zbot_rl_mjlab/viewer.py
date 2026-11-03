"""MJLab 原生查看器，支持按键切换机器人跟随相机。"""

import math
import time
from copy import copy
from threading import Condition, Thread

import mujoco
import numpy as np
import torch
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import (
    KEY_BACKSLASH,
    KEY_DOWN,
    KEY_KP_0,
    KEY_KP_2,
    KEY_KP_4,
    KEY_KP_5,
    KEY_KP_6,
    KEY_KP_7,
    KEY_KP_8,
    KEY_KP_9,
    KEY_LEFT,
    KEY_LEFT_BRACKET,
    KEY_RIGHT,
    KEY_RIGHT_BRACKET,
    KEY_UP,
)

from .tasks.snake.rz270_heading import HeadingSnakeCommand, HeadingSnakeCommandCfg
from .tasks.universal import mdp as universal_mdp
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
_HEADING_KEYS = {
    **_UNIVERSAL_KEYS,
    KEY_UP: 'motion_forward', KEY_DOWN: 'motion_backward',
    KEY_LEFT: 'motion_left', KEY_RIGHT: 'motion_right',
    KEY_KP_8: 'motion_forward', KEY_KP_2: 'motion_backward',
    KEY_KP_4: 'motion_left', KEY_KP_6: 'motion_right',
    KEY_KP_7: 'motion_turn_left', KEY_KP_9: 'motion_turn_right',
    KEY_KP_5: 'motion_stop', KEY_KP_0: 'motion_stop',
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


class _AsyncOverlayHandle:
    """Keep MuJoCo's overlay acknowledgement waits off the simulation thread.

    A close can race with set_texts/set_figures even after is_running(). Keep
    only the latest update per kind; a stalled renderer must not build a queue
    or prevent the main loop from reaching close().
    """

    def __init__(self, handle):
        self._handle = handle
        self._condition = Condition()
        self._pending = {}
        self._closed = False
        self._error = None
        self._thread = Thread(target=self._run, name="mujoco-overlays", daemon=True)
        self._thread.start()

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def _submit(self, kind, value):
        with self._condition:
            if self._error is not None:
                raise RuntimeError("MuJoCo overlay update failed") from self._error
            if not self._closed:
                self._pending[kind] = value
                self._condition.notify()

    def set_texts(self, texts):
        self._submit("set_texts", texts)

    def set_figures(self, figures):
        # MJLab mutates its figure buffers on the next frame. The worker needs
        # its own snapshots, including the viewport structs.
        if isinstance(figures, tuple):
            figures = [figures]
        self._submit("set_figures", [(copy(rect), copy(fig)) for rect, fig in figures])

    def _run(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    pending = self._pending
                    self._pending = {}
                for kind, value in pending.items():
                    if not self._handle.is_running():
                        return
                    getattr(self._handle, kind)(value)
        except Exception as error:  # noqa: BLE001 — propagate worker errors on the main thread.
            with self._condition:
                self._error = error

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify()
        # Never join: an overlay call may still be waiting for the closed
        # render loop. The daemon retains the handle until that call returns.
        self._handle.close()


def task_fsm_overlay_rows(term, env_idx, actual_velocity):
    """Return the selected robot's latched task and velocity, if available."""
    fsm = getattr(term, "task_fsm", None)
    if fsm is None:
        return []
    morphology = Morphology(int(fsm.current_morphology[env_idx]))
    rows = [("Morphology", morphology.name)]
    if morphology == Morphology.TRANSITION:
        rows.append(("Source", Morphology(int(fsm.source_morphology[env_idx])).name))
    if isinstance(term, HeadingSnakeCommand):
        target = float(term.target_heading[env_idx])
        forward = term.heading[env_idx]
        current = math.atan2(float(forward[1]), float(forward[0]))
        error = math.atan2(math.sin(target - current), math.cos(target - current))
        rows.extend([
            ('Heading target (deg)', f'{math.degrees(target):+.1f}'),
            ('Heading error (deg)', f'{math.degrees(error):+.1f}'),
            ('Forward command (m/s)', f'{float(term.requested_velocity[env_idx, 0]):.3f}'),
            ('Head site vx/vy/wz', ' / '.join(f'{value:+.3f}' for value in actual_velocity[env_idx].tolist())),
            ('Heading control', 'MANUAL' if bool(term.manual_control[env_idx]) else 'AUTO'),
            ('Speed keys', 'W/S or Up/Down: +/-0.025 m/s | X: zero'),
            ('Heading keys', 'A/D, Q/E or Left/Right: +/-10 deg'),
            ('Keypad', '8/2: speed | 4/6 or 7/9: heading | 5/0: zero speed'),
            ('Heading arrows', 'Target: orange | Head: green'),
            ('Actual arrows', 'XY velocity: cyan | Yaw rate wz: purple'),
        ])
        return rows
    rows.extend([
        ("Target", Morphology(int(fsm.target_morphology[env_idx])).name),
        ("Task", TaskState(int(fsm.task_state[env_idx])).name),
        ("Requested vx/vy/wz", " / ".join(f"{value:+.3f}" for value in term.requested_velocity[env_idx].tolist())),
        ("Actual vx/vy/wz", " / ".join(f"{value:+.3f}" for value in actual_velocity[env_idx].tolist())),
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
        self._docking_controls = 'dock' in env.unwrapped.cfg.commands
        self._heading_controls = isinstance(env.unwrapped.cfg.commands.get('transform'), HeadingSnakeCommandCfg)
        self._snake_controls = ("motion" in env.unwrapped.cfg.commands
                                and getattr(env.unwrapped.cfg, "observation_reference", "command") == "command")
        super().__init__(env, policy, key_callback=self._on_key, **kwargs)
        self._follow = False

    def setup(self):
        super().setup()
        self.viewer = _AsyncOverlayHandle(self.viewer)

    def _safe_key_callback(self, key):
        # 避免 A 同时触发 MJLab 的“显示全部环境”；此线程只入队。
        if self._motion_controls() and key in self._motion_keys():
            self._on_key(key)
            return
        return super()._safe_key_callback(key)

    def _on_key(self, key):
        # 运行于 MuJoCo 查看器线程：仅将请求加入队列。
        if self._motion_controls() and key in self._motion_keys():
            self.request_action("CUSTOM", self._motion_keys()[key])
        elif (getattr(self, '_docking_controls', False) and key in (KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET)):
            self.request_action('CUSTOM', 'dock_previous' if key == KEY_LEFT_BRACKET else 'dock_next')
        elif key == FOLLOW_KEY:
            self.request_action("CUSTOM", _FOLLOW_PAYLOAD)
        elif (key in (KEY_RIGHT_BRACKET, KEY_LEFT_BRACKET)
              and not getattr(self, "_snake_controls", False)
              and not getattr(self, '_site_snake_controls', False)):
            self.request_action("CUSTOM", "transform_stand" if key == KEY_RIGHT_BRACKET else "transform_snake")

    def _motion_keys(self):
        return _HEADING_KEYS if getattr(self, '_heading_controls', False) else _UNIVERSAL_KEYS

    def _motion_controls(self):
        return getattr(self, "_universal_controls", False) or getattr(self, "_snake_controls", False)

    def _sync_env_state_to_mjdata(self, mjd, sim_data, env_idx):
        super()._sync_env_state_to_mjdata(mjd, sim_data, env_idx)
        if mjd.eq_active.size:
            mjd.eq_active[:] = sim_data.eq_active[env_idx].cpu().numpy()

    def _set_status_overlay(self, viewer):
        env = self.env.unwrapped
        term = env.command_manager.get_term("transform") if "transform" in env.cfg.commands else None
        rows = (task_fsm_overlay_rows(term, self.env_idx, universal_mdp.tracking_velocity(env))
                if getattr(term, 'task_fsm', None) is not None else [])
        if getattr(env.cfg, 'head_tail_docking', False):
            from .tasks.reconfiguration.mdp import state
            dock = state(env)
            g = dock.geometry()
            i = self.env_idx
            rows = [('Reconfiguration', 'Head-tail passive magnetic docking'),
                    ('Bond', 'CONNECTED' if dock.enabled()[i].any() else 'OPEN'),
                    ('Port gap', f"{g['distance'][i].item() * 1000:.1f} mm"),
                    ('Port angle', f"{g['angle'][i].item() * 180 / math.pi:.1f} deg"),
                    ('Hold', f'{dock.hold_time[i].item():.2f} / {env.cfg.success_hold_s:.2f} s'),
                    ('Weld load', f'{dock.force[i].item():.1f} N / {dock.torque[i].item():.2f} Nm')]
            if 'dock' in env.cfg.commands:
                rows.append(('Dock target', f"{env.command_manager.get_term('dock').quarter[i].item() * 90} deg"))
            if hasattr(env.cfg, 'pose_hold_s'):
                from .tasks.reconfiguration import plan_guided
                pose = plan_guided.progress(env)
                stable = pose.count[i].item() >= math.ceil(env.cfg.pose_hold_s/env.step_dt-1e-8)
                rows.extend((('Pose stable now', 'YES' if stable else 'NO'),
                             ('Ever pose success', 'YES' if pose.achieved[i] else 'NO'),
                             ('Pose hold', f'{pose.count[i].item()*env.step_dt:.2f} / {env.cfg.pose_hold_s:.2f} s'),
                             ('Normal angle', f'{plan_guided.normal_angle(env)[i].item()*180/math.pi:.2f} deg')))
            if any(sensor.name == 'end_to_end_contact' for sensor in env.cfg.scene.sensors):
                rows.append(('End pressure', f"{env.scene['end_to_end_contact'].data.force[i, ..., 0].abs().max().item():.1f} N"))
        if getattr(env.cfg, 'disconnect_torque', False):
            from .tasks.reconfiguration.disconnect.state import state
            connectors = state(env)
            i = self.env_idx
            rows = [('Torque disconnect', f'Target connector {env.cfg.target_connector}'),
                    ('Connections 0..5', ' '.join('ON' if flag else 'OPEN' for flag in connectors.connected()[i].tolist())),
                    ('Target gap', f'{connectors.geometry()["distance"][i].item()*1000:.1f} mm'),
                    ('Port gaps 0..5 mm', ' '.join(f'{connectors.geometry(j)["distance"][i].item()*1000:.2f}' for j in range(6))),
                    ('Port angles 0..5 deg', ' '.join(f'{connectors.geometry(j)["angle"][i].item()*180/math.pi:.2f}' for j in range(6))),
                    ('Other min margin', f'{connectors.other_margin()[i].item():.3f}'),
                    ('Wrong break', 'YES' if connectors.wrong_break[i] else 'NO')]
        if not rows:
            return super()._set_status_overlay(viewer)
        if getattr(env.cfg, 'command_visual_actual_velocity', False):
            rows.append(('Velocity arrows', 'Command: blue/orange | Actual: green/purple'))
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
        if action == ViewerAction.CUSTOM and payload in ('dock_previous', 'dock_next'):
            env = self.env.unwrapped
            if 'dock' not in env.cfg.commands:
                return False
            term = env.command_manager.get_term('dock')
            choices = tuple(term.cfg.angles_deg)
            current = int(term.quarter[self.env_idx].item()) * 90
            index = (choices.index(current) + (1 if payload == 'dock_next' else -1)) % len(choices)
            ids = torch.tensor([self.env_idx], device=env.device, dtype=torch.long)
            term.set_target(choices[index], ids)
            env.observation_manager.compute(update_history=True, env_ids=ids)
            return True

        if action == ViewerAction.CUSTOM and isinstance(payload, str) and payload in (*_UNIVERSAL_DELTAS, "motion_stop"):
            if not self._motion_controls():
                return False
            env = self.env.unwrapped
            snake = getattr(self, "_snake_controls", False)
            term = env.command_manager.get_term("motion" if snake else "transform")
            env_ids = torch.tensor([self.env_idx], device=env.device, dtype=torch.long)
            if isinstance(term, HeadingSnakeCommand):
                if payload == 'motion_stop':
                    term.stop(env_ids)
                else:
                    if payload in ('motion_forward', 'motion_backward'):
                        step = .025 * (1 if payload == 'motion_forward' else -1)
                        term.set_forward_speed(float(term.requested_velocity[self.env_idx, 0]) + step, env_ids)
                    else:
                        step = math.radians(10) * (1 if payload in ('motion_left', 'motion_turn_left') else -1)
                        term.set_heading(float(term.target_heading[self.env_idx]) + step, env_ids)
                env.observation_manager.compute(update_history=True, env_ids=env_ids)
                print(f'机器人 {self.env_idx + 1}/{env.num_envs} 目标航向 '
                      f'{math.degrees(float(term.target_heading[self.env_idx])):+.1f}°，'
                      f'前进指令 {float(term.requested_velocity[self.env_idx, 0]):.3f} m/s')
                return True
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
