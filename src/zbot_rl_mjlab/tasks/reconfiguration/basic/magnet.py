"""Uncalibrated finite-range magnetic attraction and breakable capture approximation.

The sole frames intentionally have matching +Z at a valid face-to-face mating:
head outward normal is -sole.Z, tail outward normal is +sole.Z. Four quarter-turn
registrations are explicit prototype assumptions. No policy action toggles a bond.
"""

import math

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from .model import DOCK_SITES, TORQUE_SCALE, WELD_NAMES


class MagneticDock:
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg
        self.validate()
        robot = env.scene['robot']
        self.robot = robot
        self.sites = [robot.find_sites(name)[0][0] for name in ('foot_0_sole', 'foot_1_sole')]
        self.dock_sites = torch.tensor([robot.find_sites(name)[0][0] for name in DOCK_SITES], device=env.device)
        self.bodies = [robot.find_bodies(name)[0][0] for name in ('foot_0', 'foot_1')]
        self.body_ids = robot.indexing.body_ids[self.bodies].long()
        model = env.sim.mj_model
        self.eq_ids = torch.tensor([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                                   f'robot/{name}') for name in WELD_NAMES], device=env.device)
        if (self.eq_ids < 0).any():
            raise ValueError('head-tail equality constraints missing from model')
        self.allowed = torch.tensor([i in self.cfg.allowed_dock_quarters for i in range(4)], device=env.device)
        n = env.num_envs
        self.capture_time = torch.zeros(n, device=env.device)
        self.hold_time = torch.zeros_like(self.capture_time)
        self.cooldown = torch.zeros_like(self.capture_time)
        self.connections = torch.zeros_like(self.capture_time)
        self.breaks = torch.zeros_like(self.capture_time)
        self.force = torch.zeros_like(self.capture_time)
        self.torque = torch.zeros_like(self.capture_time)
        self.succeeded = torch.zeros(n, device=env.device, dtype=torch.bool)
        self.just_succeeded = torch.zeros_like(self.succeeded)
        self.previous_wrench = torch.zeros(n, 2, 6, device=env.device)

    def validate(self):
        positive = ('magnetic_range', 'magnetic_force', 'capture_distance', 'capture_angle',
                    'capture_speed', 'capture_angular_speed', 'capture_hold_s', 'release_distance',
                    'break_force', 'break_torque', 'success_hold_s', 'recapture_cooldown_s', 'workspace_radius')
        for name in positive:
            value = getattr(self.cfg, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('magnetic_linear_damping', 'magnetic_angular_stiffness', 'magnetic_angular_damping'):
            value = getattr(self.cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if not self.cfg.capture_distance < self.cfg.release_distance <= self.cfg.magnetic_range:
            raise ValueError('require capture_distance < release_distance <= magnetic_range')
        if not 0 < self.cfg.capture_angle < math.pi / 2:
            raise ValueError('capture_angle must be within (0, pi/2)')
        quarters = self.cfg.allowed_dock_quarters
        if not quarters or any(type(i) is not int or i not in range(4) for i in quarters) or len(set(quarters)) != len(quarters):
            raise ValueError('allowed_dock_quarters must be unique integers from 0 to 3')

    def enabled(self):
        return self.env.sim.data.eq_active[:, self.eq_ids]

    def geometry(self):
        data = self.robot.data
        a, b = self.sites
        q0 = data.site_quat_w[:, a]
        q1 = data.site_quat_w[:, b]
        candidates = data.site_quat_w[:, self.dock_sites]
        q0s = q0[:, None].expand(-1, 4, -1)
        relative = quat_mul(quat_conjugate(q0s), candidates)
        angles = 2 * torch.atan2(relative[..., 1:].norm(dim=-1), relative[..., 0].abs())
        angles = angles.masked_fill(~self.allowed[None], math.inf)
        _, quarter = angles.min(dim=-1)
        enabled = self.enabled()
        quarter = torch.where(enabled.any(-1), enabled.long().argmax(-1), quarter)
        angle = angles.gather(1, quarter[:, None]).squeeze(-1)
        rows = torch.arange(self.env.num_envs, device=self.env.device)
        relative = relative[rows, quarter]
        relative = torch.where(relative[:, :1] < 0, -relative, relative)
        qbest = candidates[rows, quarter]
        delta = data.site_pos_w[:, b] - data.site_pos_w[:, a]
        dv = data.site_lin_vel_w[:, b] - data.site_lin_vel_w[:, a]
        dw = data.site_ang_vel_w[:, b] - data.site_ang_vel_w[:, a]
        return {'p0': data.site_pos_w[:, a], 'p1': data.site_pos_w[:, b],
                'q0': q0, 'q1': q1, 'qbest': qbest, 'relative_quat': relative,
                'delta': delta, 'distance': delta.norm(dim=-1), 'angle': angle, 'quarter': quarter,
                'dv': dv, 'dw': dw, 'speed': dv.norm(dim=-1), 'angular_speed': dw.norm(dim=-1)}

    def begin_control_step(self):
        self.just_succeeded.zero_()

    def reaction(self, enabled):
        data = self.env.sim.data
        active = enabled.any(-1)
        selected = self.eq_ids[enabled.long().argmax(-1)]
        efc = data.efc
        rows = torch.arange(efc.force.shape[1], device=self.env.device)[None]
        mask = (rows < data.nefc[:, None]) & (efc.type[:] == 0) & (efc.id[:] == selected[:, None]) & active[:, None]
        rank = mask.long().cumsum(-1)
        self.force.copy_(torch.where(mask & (rank <= 3), efc.force[:].square(), 0.).sum(-1).sqrt())
        self.torque.copy_(torch.where(mask & (rank > 3), efc.force[:].square(), 0.).sum(-1).sqrt() * (0.5 * TORQUE_SCALE))

    def substep(self):
        cfg = self.cfg
        dt = self.env.physics_dt
        g = self.geometry()
        enabled = self.enabled()
        active = enabled.any(-1)
        self.reaction(enabled)
        broken = active & ((self.force > cfg.break_force) | (self.torque > cfg.break_torque)
                           | (g['distance'] > cfg.release_distance))
        self.breaks += broken.float()
        self.cooldown.copy_(torch.where(broken, cfg.recapture_cooldown_s, (self.cooldown - dt).clamp_min(0)))
        eligible = (~active & (self.cooldown <= 0) & (g['distance'] <= cfg.capture_distance)
                    & (g['angle'] <= cfg.capture_angle) & (g['speed'] <= cfg.capture_speed)
                    & (g['angular_speed'] <= cfg.capture_angular_speed))
        self.capture_time.copy_(torch.where(eligible, self.capture_time + dt, 0.))
        captured = eligible & (self.capture_time >= cfg.capture_hold_s - 1e-7)
        choice = torch.nn.functional.one_hot(g['quarter'], 4).bool()
        next_enabled = (enabled & ~broken[:, None]) | (captured[:, None] & choice)
        # Only the constraint enable bits change; all physical state remains continuous.
        self.env.sim.data.eq_active[:, self.eq_ids] = next_enabled
        self.connections += captured.float()
        settled = (active & ~broken & (g['distance'] <= cfg.capture_distance * 2)
                   & (g['angle'] <= cfg.capture_angle * 2))
        self.hold_time.copy_(torch.where(settled, self.hold_time + dt, 0.))
        achieved = self.hold_time >= cfg.success_hold_s - 1e-7
        self.just_succeeded |= achieved & ~self.succeeded
        self.succeeded |= achieved
        self.apply_attraction(g, next_enabled.any(-1))

    def apply_attraction(self, g, active):
        cfg = self.cfg
        distance = g['distance']
        qerror = quat_mul(g['q0'], quat_conjugate(g['qbest']))
        qerror = torch.where(qerror[:, :1] < 0, -qerror, qerror)
        # Site-local +Z directions have to match; do not attract through back faces.
        z = g['delta'].new_tensor((0., 0., 1.)).expand_as(g['delta'])
        z0, z1 = quat_apply(g['q0'], z), quat_apply(g['q1'], z)
        facing = ((z0 * z1).sum(-1).clamp(0., 1.)
                  * ((g['delta'] * z0).sum(-1) <= cfg.capture_distance)
                  * ((g['delta'] * z1).sum(-1) <= cfg.capture_distance))
        strength = (1 - distance / cfg.magnetic_range).clamp(0., 1.).square() * facing * (~active)
        direction = g['delta'] / distance[:, None].clamp_min(1e-6)
        # Radial damping preserves total angular momentum of the isolated pair.
        radial_speed = (g['dv'] * direction).sum(-1)
        magnitude = (cfg.magnetic_force + cfg.magnetic_linear_damping * radial_speed).clamp(0., cfg.magnetic_force)
        force = direction * (magnitude * strength)[:, None]
        tail_torque = (2 * cfg.magnetic_angular_stiffness * qerror[:, 1:]
                       - cfg.magnetic_angular_damping * g['dw']) * strength[:, None]
        com = self.robot.data.body_com_pos_w[:, self.bodies]
        wrench = torch.zeros_like(self.previous_wrench)
        wrench[:, 0, :3], wrench[:, 1, :3] = force, -force
        wrench[:, 0, 3:] = torch.cross(g['p0'] - com[:, 0], force, dim=-1) - tail_torque
        wrench[:, 1, 3:] = torch.cross(g['p1'] - com[:, 1], -force, dim=-1) + tail_torque
        field = self.env.sim.data.xfrc_applied
        field[:, self.body_ids] = field[:, self.body_ids] - self.previous_wrench + wrench
        self.previous_wrench.copy_(wrench)

    def reset(self, ids):
        if ids is None:
            ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.sim.data.eq_active[ids[:, None], self.eq_ids[None]] = False
        self.env.sim.data.xfrc_applied[ids[:, None], self.body_ids[None]] = 0.
        for name in ('capture_time', 'hold_time', 'cooldown', 'connections', 'breaks', 'force', 'torque',
                     'succeeded', 'just_succeeded', 'previous_wrench'):
            getattr(self, name)[ids] = 0

    def observation_geometry(self):
        g = self.geometry()
        return g, quat_apply_inverse(g['q0'], g['delta'])
