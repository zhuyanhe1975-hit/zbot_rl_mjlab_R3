"""Physical, single-robot counterpart to the mixed closed-loop kinematic experiment."""
import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from .disconnect.state import failure, state
from .sequential_rz270 import (SequentialAction, SequentialActionCfg, SequentialConnectorState,
                               SequentialController, add_sequential_connectors, sequential_rz270_env_cfg)
from .state_machine import Phase, ReconfigurationController
from .live_loading import LiveLoadingController

EXPERIMENT_ID = 'Zbot-Closed180-To-270-270-180-90-90-Dynamics-v1'
ORDER = (0, 1, 3, 4)


def add_mixed_connectors(spec):
    add_sequential_connectors(spec)
    for i in (3, 4):
        original = spec.site(f'module_{i}/port_out')
        variant = spec.site(f'module_{i}/port_out_rz270')
        turn = np.array([math.cos(math.pi / 4), 0., 0., -math.sin(math.pi / 4)])
        q = np.empty(4); mujoco.mju_mulQuat(q, original.quat, turn)
        rotation = np.empty(9); mujoco.mju_quat2Mat(rotation, original.quat)
        inlet = spec.site(f'module_{i+1}/port_in').pos
        rz = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        variant.quat[:] = q
        variant.pos[:] = original.pos + rotation.reshape(3, 3) @ (rz @ inlet - inlet)
    for i in range(6):
        spec.joint(f'module_{i}/joint{i+1}').limited = mujoco.mjtLimited.mjLIMITED_FALSE


class MixedState(SequentialConnectorState):
    def __init__(self, env):
        super().__init__(env)
        m = env.sim.mj_model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or '' for b in m.geom_bodyid]
        self.geom_module = torch.tensor([int(n.split('/')[0][7:]) if n.startswith('module_') else -1
                                         for n in names], device=env.device)
        self.cpu_data = mujoco.MjData(m)
        self.excluded_pairs = [(m.geom(f'module_{i}/b{i+1}_geom').id,
                                m.geom(f'module_{i+1}/a{i+2}_geom').id) for i in range(5)]
        self.contact_violation = torch.zeros(1, dtype=torch.bool, device=env.device)
        self.max_module_depth = torch.zeros(1, device=env.device)
        self.max_ground_depth = torch.zeros(1, device=env.device)
        self.max_other_load = torch.zeros(1, device=env.device)
        self.loads_at_break = torch.zeros((1, 6), device=env.device)
        self.max_excluded_depth = 0.
        self.worst_pair = torch.full((2,), -1, dtype=torch.long, device=env.device)

    def reset(self, ids):
        super().reset(ids)
        self.contact_violation.zero_(); self.max_module_depth.zero_(); self.max_ground_depth.zero_(); self.max_other_load.zero_(); self.loads_at_break.zero_()
        self.max_excluded_depth = 0.; self.worst_pair.fill_(-1)
        self.last_substep = None

    def update(self):
        if self.last_substep == self.env._sim_step_counter:
            return
        was_connected = self.connected()[:, self.cfg.target_connector].clone()
        super().update()
        broke = was_connected & ~self.connected()[:, self.cfg.target_connector]
        self.loads_at_break.copy_(torch.where(broke[:, None], self.utilization * self.cfg.break_force, self.loads_at_break))
        self.max_other_load.copy_(torch.maximum(self.max_other_load, self.utilization[:, self.other_ids].amax(-1) * self.cfg.break_force))
        d = self.env.sim.data
        contact = d.contact
        valid = (torch.arange(contact.dist.numel(), device=self.env.device) < d.nacon[0]) & (contact.worldid == 0)
        geom = contact.geom.long().clamp(0, self.geom_module.numel()-1)
        mods = self.geom_module[geom]
        depth = (-contact.dist).clamp_min(0.)
        robot = valid & (mods[:, 0] >= 0) & (mods[:, 1] >= 0) & (mods[:, 0] != mods[:, 1])
        ground = valid & ((mods[:, 0] >= 0) ^ (mods[:, 1] >= 0))
        robot_depth = torch.where(robot, depth, 0.)
        ground_depth = torch.where(ground, depth, 0.)
        max_depth, idx = robot_depth.max(dim=0)
        self.worst_pair.copy_(torch.where(max_depth > self.max_module_depth[0], geom[idx], self.worst_pair))
        self.max_module_depth.copy_(torch.maximum(self.max_module_depth, max_depth))
        self.max_ground_depth.copy_(torch.maximum(self.max_ground_depth, ground_depth.max()))
        # MuJoCo Warp prefilters these pairs. Query the open face independently
        # using the measured dynamic qpos, without modifying the simulated state.
        target = self.cfg.target_connector
        if target < 5 and not bool(self.connected()[0, target]):
            self.cpu_data.qpos[:] = d.qpos[0].cpu().numpy()
            mujoco.mj_kinematics(self.env.sim.mj_model, self.cpu_data)
            a, b = self.excluded_pairs[target]
            distance = mujoco.mj_geomDistance(self.env.sim.mj_model, self.cpu_data, a, b, .5, None)
            self.max_excluded_depth = max(self.max_excluded_depth, -float(distance), 0.)
        self.contact_violation |= ((self.max_module_depth > .002) | (self.max_ground_depth > .003)
                                   | (self.max_excluded_depth > .002))


class MixedController(SequentialController):
    def select(self, target):
        super().select(target)
        sign = -1. if target in (0, 1) else 1.
        self.goal_q = torch.tensor([[math.cos(math.pi/4), 0., 0., sign*math.sin(math.pi/4)]], device=self.env.device)

    def advance(self, s):
        if self.index < len(ORDER)-1:
            super().advance(s)
            return
        self.history.append({'connector': self.env.cfg.target_connector,
                             'durations_s': self.phase_durations[0].tolist(),
                             'joint_angle_rad': torch.cat([m.data.joint_pos for m in self.modules], -1)[0].tolist()})
        self.index += 1

    def update(self, s, enabled):
        if self.index >= len(ORDER):
            self.command.zero_()
            return self.command
        # The brace and unchanged middle connection must remain welded throughout.
        active = self.env.sim.data.eq_active[0]
        invariant = bool(active[s.eq_ids[2]]) and bool(active[s.eq_ids[5]]) and not bool(active[s.variant_ids[2]])
        if bool(s.contact_violation[0]) or not invariant:
            self.failure_code.fill_(4 if bool(s.contact_violation[0]) else 5)
            self.transition(torch.ones(1, dtype=torch.bool, device=self.env.device), Phase.FAILED)
            self.command.zero_(); s.waiting.fill_(True)
            return self.command
        result = ReconfigurationController.update(self, s, enabled)
        if self.phase.item() == Phase.DONE:
            if bool(failure(self.env).any()):
                self.failure_code.fill_(1); self.phase.fill_(Phase.FAILED); s.waiting.fill_(True)
            else:
                self.advance(s)
        return result


@dataclass(kw_only=True)
class MixedActionCfg(SequentialActionCfg):
    def build(self, env):
        return MixedAction(self, env)


class MixedAction(SequentialAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not self.native_pd:
            raise ValueError('Mixed dynamic experiment requires MJLab native PD')
        env._disconnect_state = MixedState(env)
        self.distance_controller = MixedController(env, self.modules)
        # The 56 N prediction limit reserves 8 N below the actual 64 N stop.
        self.loading = LiveLoadingController(env, other_limit=56., increment=.20, load_step=3.,
                                             safety_weight=.00002, search_neighbors=True, safe_fallback=True)
        self.jacobian_report['experiment'] = EXPERIMENT_ID


def mixed_dynamic_env_cfg():
    cfg = sequential_rz270_env_cfg(order=ORDER)
    cfg.scene.spec_fn = add_mixed_connectors
    cfg.actions = {'joint_torque': MixedActionCfg(entity_name='module_0')}
    cfg.experiment_id = EXPERIMENT_ID
    return cfg
