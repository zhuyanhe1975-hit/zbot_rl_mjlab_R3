"""Batched PPO motion residual over independent guarded mixed-clock FSMs."""
from dataclasses import dataclass, fields
import math

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.envs import mdp as common_mdp
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul, quat_conjugate

from .mixed_closed_dynamic import mixed_dynamic_env_cfg
from .sequential_rz270 import SequentialRz270EnvCfg
from .state_machine import Phase
from .disconnect.state import state, observations as base_observations, safety_penalty, failure
from .disconnect.env_cfg import torque_cost

MODE = 'mixed_closed_batched_gpu_contact_filter_guided_motion_residual_v4'
MOVING = (Phase.SEPARATE, Phase.ROTATE, Phase.APPROACH)


def controller(env):
    return env.action_manager.get_term('joint_torque').distance_controller


def moving_mask(env):
    c = controller(env)
    return ((c.phase == Phase.SEPARATE) | (c.phase == Phase.ROTATE) | (c.phase == Phase.APPROACH)) & ~failed(env)


@dataclass
class MixedPpoEnvCfg(SequentialRz270EnvCfg):
    mixed_reconfiguration_ppo: bool = True
    motion_residual_speed: float = 2.



def failed(env):
    s = state(env)
    return failure(env) | s.contact_violation | (controller(env).phase == Phase.FAILED)


def completed(env):
    c = controller(env)
    return (c.phase == Phase.DONE) & (c.index == 4) & ~failed(env)


def progress(env):
    c = controller(env); s = state(env); g = s.geometry()
    phase = c.phase
    gap = g['distance']
    separate = (gap / .05).clamp(0., 1.)
    rotate = 1. + (1. - c.angle_error / (math.pi / 2)).clamp(0., 1.)
    approach = 2. + (1. - gap / .055).clamp(0., 1.)
    settle = 3. + (c.phase_time / (env.cfg.settle_s + env.cfg.weld_ramp_s)).clamp(0., 1.)
    local = torch.where(phase == Phase.SEPARATE, separate,
             torch.where(phase == Phase.ROTATE, rotate,
             torch.where(phase == Phase.APPROACH, approach,
             torch.where(phase == Phase.SETTLE, settle, 0.))))
    # Terminal failure is handled by failure_penalty rather than pretending
    # that the failed phase completed. progress_reward masks this transition.
    local = torch.where(phase == Phase.FAILED, torch.zeros_like(local), local)
    return (4. * c.index + local).clamp(0., 16.) / 16.


def progress_reward(env):
    a = env.action_manager.get_term('joint_torque')
    delta = progress(env) - a.progress_before
    return torch.where(failed(env), 0., delta) / env.step_dt


def success_reward(env):
    return completed(env).float() / env.step_dt


def failure_penalty(env):
    return failed(env).float() / env.step_dt


def time_penalty(env):
    return torch.ones(env.num_envs, device=env.device)


def observation(env):
    s = state(env); c = controller(env); a = env.action_manager.get_term('joint_torque'); g = s.geometry()
    other = (s.target_indices() + 1) % 6
    dst = torch.stack([m.data.site_quat_w[:,site] for m,site in zip(s.modules,s.in_sites)],1)[s.worlds,other]
    relative_q = quat_mul(quat_conjugate(dst), g['q0'])
    error = quat_mul(c.goal_q, quat_conjugate(relative_q))
    error = torch.where(error[:, :1] < 0., -error, error)
    reference_q = s.modules[0].data.root_link_quat_w
    reference_p = s.modules[0].data.root_link_pos_w
    root_positions = torch.cat([quat_apply_inverse(reference_q, m.data.root_link_pos_w-reference_p) for m in s.modules], -1)
    velocities = torch.cat([torch.cat((quat_apply_inverse(reference_q, m.data.root_link_lin_vel_w),
                                     quat_apply_inverse(reference_q, m.data.root_link_ang_vel_w)), -1) for m in s.modules], -1)
    z = reference_p.new_tensor([0., 0., 1.]).expand_as(reference_p)
    gravity = torch.cat([quat_apply_inverse(m.data.root_link_quat_w, z) for m in s.modules], -1)
    phase = torch.nn.functional.one_hot(c.phase, len(Phase)).float()
    target = torch.nn.functional.one_hot(s.target_indices(), 6).float()
    return torch.cat((base_observations(env), phase, target, quat_apply_inverse(dst, g['delta'])/.1,
                      error, quat_apply_inverse(dst, g['dv']), quat_apply_inverse(dst, g['dw']),
                      root_positions, velocities, gravity, a.nominal_velocity/6., c.command/6., a.motion_action,
                      torch.atan2(torch.sin(a.guide_position-torch.cat([m.data.joint_pos for m in s.modules], -1)),
                                  torch.cos(a.guide_position-torch.cat([m.data.joint_pos for m in s.modules], -1)))/math.pi,
                      env.sim.data.eq_active[:, s.variant_ids].float(),
                      (c.phase_time/env.cfg.motion_duration_s).clamp_max(40.)[:, None],
                      torch.as_tensor(c.index,device=env.device).expand(env.num_envs)[:,None]/4.), -1)


def residual_speed(env):
    a = env.action_manager.get_term('joint_torque')
    return (a.motion_action * env.cfg.motion_residual_speed).abs().amax(-1)


def diagnostic(env, key):
    s=state(env);a=env.action_manager.get_term('joint_torque');c=controller(env)
    if key=='module_depth':return s.max_module_depth
    if key=='ground_depth':return s.max_ground_depth
    if key=='loading_feasible':return (a.loading.feasible&(c.phase==Phase.LOAD)).float()
    if key=='failure_code':
        return torch.where(s.contact_violation,4,torch.where(s.safety_margin_violation|s.wrong_break,1,
                           torch.where(a.torque_failure,6,c.failure_code)))
    raise ValueError(key)


def motion_ppo_env_cfg(play=False):
    base = mixed_dynamic_env_cfg()
    cfg = MixedPpoEnvCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    from .batched_action import BatchedMotionActionCfg
    from .batched_state import add_batched_connectors
    cfg.actions = {'joint_torque': BatchedMotionActionCfg(entity_name='module_0')}
    cfg.scene.spec_fn = add_batched_connectors
    cfg.scene.num_envs = 1 if play else 4096
    cfg.auto_reset = not play
    cfg.episode_length_s = 120.
    cfg.observations = {name: ObservationGroupCfg({'reconfiguration': ObservationTermCfg(func=observation)},
                           concatenate_terms=True, enable_corruption=False) for name in ('actor', 'critic')}
    cfg.rewards = {'reconfiguration_progress': RewardTermCfg(func=progress_reward, weight=4.),
                   'success': RewardTermCfg(func=success_reward, weight=10.),
                   'penalty_safety_margin': RewardTermCfg(func=safety_penalty, weight=-2.),
                   'penalty_failure': RewardTermCfg(func=failure_penalty, weight=-10.),
                   'penalty_torque': RewardTermCfg(func=torque_cost, weight=-.01),
                   'penalty_time': RewardTermCfg(func=time_penalty, weight=-.02)}
    cfg.terminations = {'failure': TerminationTermCfg(func=failed),
                        'success': TerminationTermCfg(func=completed),
                        'time_out': TerminationTermCfg(func=common_mdp.time_out, time_out=True)}
    cfg.metrics['success'] = MetricsTermCfg(func=completed, reduce='max')
    cfg.metrics['reconfiguration_progress'] = MetricsTermCfg(func=progress)
    cfg.metrics['policy_motion_active'] = MetricsTermCfg(func=moving_mask)
    cfg.metrics['policy_residual_speed_rad_s'] = MetricsTermCfg(func=residual_speed)
    for name,key in [('module_penetration_max_m','module_depth'),('ground_penetration_max_m','ground_depth'),
                     ('gpu_loading_feasible','loading_feasible'),('physics_failure_code','failure_code')]:
        cfg.metrics[name]=MetricsTermCfg(func=diagnostic,params={'key':key})
    return cfg
