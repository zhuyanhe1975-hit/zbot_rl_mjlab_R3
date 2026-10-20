"""Site-frame omnidirectional locomotion, isolated from archived teacher semantics."""
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from ..common import constant
from ..transformation import mdp as transform
from ..transformation.roll_reset import collision_radial_envelope
from . import mdp
from .deployment import DeploymentCommand, DeploymentCommandCfg


@dataclass(kw_only=True)
class OmniCommandCfg(DeploymentCommandCfg):
    """Both modes sample stop, six signed axes, and mixed commands from PPO step zero."""

    def build(self, env):
        return OmniCommand(self, env)


class OmniCommand(DeploymentCommand):
    def _sample_velocity(self, ids):
        ids = ids[~self.manual_control[ids]]
        competence = self.tracking_ema[self.standing_mode[ids].long()]
        scale = .2 + .8 * ((competence - .6) / .25).clamp(0, 1)
        limits = constant(self.requested_velocity, self.cfg.velocity_limits)
        # Eight equally likely categories: stop, each signed axis, and a mixed vector.
        category = torch.randint(8, (len(ids),), device=self.device)
        axis = ((category - 1).clamp_min(0) // 2).clamp_max(2)
        sign = torch.where(category.remainder(2) == 1, 1., -1.)
        amplitude = .4 + .6 * torch.rand(len(ids), 3, device=self.device)
        single = torch.nn.functional.one_hot(axis, 3) * sign[:, None] * amplitude
        mixed = amplitude * torch.where(torch.rand(len(ids), 3, device=self.device) < .5, -1., 1.)
        sampled = torch.where((category == 7)[:, None], mixed, single)
        sampled = torch.where((category == 0)[:, None], 0., sampled)
        self.requested_velocity[ids] = sampled * limits * scale[:, None]


def motion_site_index(env):
    """Resolve the asset index once; all per-step frame math stays on the device."""
    if not hasattr(env, '_omni_motion_site_index'):
        env._omni_motion_site_index = env.scene['robot'].find_sites('foot_0_sole')[0][0]
    return env._omni_motion_site_index


def _frame_step(quat, standing, previous, initialized, initial):
    """Pure tensor math shared by eager execution and the packed graph adapter."""
    x = quat_apply(quat, constant(quat, (1., 0., 0.)).expand(quat.shape[0], -1))[:, :2]
    minus_z = quat_apply(quat, constant(quat, (0., 0., -1.)).expand(quat.shape[0], -1))[:, :2]
    projected = torch.where(standing[:, None], x, minus_z)
    other = torch.where(standing[:, None], minus_z, x)
    norm = projected.norm(dim=-1, keepdim=True)
    other_norm = other.norm(dim=-1, keepdim=True)
    valid = torch.isfinite(projected).all(-1) & (norm[:, 0] > .15)
    fallback = ~initialized & ~valid & torch.isfinite(other).all(-1) & (other_norm[:, 0] > .15)
    normalized = torch.nan_to_num(projected / norm.clamp_min(1e-6))
    alternate = torch.nan_to_num(other / other_norm.clamp_min(1e-6))
    next_heading = torch.where(valid[:, None], normalized, torch.where(fallback[:, None], alternate, previous))
    first = ~initialized & (valid | fallback)
    next_initial = torch.where(first[:, None], next_heading, initial)
    next_initialized = initialized | valid | fallback
    return next_heading, next_initial, next_initialized


def packed_frame_step(packed):
    """Pure N×10 -> N×5 adapter for fixed-shape CUDA capture.

    Input: quaternion4, standing1, previous heading2, initialized1, initial2.
    Output: next heading2, next initial2, next initialized1 (same float dtype).
    """
    heading, initial, initialized = _frame_step(
        packed[:, :4], packed[:, 4] > 0, packed[:, 5:7],
        packed[:, 7] > 0, packed[:, 8:10],
    )
    return torch.cat((heading, initial, initialized[:, None].to(packed.dtype)), -1)


def site_heading(env):
    """Site +X (biped) or -Z (snake), projected to the gravity-aligned plane.

    Retain the previous heading for a vertical primary axis; first use falls
    back to the other site axis. The optional graph captures only pure math.
    """
    term = mdp.command_term(env)
    backend = getattr(getattr(env, 'cfg', None), 'omni_frame_backend', 'eager')
    if backend not in ('eager', 'cuda_graph'):
        raise ValueError("omni_frame_backend must be 'eager' or 'cuda_graph'")
    quat = env.scene['robot'].data.site_quat_w[:, motion_site_index(env)]
    if backend == 'cuda_graph':
        from ...cuda_graph import FixedTensorGraph
        packed = torch.cat((quat, term.command[:, :1], term.heading,
                            term.heading_initialized[:, None].to(quat.dtype), term.initial_heading), -1)
        if not hasattr(env, '_omni_frame_graph'):
            env._omni_frame_graph = FixedTensorGraph(packed_frame_step, packed)
        result = env._omni_frame_graph(packed)
        heading, initial, initialized = result[:, :2], result[:, 2:4], result[:, 4] > 0
    else:
        heading, initial, initialized = _frame_step(
            quat, term.standing_mode, term.heading, term.heading_initialized, term.initial_heading,
        )
    # State writes remain explicit, outside capture. Preserve the public alias.
    term.heading.copy_(heading)
    term.initial_heading.copy_(initial)
    term.heading_initialized.copy_(initialized)
    return term.heading


def _project_velocity(env, linear):
    forward = site_heading(env)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    yaw = env.scene['robot'].data.site_ang_vel_w[:, motion_site_index(env), 2]
    return torch.stack(((linear[..., :2] * forward).sum(-1), (linear[..., :2] * lateral).sum(-1), yaw), -1)


def tracking_velocity(env):
    """COM translation prevents head-wiggling rewards; yaw is site angular velocity."""
    data = env.scene['robot'].data
    masses = transform.body_masses(env)
    linear = (data.body_com_lin_vel_w * masses[..., None]).sum(-2)
    linear /= masses.sum(-1, keepdim=True).clamp_min(1e-6)
    return _project_velocity(env, linear)


def site_velocity(env):
    """Actual site velocity for observation/diagnostics, distinct from COM tracking."""
    return _project_velocity(env, env.scene['robot'].data.site_lin_vel_w[:, motion_site_index(env)])


def reset_omni(env, env_ids):
    """Random-yaw canonical biped/snake starts, only at explicit environment resets."""
    term = mdp.command_term(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not len(ids):
        return
    training = term.cfg.training
    robot = env.scene['robot']
    root = robot.data.default_root_state[ids].clone()
    random_play = not training and getattr(term.cfg, 'randomize_play_modes', False)
    if random_play and term.cfg.mode == 'source':
        choose_stand = (torch.randperm(len(ids), device=env.device) < (len(ids) + 1) // 2
                        if len(ids) > 1 else torch.rand(1, device=env.device) < .5)
    elif random_play:
        choose_stand = torch.full((len(ids),), term.cfg.mode == 'stand', dtype=torch.bool, device=env.device)
    else:
        choose_stand = torch.rand(len(ids), device=env.device) < .5 if training else torch.ones(len(ids), dtype=torch.bool, device=env.device)
    role_sampler = getattr(term, 'training_roles', None)
    if training and role_sampler is not None:
        if term.cfg.mode != 'source':
            choose_stand = torch.full((len(ids),), term.cfg.mode == 'stand', device=env.device, dtype=torch.bool)
        else:
            roles = role_sampler(ids)
            # Dedicated motion starts match their target; transformations start opposite.
            choose_stand = torch.where(roles >= 0, (roles == 0) | (roles == 3), choose_stand)
    if random_play:
        if not hasattr(env, '_omni_play_standing'):
            env._omni_play_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._omni_play_standing[ids] = choose_stand
    if not hasattr(env, '_omni_standing_root_height'):
        env._omni_standing_root_height = .003 - collision_radial_envelope(tuple(env.cfg.standing_joint_pos))[1]
    offset = root.new_zeros(len(ids), 3)
    offset[:, 2] = torch.where(choose_stand, env._omni_standing_root_height, .0555)
    offset[:, 1] = torch.where(choose_stand, -.06, 0.)
    canonical = root.new_tensor([2**-.5, 0., -(2**-.5), 0.]).expand(len(ids), -1)
    canonical = torch.where(choose_stand[:, None], root.new_tensor([1., 0., 0., 0.]), canonical)
    yaw = (torch.rand(len(ids), device=env.device) * 2 - 1) * torch.pi if training else root.new_zeros(len(ids))
    yaw_quat = torch.stack(((yaw / 2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin()), -1)
    root[:, :3] = env.scene.env_origins[ids] + quat_apply(yaw_quat, offset)
    root[:, 3:7] = quat_mul(yaw_quat, canonical)
    root[:, 7:] = 0
    q = root.new_tensor(env.cfg.standing_joint_pos).expand(len(ids), -1)
    q = torch.where(choose_stand[:, None], q, 0.)
    if getattr(term.cfg, 'initialize_matched_motion', False):
        if not hasattr(env, '_omni_initial_standing'):
            env._omni_initial_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._omni_initial_standing[ids] = choose_stand
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)
    term.heading_initialized[ids] = False
