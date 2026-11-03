"""Forward head-site progress along the head and sampled world-heading targets."""

import math
from dataclasses import dataclass

import torch

from ..universal import mdp
from ..universal.omni import motion_site_index
from .site_omni import (
    SiteSnakeCommand,
    SiteSnakeCommandCfg,
    site_snake_state,
    site_tilt,
)


@dataclass(kw_only=True)
class HeadingSnakeCommandCfg(SiteSnakeCommandCfg):
    heading_range: tuple[float, float] = (-math.pi, math.pi)
    forward_speed_cap: float = .2
    forward_speed_range: tuple[float, float] = (0., .2)
    forward_stop_probability: float = .1
    play_forward_speed: float | None = None
    play_heading: float | None = None

    def __post_init__(self):
        super().__post_init__()
        low, high = self.heading_range
        if not all(math.isfinite(v) for v in (low, high)) or not -math.pi <= low <= high <= math.pi:
            raise ValueError('heading_range must be ordered within [-pi, pi]')
        if not math.isfinite(self.forward_speed_cap) or self.forward_speed_cap <= 0:
            raise ValueError('forward_speed_cap must be positive and finite')
        low, high = self.forward_speed_range
        if (not all(math.isfinite(v) for v in (low, high))
                or not 0 <= low <= high <= self.forward_speed_cap):
            raise ValueError('forward_speed_range must be ordered within [0, forward_speed_cap]')
        if not math.isfinite(self.forward_stop_probability) or not 0 <= self.forward_stop_probability <= 1:
            raise ValueError('forward_stop_probability must be in [0, 1]')
        if self.play_forward_speed is not None and (
            not math.isfinite(self.play_forward_speed) or not 0 <= self.play_forward_speed <= self.forward_speed_cap
        ):
            raise ValueError('play_forward_speed must be within [0, forward_speed_cap] or None')
        if self.play_heading is not None and not math.isfinite(self.play_heading):
            raise ValueError('play_heading must be finite or None')

    def build(self, env):
        return HeadingSnakeCommand(self, env)


class HeadingSnakeCommand(SiteSnakeCommand):
    def __init__(self, cfg, env):
        # Allocate before inherited initialization: resampling may access this buffer.
        self.target_heading = torch.zeros(env.num_envs, device=env.device)
        super().__init__(cfg, env)

    def _sample_velocity(self, ids):
        ids = ids[~self.manual_control[ids]]
        low, high = self.cfg.heading_range
        self.target_heading[ids] = low + (high - low) * torch.rand(len(ids), device=self.device)
        if not self.cfg.training and self.cfg.play_heading is not None:
            self.target_heading[ids] = self.cfg.play_heading
        # Compatibility slots describe forward progress only; no sideways/yaw sampling.
        self.requested_velocity[ids] = 0.
        low, high = self.cfg.forward_speed_range
        speed = low + (high - low) * torch.rand(len(ids), device=self.device)
        stopped = torch.rand(len(ids), device=self.device) < self.cfg.forward_stop_probability
        speed = torch.where(stopped, 0., speed)
        if not self.cfg.training and self.cfg.play_forward_speed is not None:
            speed = torch.full_like(speed, self.cfg.play_forward_speed)
        self.requested_velocity[ids, 0] = speed

    def set_heading(self, angle, env_ids=None):
        """Manually steer selected robots by world heading without resetting pose."""
        if not math.isfinite(angle):
            raise ValueError('target heading must be finite')
        ids = self._ids(env_ids)
        self.target_heading[ids] = math.atan2(math.sin(angle), math.cos(angle))
        self.requested_velocity[ids, 1:] = 0.
        self.manual_control[ids] = True
        self.time_left[ids] = math.inf

    def set_forward_speed(self, speed, env_ids=None):
        """Set only nonnegative forward speed, preserving the heading target."""
        if not math.isfinite(speed):
            raise ValueError('forward speed must be finite')
        ids = self._ids(env_ids)
        self.requested_velocity[ids] = 0.
        self.requested_velocity[ids, 0] = min(max(speed, 0.), self.cfg.forward_speed_cap)
        self.manual_control[ids] = True
        self.time_left[ids] = math.inf

    def stop(self, env_ids=None):
        self.set_forward_speed(0., env_ids)

    def _debug_vis_impl(self, visualizer):
        indices = visualizer.get_env_indices(self.num_envs)
        if self.cfg.debug_vis_selected_only and hasattr(visualizer, 'env_idx'):
            indices = [visualizer.env_idx] if visualizer.env_idx in indices else []
        if not indices:
            return
        data = self._env.scene['robot'].data
        site = motion_site_index(self._env)
        origins = data.site_pos_w[:, site] + data.site_pos_w.new_tensor((0., 0., .08))
        target = torch.stack((self.target_heading.cos(), self.target_heading.sin()), -1)
        target_vectors = torch.zeros_like(origins)
        target_lengths = .2 * (self.requested_velocity[:, 0] / self.cfg.forward_speed_cap).clamp(0., 1.)
        target_vectors[:, :2] = target_lengths[:, None] * target
        head_vectors = torch.zeros_like(origins)
        head_vectors[:, :2] = .2 * mdp.heading(self._env)
        # Direct site measurements in world axes: XY translation and signed wz only.
        linear_vectors = torch.zeros_like(origins)
        linear_vectors[:, :2] = data.site_lin_vel_w[:, site, :2] * (.2 / self.cfg.forward_speed_cap)
        yaw_vectors = torch.zeros_like(origins)
        yaw_vectors[:, 2] = .1 * data.site_ang_vel_w[:, site, 2]
        starts = origins.detach().cpu().numpy()
        for vectors, color, max_width in (
            (target_vectors, (1., .65, .1, .9), .006),
            (head_vectors, (.1, 1., .2, .9), .006),
            (linear_vectors, (.1, .8, 1., .9), .004),
            (yaw_vectors, (.8, .3, 1., .9), .004),
        ):
            lengths = vectors.norm(dim=-1)
            ends = (origins + vectors).detach().cpu().numpy()
            visible = ((lengths > 1e-6) & torch.isfinite(vectors).all(-1)).cpu().tolist()
            # Scale width with length so short arrows retain a valid arrow head.
            widths = (.06 * lengths).clamp(max=max_width).cpu().tolist()
            for i in indices:
                if visible[i]:
                    visualizer.add_arrow(starts[i], ends[i], color=color, width=widths[i])

    def _resample_command(self, env_ids):
        # Both training and playback sample absolute headings on the command timer.
        # Parent methods retain the snake-only task state and manual-control mask.
        super()._resample_command(env_ids)
        if not self.cfg.training and not self.cfg.play_random_velocity:
            self._sample_velocity(env_ids)


def heading_error(env):
    term = mdp.command_term(env)
    forward = mdp.heading(env)
    current = torch.atan2(forward[:, 1], forward[:, 0])
    delta = term.target_heading - current
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def linear_speed_score(actual, requested, stop_speed_scale=.05):
    """Relative triangular score: reverse negative, overspeed decays, stop is finite."""
    if not math.isfinite(stop_speed_scale) or stop_speed_scale <= 0:
        raise ValueError('stop_speed_scale must be positive and finite')
    # Use a separate nonzero scale for exact stop commands; tiny positive requests
    # retain relative scoring rather than silently becoming stop commands.
    scale = torch.where(requested > 0., requested, torch.full_like(requested, stop_speed_scale))
    return (1. - (actual - requested).abs() / scale).clamp(-1., 1.)


def head_forward_velocity(env):
    velocity = env.scene['robot'].data.site_lin_vel_w[:, motion_site_index(env)]
    return (velocity[:, :2] * mdp.heading(env)).sum(-1)


def forward_speed_command(env):
    return mdp.command_term(env).requested_velocity[:, 0]


def forward_speed_error(env):
    return (head_forward_velocity(env) - forward_speed_command(env)).abs()


def head_forward_speed(env, stop_speed_scale=.05):
    return linear_speed_score(head_forward_velocity(env), forward_speed_command(env), stop_speed_scale)


def heading_alignment(env):
    """Linear absolute wrapped-angle cost, sensitive to small heading errors."""
    return -heading_error(env).abs()


def heading_observation(env):
    error = heading_error(env)
    # Continuous at +/-pi and explicitly visible to both actor and critic.
    return torch.cat((site_snake_state(env), error.sin()[:, None], error.cos()[:, None]), -1)


def heading_error_abs(env):
    return heading_error(env).abs()


def target_heading(env):
    return mdp.command_term(env).target_heading


def head_tilt_angle_penalty(env):
    """Squared angle between the head-site up axis and opposite gravity (rad²)."""
    return site_tilt(env).square()


def head_lateral_speed(env):
    """Absolute head-site speed along the head's horizontal left/right axis."""
    forward = mdp.heading(env)
    left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    velocity = env.scene['robot'].data.site_lin_vel_w[:, motion_site_index(env)]
    return (velocity[:, :2] * left).sum(-1).abs()


def head_lateral_speed_penalty(env):
    return head_lateral_speed(env)
