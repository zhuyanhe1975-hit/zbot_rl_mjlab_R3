"""Bounded, rollout-level tracking-priority controller for a separate snake task."""
import math
from copy import deepcopy
from typing import Any

import torch

from .omni_teacher import _full_resume
from .site_snake_teacher import SiteSnakeTeacherRunner
from .tasks.universal.observation_layout import (
    SITE_SNAKE_OBS_DIM,
    teacher_motion_fields,
)

AXES = ('vx', 'vy', 'wz')
VELOCITY_REWARDS = ('track_forward_velocity', 'track_lateral_velocity', 'track_angular_velocity')
STATE_KEY = 'adaptive_velocity_weights'


class VelocityWeightBalancer:
    """Preserve the total budget, using active-axis unweighted competency only."""

    state_version = 1

    def __init__(self, budget: float, device: str | torch.device = 'cpu'):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError('velocity weight budget must be positive and finite')
        self.budget = float(budget)
        self.weights = torch.full((3,), budget/3, device=device)
        self.competency = torch.full_like(self.weights, .5)
        self.seen = torch.zeros(3, dtype=torch.bool, device=device)
        self.updates = 0

    @staticmethod
    def measure(observations: torch.Tensor, limits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.shape[-1] != SITE_SNAKE_OBS_DIM:
            raise ValueError('velocity balancer requires raw 61D site snake observations')
        target, actual, _ = teacher_motion_fields(observations)
        target, actual = target.reshape(-1, 3), actual.reshape(-1, 3)
        active = ((target.abs() >= .05*limits) & torch.isfinite(target) & torch.isfinite(actual))
        # Exclude zero/tiny requests and uncommanded drift; do not divide weighted rewards.
        error = (target-actual).abs()/target.abs().clamp_min(1e-6)
        score = (1.-2.*error).clamp(-2., 1.)
        count = active.sum(0)
        mean = torch.where(active, score, 0.).sum(0)/count.clamp_min(1)
        return mean, count

    @torch.no_grad()
    def advance(self, measured: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        score, count = measured
        eligible = count >= 128
        competency = (score+2.)/3.
        ema = torch.where(self.seen, self.competency.lerp(competency, .05), competency)
        self.competency.copy_(torch.where(eligible, ema, self.competency))
        self.seen |= eligible
        used = self.weights.clone()
        if self._ready_to_update(eligible):
            # Blend equal shares with a difficulty softmax: each weight remains in
            # [0.5, 2] times the budget/3, including throughout slow weight updates.
            self.weights.lerp_(self._target_weights(), .05)
        self.updates += 1
        values = torch.stack((used, self.weights, score, self.competency*3.-2., count)).cpu().tolist()
        return {f'velocity_{name}_{axis}': float(value)
                for name, row in zip(('weight_used', 'weight_next', 'score', 'score_ema', 'samples'), values, strict=True)
                for axis, value in zip(AXES, row, strict=True)}

    def _target_weights(self):
        return self.budget*(.5/3.+.5*torch.softmax(-2.*self.competency, dim=0))

    def _ready_to_update(self, eligible):
        return self.seen.all() and eligible.any()

    def _weights_within_bounds(self, weights, tolerance):
        return ((weights >= self.budget/6.-tolerance) & (weights <= 2*self.budget/3.+tolerance)).all()

    def state_dict(self) -> dict:
        return {'version': self.state_version, 'budget': self.budget, 'weights': self.weights.cpu().tolist(),
                'competency': self.competency.cpu().tolist(), 'seen': self.seen.cpu().tolist(),
                'updates': self.updates}

    @classmethod
    def from_state(cls, state: Any, device: str | torch.device = 'cpu'):
        if not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] != cls.state_version:
            raise ValueError('checkpoint lacks matching adaptive velocity state')
        budget = state.get('budget')
        if not isinstance(budget, (float, int)) or isinstance(budget, bool):
            raise TypeError('invalid adaptive velocity budget')
        budget = float(budget)
        result = cls(budget, device)
        tolerance = budget*1e-6
        for name in ('weights', 'competency'):
            values = state.get(name)
            if (not isinstance(values, list) or len(values) != 3
                    or any(type(v) not in (int, float) for v in values)):
                raise ValueError('invalid adaptive velocity state')
        try:
            weights = result.weights.new_tensor(state['weights'])
            competency = result.competency.new_tensor(state['competency'])
            seen = state['seen']
            updates = state['updates']
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('invalid adaptive velocity state') from error
        if (weights.shape != (3,) or competency.shape != (3,)
                or not torch.isfinite(weights).all() or not torch.isfinite(competency).all()
                or not ((competency >= 0) & (competency <= 1)).all()
                or not (weights > 0).all()
                or not result._weights_within_bounds(weights, tolerance)
                or not math.isclose(weights.sum().item(), budget, rel_tol=1e-5)
                or not isinstance(seen, list) or len(seen) != 3 or any(type(v) is not bool for v in seen)
                or type(updates) is not int or updates < 0):
            raise ValueError('invalid adaptive velocity state')
        result.weights.copy_(weights)
        result.competency.copy_(competency)
        result.seen.copy_(torch.tensor(seen, device=device))
        result.updates = updates
        return result


class AdaptiveSiteSnakeTeacherRunner(SiteSnakeTeacherRunner):
    velocity_balancer_cls = VelocityWeightBalancer

    @staticmethod
    def _metadata_schema():
        return SiteSnakeTeacherRunner._metadata_schema() | {
            'role': 'adaptive_site_snake_teacher',
            'reward_recipe': 'site_snake_bounded_competency_weights_v1',
            'velocity_score_recipe': SiteSnakeTeacherRunner._metadata_schema()['reward_recipe'],
            'velocity_weight_axes': AXES,
        }

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint):
        AdaptiveSiteSnakeTeacherRunner._prepare_playback_for(env_cfg, checkpoint, AdaptiveSiteSnakeTeacherRunner)

    @staticmethod
    def _prepare_playback_for(env_cfg, checkpoint, runner_cls):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        metadata = runner_cls._validate_playback_checkpoint(saved)
        balancer = runner_cls._load_velocity_balancer(saved, playback=True)
        if not getattr(env_cfg, 'site_snake_omni', False):
            raise ValueError('adaptive checkpoint requires a site snake environment')
        command = env_cfg.commands['transform']
        command.mode, command.task_observation_encoding = 'snake', 'scalar'
        command.velocity_limits = metadata['velocity_limits']
        env_cfg.motion_site = metadata['motion_site']
        SiteSnakeTeacherRunner._restore_action_cfg(env_cfg, metadata)
        for name, weight in zip(VELOCITY_REWARDS, balancer.weights.tolist(), strict=True):
            env_cfg.rewards[name].weight = weight
        return metadata

    @classmethod
    def _load_velocity_balancer(cls, saved, device: str | torch.device = 'cpu', playback=False):
        return cls.velocity_balancer_cls.from_state((saved.get('infos') or {}).get(STATE_KEY), device)

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self._deferred_velocity_saves: list[tuple[Any, Any]] | None = None
        super().__init__(env, train_cfg, log_dir, device)
        physical = env.unwrapped
        weights = [physical.reward_manager.get_term_cfg(name).weight for name in VELOCITY_REWARDS]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError('adaptive velocity base weights must be finite and nonnegative')
        self.velocity_balancer = self.velocity_balancer_cls(sum(weights), device)
        if self._training:
            self._apply_velocity_weights()
            update = self.alg.update

            def update_and_balance():
                limits = self.velocity_balancer.weights.new_tensor(
                    physical.cfg.commands['transform'].velocity_limits)
                # Reduce before PPO clears storage; no pointers into mutable storage survive.
                measured = self._measure_velocity(self.alg.storage.observations['actor'], limits)
                self._deferred_velocity_saves = []
                try:
                    losses = dict(update())
                    losses.update(self._advance_velocity_weights(measured))
                    self._apply_velocity_weights()  # New weights start with the next rollout.
                finally:
                    pending = self._deferred_velocity_saves
                    self._deferred_velocity_saves = None
                assert pending is not None
                for path, infos in pending:
                    self.save(path, infos)
                return losses

            self.alg.update = update_and_balance

    def _measure_velocity(self, observations, limits):
        return self.velocity_balancer.measure(observations, limits)

    def _advance_velocity_weights(self, measured):
        return self.velocity_balancer.advance(measured)

    def _apply_velocity_weights(self):
        env = self.env.unwrapped
        for name, weight in zip(VELOCITY_REWARDS, self.velocity_balancer.weights.cpu().tolist(), strict=True):
            # RewardManager deep-copies its config: update both the live term and the saved config.
            env.reward_manager.get_term_cfg(name).weight = weight
            env.cfg.rewards[name].weight = weight

    def save(self, path, infos=None):
        if self._deferred_velocity_saves is not None:
            self._deferred_velocity_saves.append((path, deepcopy(infos)))
            return None
        return super().save(path, {**(infos or {}), STATE_KEY: self.velocity_balancer.state_dict()})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validator = self._validate_checkpoint if self._training else self._validate_playback_checkpoint
        validator(saved)
        restored = self._load_velocity_balancer(saved, self.env.device, playback=not self._training)
        infos = super().load(path, load_cfg, strict, map_location)
        if _full_resume(load_cfg) or not self._training:
            self.velocity_balancer = restored
            self._apply_velocity_weights()
        return infos
