"""Experiment20 retains adaptive PPO but uses a separate base-reference contract."""
import math
from typing import ClassVar, cast

import torch

from .adaptive_site_snake import (
    STATE_KEY,
    VELOCITY_REWARDS,
    AdaptiveSiteSnakeTeacherRunner,
    VelocityWeightBalancer,
)
from .tasks.snake.site_omni import relative_velocity_scores
from .tasks.universal.observation_layout import (
    SITE_SNAKE_OBS_DIM,
    teacher_motion_fields,
)


class RewardContributionBalancer(VelocityWeightBalancer):
    """Learn balanced planar motion before accelerating yaw learning."""

    state_version = 7

    def __init__(self, budget, device='cpu'):
        super().__init__(budget, device)
        self.weights.copy_(self.weights.new_tensor([.475*budget, .475*budget, .05*budget]))
        self.stage = 1
        self.planar_positive_updates = 0
        self.planar_regression_updates = 0
        self._latest_scores = self.weights.new_zeros(3)
        self.reward_mean_ema = torch.zeros_like(self.weights)
        self.reward_mean_initialized = False
        self._pending_reward_means = None
        self._fresh_reward_means = False
        self.yaw_target_limited = False
        self.planar_target_limited = False

    def record_reward_means(self, means):
        values = means.detach().to(self.weights)
        if values.shape != (3,) or not torch.isfinite(values).all() or ((values < -2.) | (values > 1.)).any():
            raise ValueError('reward means must be three finite values in [-2, 1]')
        self._pending_reward_means = values.clone()

    @torch.no_grad()
    def advance(self, measured):
        self._fresh_reward_means = False
        self.yaw_target_limited = False
        self.planar_target_limited = False
        self._latest_scores = measured[0]
        if self._pending_reward_means is not None and (measured[1] >= 128).any():
            if self.reward_mean_initialized:
                self.reward_mean_ema.lerp_(self._pending_reward_means, .05)
            else:
                self.reward_mean_ema.copy_(self._pending_reward_means)
            self.reward_mean_initialized = True
            self._fresh_reward_means = True
        self._pending_reward_means = None
        metrics = super().advance(measured)
        metrics.update(velocity_learning_stage=float(self.stage),
                       velocity_planar_positive_updates=float(self.planar_positive_updates),
                       velocity_planar_regression_updates=float(self.planar_regression_updates))
        return metrics

    def _ready_to_update(self, eligible):
        if not self._fresh_reward_means:
            self.planar_positive_updates = 0
            self.planar_regression_updates = 0
            return False
        if eligible[:2].all():
            active_scores = self.competency[:2]*3.-2.
            if self.stage == 1:
                stable_positive = ((active_scores > .02).all()
                                   and (self.reward_mean_ema[:2] > .02).all()
                                   and (self._latest_scores[:2] > 0.).all())
                self.planar_positive_updates = self.planar_positive_updates+1 if stable_positive else 0
                if self.planar_positive_updates >= 20:
                    self.stage = 2
                    self.planar_positive_updates = 0
            else:
                regressed = (active_scores < -.02).any() or (self.reward_mean_ema[:2] < -.02).any()
                self.planar_regression_updates = self.planar_regression_updates+1 if regressed else 0
                if self.planar_regression_updates >= 10:
                    self.stage = 1
                    self.planar_regression_updates = 0
        else:
            self.planar_positive_updates = 0
            self.planar_regression_updates = 0
        return eligible.any() and self.reward_mean_ema.abs().max() > 1e-4

    def _target_weights(self):
        planar_means = self.reward_mean_ema[:2]
        if self.stage == 1:
            # Increase the weaker planar axis's weight without equating negative returns.
            shares = .25+.5*torch.softmax(-2.*self.competency[:2], dim=0)
        elif (planar_means > 1e-4).all():
            # wx*mx = wy*my; retain the overall velocity reward budget.
            desired_x = planar_means[1]/planar_means.sum()
            x_share = desired_x.clamp(.25, .75)
            self.planar_target_limited = bool((desired_x < .25) | (desired_x > .75))
            shares = torch.stack((x_share, 1.-x_share))
        else:
            # A positive/negative pair cannot be equalized with positive weights.
            # Keep the planar split instead of rescaling penalties to fit a ratio.
            shares = self.weights[:2]/self.weights[:2].sum()
        planar = (shares*self.reward_mean_ema[:2]).sum()
        yaw = self.reward_mean_ema[2]
        if self.stage == 1:
            yaw_share = self.weights.new_tensor(.05)
        elif yaw <= .02:
            # Once planar motion is established, strengthen the yaw learning signal.
            yaw_share = self.weights.new_tensor(.4)
        elif planar <= 1e-4:
            # Suppress positive yaw return until planar return becomes positive.
            yaw_share = self.weights.new_tensor(.001)
            self.yaw_target_limited = True
        else:
            desired = planar/(4.*yaw+planar)
            yaw_share = desired.clamp(.001, .6)
            self.yaw_target_limited = bool((desired < .001) | (desired > .6))
        return self.budget*torch.cat(((1.-yaw_share)*shares, yaw_share.reshape(1)))

    def _weights_within_bounds(self, weights, tolerance):
        planar_total = weights[:2].sum()
        split_valid = ((weights[:2] >= .25*planar_total-tolerance)
                       & (weights[:2] <= .75*planar_total+tolerance)).all()
        return (((weights[:2] >= .1*self.budget-tolerance)
                 & (weights[:2] <= .75*self.budget+tolerance)).all()
                & (weights[2] >= .001*self.budget-tolerance)
                & (weights[2] <= .6*self.budget+tolerance) & split_valid)

    def state_dict(self):
        return super().state_dict() | {'yaw_to_planar_reward_target': .25, 'forward_to_lateral_reward_target': 1., 'contribution_statistic': 'signed_mean',
            'reward_mean_ema': self.reward_mean_ema.cpu().tolist(),
            'reward_mean_initialized': self.reward_mean_initialized,
            'stage': self.stage, 'planar_positive_updates': self.planar_positive_updates,
            'planar_regression_updates': self.planar_regression_updates}

    @classmethod
    def from_state(cls, state, device='cpu'):
        result = cast('RewardContributionBalancer', super().from_state(state, device))
        values = state.get('reward_mean_ema')
        if (state.get('yaw_to_planar_reward_target') != .25
                or state.get('forward_to_lateral_reward_target') != 1.
                or state.get('contribution_statistic') != 'signed_mean'
                or not isinstance(values, list) or len(values) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) or not -2 <= v <= 1 for v in values)
                or type(state.get('reward_mean_initialized')) is not bool):
            raise ValueError('checkpoint reward contribution state is invalid')
        stage = state.get('stage')
        positive = state.get('planar_positive_updates')
        regression = state.get('planar_regression_updates')
        if (type(stage) is not int or stage not in (1, 2)
                or type(positive) is not int or not 0 <= positive < 20
                or type(regression) is not int or not 0 <= regression < 10
                or (stage == 1 and regression != 0) or (stage == 2 and positive != 0)):
            raise ValueError('checkpoint velocity curriculum state is invalid')
        result.stage, result.planar_positive_updates, result.planar_regression_updates = stage, positive, regression
        result.reward_mean_ema.copy_(result.weights.new_tensor(values))
        result.reward_mean_initialized = state['reward_mean_initialized']
        return result


class BaseSnakeAdaptiveTeacherRunner(AdaptiveSiteSnakeTeacherRunner):
    velocity_balancer_cls = RewardContributionBalancer
    _playback_recipes: ClassVar[dict[str, tuple[float, int]]] = {
        'base_snake_relative_equal_planar_half_yaw_v7': (.5, 5),
        'base_snake_relative_equal_planar_quarter_yaw_v8': (.25, 6),
    }

    @staticmethod
    def _metadata_schema():
        return AdaptiveSiteSnakeTeacherRunner._metadata_schema() | {
            'role': 'adaptive_base_snake_teacher', 'schema_version': 7,
            'snake_omni_version': 13,
            'observation_recipe': 'site_base_frame_error_gravity_v3',
            'tracking_point': 'base', 'motion_site': 'base_motion',
            'command_axes': ('base_minus_z_gravity_aligned', 'left', 'world_plus_z'),
            'reward_recipe': 'base_snake_relative_planar_then_yaw_v9',
            'yaw_to_planar_reward_target': .25, 'forward_to_lateral_reward_target': 1.,
            'velocity_score_recipe': 'base_relative_clipped_minus2_plus1_v1',
            'contribution_statistic': 'signed_mean',
            'qualification_protocol': 'site_snake_base_omni_v5',
        }

    @classmethod
    def _validate_playback_checkpoint(cls, saved: dict) -> dict:
        metadata = (saved.get('infos') or {}).get('privileged_omni_teacher')
        if (isinstance(metadata, dict)
                and isinstance(metadata.get('reward_recipe'), str)
                and metadata.get('reward_recipe') in cls._playback_recipes
                and type(metadata.get('yaw_to_planar_reward_target')) is float
                and metadata['yaw_to_planar_reward_target'] == cls._playback_recipes[metadata['reward_recipe']][0]):
            # Training reward control changed; validate every inference contract field.
            compatible_metadata = metadata | {
                'reward_recipe': cls._metadata_schema()['reward_recipe'],
                'yaw_to_planar_reward_target': .25,
            }
            compatible = saved | {'infos': saved['infos'] | {'privileged_omni_teacher': compatible_metadata}}
            cls._validate_checkpoint(compatible)
            return metadata
        return cls._validate_checkpoint(saved)

    @classmethod
    def _load_velocity_balancer(cls, saved, device: str | torch.device = 'cpu', playback=False):
        if playback:
            metadata = cls._validate_playback_checkpoint(saved)
            if metadata['reward_recipe'] in cls._playback_recipes:
                target, version = cls._playback_recipes[metadata['reward_recipe']]
                state = (saved.get('infos') or {}).get(STATE_KEY)
                if (not isinstance(state, dict) or type(state.get('version')) is not int
                        or state['version'] != version or type(state.get('yaw_to_planar_reward_target')) is not float
                        or state['yaw_to_planar_reward_target'] != target):
                    raise ValueError('checkpoint lacks matching legacy base velocity state')
                # Translate the state format in memory only. Playback never advances it.
                state = state | {'version': RewardContributionBalancer.state_version,
                                 'yaw_to_planar_reward_target': .25, 'stage': 1,
                                 'planar_positive_updates': 0, 'planar_regression_updates': 0}
                return RewardContributionBalancer.from_state(state, device)
        return super()._load_velocity_balancer(saved, device, playback)

    def _schema(self):
        schema = super()._schema()
        metadata = getattr(self, '_playback_metadata', None)
        if not self._training and metadata is not None:
            # Full playback loads also compare the saved reward contract in OmniTeacherRunner.
            schema.update({key: metadata[key] for key in ('reward_recipe', 'yaw_to_planar_reward_target')})
        return schema

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint):
        AdaptiveSiteSnakeTeacherRunner._prepare_playback_for(env_cfg, checkpoint, BaseSnakeAdaptiveTeacherRunner)

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        self._velocity_reward_sum = self.velocity_balancer.weights.new_zeros(3)
        self._velocity_reward_steps = 0
        self._last_velocity_reward_rates = self.velocity_balancer.weights.new_zeros(3)
        if self._training:
            self._install_velocity_reward_capture()

    def _install_velocity_reward_capture(self):
        manager = self.env.unwrapped.reward_manager
        indices = [manager.active_terms.index(name) for name in VELOCITY_REWARDS]
        compute = manager.compute

        def compute_and_capture(dt):
            total = compute(dt)
            # Capture weighted rates before reset; weights are constant within a rollout.
            values = manager._step_reward[:, indices].detach()
            self._velocity_reward_sum.add_(values.mean(0).to(self._velocity_reward_sum))
            self._velocity_reward_steps += 1
            return total

        manager.compute = compute_and_capture

    def _measure_velocity(self, observations, limits):
        if observations.shape[-1] != SITE_SNAKE_OBS_DIM:
            raise ValueError('relative velocity weights require raw 61D observations')
        target, actual, _ = teacher_motion_fields(observations)
        target, actual = target.reshape(-1, 3), actual.reshape(-1, 3)
        active = (target.abs() >= .05*limits) & torch.isfinite(target) & torch.isfinite(actual)
        score = relative_velocity_scores(target, actual, limits)
        count = active.sum(0)
        mean = torch.where(active, score, 0.).sum(0)/count.clamp_min(1)
        if getattr(self, '_velocity_reward_steps', 0):
            rates = self._velocity_reward_sum/self._velocity_reward_steps
            reward_means = rates/self.velocity_balancer.weights
            self._velocity_reward_sum.zero_()
            self._velocity_reward_steps = 0
        else:
            if getattr(self, '_training', False):
                raise RuntimeError('training reward contribution capture has no collected steps')
            # Direct diagnostic calls without environment stepping use all finite states.
            finite = torch.isfinite(target) & torch.isfinite(actual)
            reward_means = torch.where(finite, score, 0.).sum(0)/finite.sum(0).clamp_min(1)
            rates = reward_means*self.velocity_balancer.weights
        self._last_velocity_reward_rates = rates.detach().clone()
        cast(RewardContributionBalancer, self.velocity_balancer).record_reward_means(reward_means.clamp(-2., 1.))
        return mean, count

    def _advance_velocity_weights(self, measured):
        metrics = super()._advance_velocity_weights(measured)
        rates = self._last_velocity_reward_rates
        planar = float((rates[0]+rates[1]).item())
        yaw = float(rates[2].item())
        forward, lateral = float(rates[0].item()), float(rates[1].item())
        planar_ratio_valid = forward > 1e-8 and lateral > 1e-8
        ratio_valid = planar > 1e-8 and yaw > 1e-8
        metrics.update(velocity_reward_planar_mean=planar, velocity_reward_yaw_mean=yaw,
            velocity_reward_ratio_valid=float(ratio_valid),
            velocity_yaw_to_planar_reward_ratio=yaw/planar if ratio_valid else 0.,
            velocity_reward_forward_mean=forward, velocity_reward_lateral_mean=lateral,
            velocity_planar_reward_ratio_valid=float(planar_ratio_valid),
            velocity_forward_to_lateral_reward_ratio=forward/lateral if planar_ratio_valid else 0.,
            velocity_planar_target_limited=float(cast(RewardContributionBalancer, self.velocity_balancer).planar_target_limited),
            velocity_reward_ratio_target=.25,
            velocity_yaw_target_limited=float(cast(RewardContributionBalancer, self.velocity_balancer).yaw_target_limited))
        return metrics

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if not self._training:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            self._playback_metadata = self._validate_playback_checkpoint(saved)
        infos = super().load(path, load_cfg, strict, map_location)
        self._velocity_reward_sum.zero_()
        self._velocity_reward_steps = 0
        self._last_velocity_reward_rates.zero_()
        return infos
