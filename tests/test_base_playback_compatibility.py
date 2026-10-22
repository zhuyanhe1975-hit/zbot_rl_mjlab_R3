"""Reward-target changes preserve inference compatibility, with strict training resumes."""
from copy import deepcopy
from typing import cast

import pytest
import torch

from zbot_rl_mjlab.adaptive_site_snake import VELOCITY_REWARDS
from zbot_rl_mjlab.base_snake_teacher import (
    BaseSnakeAdaptiveTeacherRunner,
    RewardContributionBalancer,
)
from zbot_rl_mjlab.experiments.site_snake import base_snake_env_cfg
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg


def legacy_checkpoint():
    state = RewardContributionBalancer(2.).state_dict() | {
        'version': 5, 'yaw_to_planar_reward_target': .5,
        'weights': [2/3, 2/3, 2/3], 'reward_mean_ema': [-.7, -.6, -.5],
        'reward_mean_initialized': True, 'updates': 101,
    }
    metadata = BaseSnakeAdaptiveTeacherRunner._metadata_schema() | {
        'reward_recipe': 'base_snake_relative_equal_planar_half_yaw_v7',
        'yaw_to_planar_reward_target': .5, 'velocity_limits': (.2, .2, .5),
        'joint_target_limit': 2*torch.pi/3, 'snake_target_limit': 2*torch.pi/3,
        'max_target_speed': 2*torch.pi,
    }
    return {'infos': {'privileged_omni_teacher': metadata, 'adaptive_velocity_weights': state}}


def test_half_yaw_checkpoint_can_play_without_mutating_saved_contract(tmp_path):
    saved = legacy_checkpoint()
    original = deepcopy(saved)
    path = tmp_path/'half.pt'
    torch.save(saved, path)
    cfg = base_snake_env_cfg(play=True)
    BaseSnakeAdaptiveTeacherRunner.prepare_playback_env_cfg(cfg, path)
    assert cfg.motion_site == 'base_motion'
    assert cast(SnakeCrawlJointTargetRateActionCfg, cfg.actions['joint_pos']).target_limit == pytest.approx(2*torch.pi/3)
    for name in VELOCITY_REWARDS:
        assert cfg.rewards[name].weight == pytest.approx(2/3)
    restored = BaseSnakeAdaptiveTeacherRunner._load_velocity_balancer(saved, playback=True)
    assert isinstance(restored, RewardContributionBalancer)
    torch.testing.assert_close(restored.reward_mean_ema, torch.tensor([-.7, -.6, -.5]))
    torch.testing.assert_close(restored.weights, torch.full((3,), 2/3))
    assert restored.updates == 101
    assert saved == original == torch.load(path, weights_only=False)


def test_half_yaw_checkpoint_still_rejected_for_training():
    saved = legacy_checkpoint()
    with pytest.raises(ValueError, match='compatible'):
        BaseSnakeAdaptiveTeacherRunner._validate_checkpoint(saved)
    with pytest.raises(ValueError):
        BaseSnakeAdaptiveTeacherRunner._load_velocity_balancer(saved, playback=False)


@pytest.mark.parametrize('change', [
    {'tracking_point': 'foot_0_sole'}, {'motion_site': 'foot_0_sole'},
    {'actor_obs_dim': 60}, {'action_dim': 5}, {'observation_recipe': 'different'},
    {'reward_recipe': 'unknown'}, {'yaw_to_planar_reward_target': .125},
    {'joint_target_limit': 0.}, {'max_target_speed': float('nan')},
])
def test_playback_keeps_non_reward_target_contract_checks(change):
    saved = legacy_checkpoint()
    saved['infos']['privileged_omni_teacher'].update(change)
    with pytest.raises(ValueError):
        BaseSnakeAdaptiveTeacherRunner._validate_playback_checkpoint(saved)


@pytest.mark.parametrize('change', [
    {'version': 4}, {'yaw_to_planar_reward_target': .25},
    {'weights': [float('nan')]*3}, {'reward_mean_ema': [-3., 0., 0.]},
    {'contribution_statistic': 'mean_absolute'},
])
def test_playback_rejects_invalid_legacy_adaptive_state(change):
    saved = legacy_checkpoint()
    saved['infos']['adaptive_velocity_weights'].update(change)
    with pytest.raises(ValueError):
        BaseSnakeAdaptiveTeacherRunner._load_velocity_balancer(saved, playback=True)


def test_previous_quarter_yaw_recipe_still_plays_with_saved_weights(tmp_path):
    saved = legacy_checkpoint()
    saved['infos']['privileged_omni_teacher'].update(
        reward_recipe='base_snake_relative_equal_planar_quarter_yaw_v8', yaw_to_planar_reward_target=.25)
    state = saved['infos']['adaptive_velocity_weights']
    state.update(version=6, yaw_to_planar_reward_target=.25)
    for key in ('stage', 'planar_positive_updates', 'planar_regression_updates'):
        state.pop(key, None)
    path = tmp_path/'quarter.pt'
    torch.save(saved, path)
    cfg = base_snake_env_cfg(play=True)
    BaseSnakeAdaptiveTeacherRunner.prepare_playback_env_cfg(cfg, path)
    for name in VELOCITY_REWARDS:
        assert cfg.rewards[name].weight == pytest.approx(2/3)
    with pytest.raises(ValueError, match='compatible'):
        BaseSnakeAdaptiveTeacherRunner._validate_checkpoint(saved)
