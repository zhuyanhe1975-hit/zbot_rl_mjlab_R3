"""Archived reward units and teacher-only phase adaptations, without simulation."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from test_universal_task import state_env

from zbot_rl_mjlab.experiments.omni_teacher import (
    configure_teacher_rewards as _recipe_configure_teacher_rewards,
)
from zbot_rl_mjlab.experiments.universal import universal_env_cfg
from zbot_rl_mjlab.tasks.transformation import commanded, robust
from zbot_rl_mjlab.tasks.universal import teacher_rewards as rewards
from zbot_rl_mjlab.tasks.walking import mdp as walking

BESTS = Path(__file__).resolve().parents[1] / 'bests'
TRANSFORM = BESTS / 'transform_snake_biped_bidirectional_2026-10-17_iter599'
WALKING = BESTS / 'deploy_walking_2026-10-11_iter1499'


def archive(path):
    # BaseLoader accepts legacy Python tags as data without executing constructors.
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_recipe_matches_effective_archive_weights_and_parameters():
    cfg = _recipe_configure_teacher_rewards(universal_env_cfg())
    effective = json.loads((TRANSFORM / 'final_effective_weights.json').read_text())
    transform = archive(TRANSFORM / 'env.yaml')
    walking = archive(WALKING / 'env.yaml')
    for name, weight in effective.items():
        assert cfg.rewards[name].weight == weight
        actual = cfg.rewards[name].params
        expected = transform['rewards'][name]['params']
        for key, value in actual.items():
            if isinstance(value, tuple):
                assert value == tuple(float(x) for x in expected[key])
            elif isinstance(value, (int, float)):
                assert value == float(expected[key])
            else:
                assert value == expected[key]
    for name in ('support_foot_slip', 'foot_nonparallel_ground'):
        assert cfg.rewards[name].weight == float(walking['rewards'][name]['weight'])
        for key, value in cfg.rewards[name].params.items():
            expected = walking['rewards'][name]['params'][key]
            if isinstance(value, tuple):
                assert value == tuple(expected)
            elif isinstance(value, (int, float)):
                assert value == float(expected)
            else:
                assert value == expected
    assert cfg.rewards['joint_energy'].weight == float(walking['rewards']['joint_energy']['weight'])
    assert cfg.rewards['joint_energy'].params['energy_scale'] == float(walking['rewards']['joint_energy']['params']['energy_scale'])
    assert cfg.rewards['foot_nonparallel_ground'].params['std'] == .1
    assert not {'energy', 'contact_impact', 'failure'} & cfg.rewards.keys()
    assert cfg.rewards['velocity_tracking'].weight == 1
    assert cfg.rewards['mode_posture'].weight == 2
    assert cfg.rewards['transition_progress'].weight == 2
    assert cfg.rewards['standstill'].weight == 1
    assert cfg.rewards['action_rate'].weight == -.01


def test_dataclass_constants_match_archived_yaml():
    cfg = cast(rewards.ArchiveOmniTeacherEnvCfg, universal_env_cfg(cfg_cls=rewards.ArchiveOmniTeacherEnvCfg))
    source = archive(TRANSFORM / 'env.yaml')
    for key in ('joint_standing_error_scale', 'standing_pose_gravity_scale', 'standing_base_height', 'sole_alignment_height_std', 'sole_alignment_angle_std_deg', 'ascent_reward_stop_height', 'support_angle_limit_deg', 'support_height_limit', 'curriculum_start_stage'):
        assert getattr(cfg, key) == float(source[key])
    assert cfg.standing_joint_pos == tuple(float(x) for x in source['standing_joint_pos'])
    assert cfg.require_sole_support_for_positive_rewards is (source['require_sole_support_for_positive_rewards'] == 'true')


def test_normalized_energy_preserves_absolute_power_units_and_bounds():
    data = SimpleNamespace(actuator_force=torch.tensor([[2., -3.], [20., 20.], [0., 0.]]), joint_vel=torch.tensor([[-1., 1.], [2., -2.], [100., 100.]]))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)})
    result = walking.joint_energy_penalty(env, energy_scale=10.)
    torch.testing.assert_close(result, torch.tensor([.5, 1., 0.]))
    assert torch.isfinite(result).all() and ((0 <= result) & (result <= 1)).all()


@pytest.mark.parametrize('wrapper,source', [('transition_pose', 'pose_reward'), ('transition_height', 'height_reward'), ('transition_sole', 'sole_reward'), ('transition_finish_speed', 'stand_finish_joint_speed_cost'), ('transition_descent_impact', 'descent_ground_impact_cost')])
def test_transform_rewards_do_not_freeze_motion(monkeypatch, wrapper, source):
    env, term = state_env()
    term.transition_mask[:] = torch.tensor([False, True])
    monkeypatch.setattr(commanded, source, lambda env, **kw: torch.tensor([2., -3.]))
    result = getattr(rewards, wrapper)(env)
    torch.testing.assert_close(result, torch.tensor([0., -3.]))


def test_biped_foot_costs_do_not_penalize_snake_or_transition(monkeypatch):
    env, term = state_env()
    monkeypatch.setattr(walking, 'support_foot_slip', lambda env, **kw: torch.tensor([2., 3.]))
    monkeypatch.setattr(walking, 'foot_nonparallel_ground_penalty', lambda env, **kw: torch.tensor([2., 3.]))
    term.transition_mask[:] = False
    term.command[:, 0] = torch.tensor([1., -1.])
    for func in (rewards.biped_support_foot_slip, rewards.biped_foot_nonparallel):
        torch.testing.assert_close(func(env), torch.tensor([2., 0.]))
        term.transition_mask[0] = True
        assert func(env).eq(0).all()
        term.transition_mask[0] = False


def teacher_env():
    env, term = state_env()
    cfg = cast(rewards.ArchiveOmniTeacherEnvCfg, universal_env_cfg(cfg_cls=rewards.ArchiveOmniTeacherEnvCfg))
    env.cfg = cfg
    env.command_manager.get_command = lambda name: term.command
    return env, term


def test_ascent_peak_memory_resets_on_command_and_episode():
    env, term = teacher_env()
    term.command[:, 0] = 1
    term.transition_mask[:] = True
    data = env.scene['robot'].data
    data.body_link_pos_w[:, 6, 2] = .25
    rewards.transition_ascent(env)
    assert rewards._ascent_memory(env).peak_height.ge(.25).all()
    data.body_link_pos_w[:, 6, 2] = .1
    assert rewards.transition_ascent(env).eq(0).all()  # peak latch persists
    term.start_step[0] += 1
    rewards.transition_ascent(env)
    assert rewards._ascent_memory(env).peak_height[0] == pytest.approx(.1)
    rewards.reset_ascent_memory(env, torch.tensor([1]))
    rewards.transition_ascent(env)
    assert rewards._ascent_memory(env).peak_height[1] == pytest.approx(.1)
    term.transition_mask[:] = False
    assert rewards.transition_ascent(env).eq(0).all()


def test_real_transform_rewards_are_finite_and_finish_is_height_gated():
    env, term = teacher_env()
    term.transition_mask[:] = True
    term.command[:, 0] = torch.tensor([1., -1.])
    for func in (rewards.transition_pose, rewards.transition_height, rewards.transition_ascent, rewards.transition_sole, rewards.transition_finish_speed):
        result = func(env)
        assert result.shape == (2,) and torch.isfinite(result).all()
    data = env.scene['robot'].data
    data.joint_vel[:] = 2
    data.body_link_pos_w[:, 6, 2] = .1
    assert rewards.transition_finish_speed(env).eq(0).all()
    data.body_link_pos_w[:, 6, 2] = .264
    torch.testing.assert_close(rewards.transition_finish_speed(env), torch.tensor([4., 0.]))


def test_ascent_matches_archive_formula_and_rearms_after_reverse_command():
    env, term = teacher_env()
    term.transition_mask[:] = True
    term.command[:, 0] = 1
    data = env.scene['robot'].data
    data.body_link_pos_w[:, 6, 2] = .1
    data.body_link_lin_vel_w[:, 6, 2] = torch.tensor([.15, -.03])
    data.joint_pos[:] = 0
    torch.testing.assert_close(rewards.transition_ascent(env), commanded.velocity_reward(env))
    data.body_link_pos_w[:, 6, 2] = .25
    assert rewards.transition_ascent(env).eq(0).all()
    term.command[:, 0] = -1
    data.body_link_pos_w[:, 6, 2] = .1
    assert rewards.transition_ascent(env).eq(0).all()
    term.command[:, 0] = 1
    result = rewards.transition_ascent(env)
    assert result[0] > 0 and result[1] < 0
    # Teacher cache remains isolated from the legacy fall/retry state.
    assert robust.fall_retry_state(env).peak_base_height.eq(.1).all()


def test_descending_impact_units_and_failure_charged_once():
    env, term = teacher_env()
    env.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(2, 12)))
    term.transition_mask[:] = True
    term.command[:, 0] = torch.tensor([1., -1.])
    env.scene['transform_ground_contact'].data.force_history[..., 2] = 4 * 12 * 9.81
    torch.testing.assert_close(rewards.transition_descent_impact(env), torch.tensor([0., 4.]))
    term.transition_mask[:] = False
    assert rewards.transition_descent_impact(env).eq(0).all()
    cfg = _recipe_configure_teacher_rewards(universal_env_cfg())
    env.termination_manager = SimpleNamespace(terminated=torch.tensor([True, False]))
    penalty = cfg.rewards['reset_failure']
    assert penalty.func(env).eq(0).all()  # Walking falls retain termination, without the transform-only fee.
    term.transition_mask[:] = True
    torch.testing.assert_close(penalty.func(env) * penalty.weight * env.step_dt, torch.tensor([-10., 0.]))
    env.termination_manager.terminated[:] = False  # timeouts are not failures
    assert penalty.func(env).eq(0).all()


def test_contact_sensor_api_supports_real_biped_costs():
    env, term = teacher_env()
    term.command[:, 0] = torch.tensor([1., -1.])
    term.transition_mask[:] = False
    data = env.scene['robot'].data
    data.site_lin_vel_w = torch.ones(2, 2, 3)
    data.site_ang_vel_w = torch.ones(2, 2, 3)
    for func in (rewards.biped_support_foot_slip, rewards.biped_foot_nonparallel):
        result = func(env)
        assert result.shape == (2,) and torch.isfinite(result).all()
        assert result[1] == 0
