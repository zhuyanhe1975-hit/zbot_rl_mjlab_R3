"""Transformation failure history must not leak between targets or penalize locomotion."""
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from test_universal_task import state_env

from zbot_rl_mjlab.tasks.universal import teacher_transition as failure


def env_state():
    env, term = state_env()
    term.cfg.training = True
    term.command[:, 0] = 1
    term.transition_mask[:] = True
    env.command_manager.get_command = lambda name: term.command
    data = env.scene['robot'].data
    data.joint_vel.zero_()
    data.body_link_lin_vel_w.zero_()
    data.body_link_ang_vel_w.zero_()
    base_id = env.scene['robot'].find_bodies('base')[0][0]
    data.body_link_pos_w[:, base_id, 2] = .08
    return env, term, data, base_id


def tick(env, count=1):
    result = torch.zeros(env.num_envs, dtype=torch.bool)
    for _ in range(count):
        env.common_step_counter += 1
        result = failure.transition_failure(env)
    return result


def test_all_detector_constants_match_successful_bidirectional_archive():
    archive = Path(__file__).resolve().parents[1] / 'bests/transform_snake_biped_bidirectional_2026-10-17_iter599/env.yaml'
    cfg = yaml.load(archive.read_text(), Loader=yaml.BaseLoader)
    for name, value in asdict(failure.ARCHIVE_FAILURE_CFG).items():
        assert value == float(cfg[name])


def test_initial_ascent_fall_requires_actual_drop_and_hold_once_per_step():
    env, term, data, base = env_state()
    assert not failure.transition_failure(env).any()
    data.body_link_pos_w[:, base, 2] = .06
    for step in range(4):
        env.common_step_counter += 1
        result = failure.transition_failure(env)
        before = failure.transition_memory(env).low_time.clone()
        assert not failure.transition_failure(env).any()
        torch.testing.assert_close(failure.transition_memory(env).low_time, before)
        assert not result.any()
    assert tick(env).all()  # .1 seconds continuously below .1m after a .02m drop.
    term.command[:, 0] = -1
    assert not failure.transition_failure(env).any()  # Descent is never an ascent fall.


def test_stagnation_uses_command_grace_and_one_second_hold():
    env, _term, _data, _base = env_state()
    for _ in range(14):
        assert not tick(env).any()
    assert failure.transition_memory(env).still_time.eq(0).all()
    for _ in range(50):
        assert not tick(env).any()
    assert not tick(env).any()  # Float32 grace starts at the first representable eligible step.
    assert tick(env).all()  # Archive uses >hold+epsilon, rather than >=hold.


def test_command_change_and_episode_reset_clear_all_history_without_physics_write():
    env, term, data, base = env_state()
    failure.transition_failure(env)
    data.body_link_pos_w[:, base, 2] = .06
    tick(env, 4)
    term.start_step[:] = env.common_step_counter
    assert not failure.transition_failure(env).any()
    assert failure.transition_memory(env).low_time.eq(0).all()
    before = data.joint_pos.clone(), data.body_link_pos_w.clone(), term.command.clone()
    failure.reset_transition_failure_memory(env, torch.tensor([0]))
    assert failure.transition_memory(env).peak[0].isneginf()
    assert failure.transition_memory(env).peak[1].isfinite()
    for actual, expected in zip((data.joint_pos, data.body_link_pos_w, term.command), before):
        torch.testing.assert_close(actual, expected)
    data.body_link_pos_w[:, base, 2] = .04
    assert not tick(env).any()  # Selected episode reset erased its previously meaningful peak.


def test_only_transitions_not_stable_motion_or_manual_play_are_eligible():
    env, term, _data, _base = env_state()
    term.transition_mask[:] = False
    tick(env, 100)
    assert not failure.transition_failure(env).any()
    term.transition_mask[:] = True
    term.manual_control[:] = True
    tick(env, 100)
    assert not failure.transition_failure(env).any()
    term.manual_control[:] = False
    term.cfg.training = False
    assert not failure.transition_failure(env).any()


def test_canonical_stand_and_snake_goals_do_not_count_as_abandoned():
    env, term, data, base = env_state()
    term.command[:, 0] = torch.tensor([1., -1.])
    data.body_link_pos_w[:, base, 2] = torch.tensor([env.cfg.standing_base_height, .0525])
    data.joint_pos[1] = 0
    for _ in range(100):
        assert not tick(env).any()


def test_real_joint_or_base_motion_prevents_stagnation():
    env, _term, data, base = env_state()
    data.joint_vel[0] = .04
    data.body_link_lin_vel_w[1, base, 0] = .02
    for _ in range(100):
        assert not tick(env).any()
    assert failure.transition_memory(env).still_time.eq(0).all()


def test_early_contact_requires_meaningful_rise_drop_and_impact_or_separation():
    env, _term, data, base = env_state()
    contacts = env.scene['base_ground_contact'].data.found
    contacts[:] = True
    assert not failure.transition_failure(env).any()  # No drop yet.
    data.body_link_pos_w[:, base, 2] = .06
    assert not tick(env).any()  # No downward speed and never left ground.
    data.body_link_lin_vel_w[0, base, 2] = -.03
    result = tick(env)
    assert result.tolist() == [True, False]
    contacts[1] = False
    tick(env)
    contacts[1] = True
    assert tick(env).all()  # Ground separation makes downward-velocity threshold unnecessary.


def test_inference_created_memory_can_reset_and_update_outside_inference():
    env, _term, _data, _base = env_state()
    with torch.inference_mode():
        failure.transition_failure(env)
    assert failure.transition_memory(env).peak.is_inference()
    failure.reset_transition_failure_memory(env, None)
    assert failure.transition_memory(env).peak.isneginf().all()
    assert not tick(env).any()
