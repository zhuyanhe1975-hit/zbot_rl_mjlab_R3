"""稳定完成目标后交替切换命令，无需重置机器人。"""

import pytest
import torch
from test_transformation_commanded import environment

from zbot_rl_mjlab.experiments.bidirectional import SnakeBipedBidirectional
from zbot_rl_mjlab.tasks.transformation import commanded, mdp, robust
from zbot_rl_mjlab.tasks.transformation.commanded import BidirectionalEnvCfg


def ready_environment(monkeypatch):
    env, term = environment()
    term.reset(torch.arange(2))
    env._test_goal_valid = torch.ones(2, dtype=torch.bool)
    monkeypatch.setattr(commanded, "final_condition", lambda current, **kwargs: current._test_goal_valid.clone())
    env.common_step_counter = 100
    env.episode_length_buf[:] = 100
    mdp.state(env).hold_time.fill_(env.cfg.success_hold_s)
    term.switch_hold_time.fill_(env.cfg.success_hold_s)
    return env, term


@pytest.mark.parametrize("play", [False, True])
def test_default_source_mode_enables_automatic_switches_and_logs_ever_success(play):
    cfg = SnakeBipedBidirectional().build_env(play=play)
    assert isinstance(cfg, BidirectionalEnvCfg)
    assert cfg.commands["transform"].auto_switch_on_success
    assert cfg.metrics["success"].reduce == "max"
    assert cfg.success_hold_s == 2 and cfg.episode_length_s == 16


def test_both_directions_flip_once_and_preserve_physical_and_episode_state(monkeypatch):
    env, term = ready_environment(monkeypatch)
    data = env.scene["robot"].data
    tensors = [data.joint_pos, data.joint_vel, data.body_link_quat_w,
               data.body_link_pos_w, data.body_link_lin_vel_w,
               env.action_manager.get_term("joint_pos").commanded_target,
               env.episode_length_buf, env._transform_initial_standing]
    before = [t.clone() for t in tensors]
    retry = robust.fall_retry_state(env)
    retry.armed.fill_(True)
    retry.peak_base_height.fill_(0.3)
    term.compute(env.step_dt)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([-1., 1.]))
    assert term.start_step.eq(100).all() and not term.manual_control.any()
    assert term.metrics["completed_transitions"].eq(1).all()
    assert mdp.state(env).hold_time.eq(0).all()
    assert term.switch_hold_time.eq(0).all()
    assert not retry.armed.any() and retry.peak_base_height.isneginf().all()
    for actual, expected in zip(tensors, before, strict=True):
        torch.testing.assert_close(actual, expected)
    term.compute(0.)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([-1., 1.]))
    assert term.metrics["completed_transitions"].eq(1).all()
    assert commanded.command(env).eq(term.command).all()


def test_no_switch_before_continuous_hold_or_when_goal_is_no_longer_valid(monkeypatch):
    env, term = ready_environment(monkeypatch)
    term.switch_hold_time[0] = env.cfg.success_hold_s - env.step_dt
    env._test_goal_valid[1] = False
    term.compute(env.step_dt)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., -1.]))
    assert not term.metrics["completed_transitions"].any()


def test_stability_timer_restarts_after_interruption_and_each_switch(monkeypatch):
    env, term = ready_environment(monkeypatch)
    mdp.state(env).hold_time.zero_()
    term.switch_hold_time.zero_()
    for step in range(101, 151):
        env.common_step_counter = step
        commanded.update_state(env)
        term.compute(env.step_dt)
    assert not term.metrics["completed_transitions"].any()
    env._test_goal_valid.fill_(False)
    env.common_step_counter += 1
    commanded.update_state(env)
    assert mdp.state(env).hold_time.eq(0).all()
    assert term.switch_hold_time.eq(0).all()
    env._test_goal_valid.fill_(True)
    for _ in range(200):
        env.common_step_counter += 1
        commanded.update_state(env)
        term.compute(env.step_dt)
    assert term.metrics["completed_transitions"].eq(2).all()
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., -1.]))


def test_manual_world_is_locked_while_other_world_keeps_alternating(monkeypatch):
    env, term = ready_environment(monkeypatch)
    term.set_target("stand", torch.tensor([0]))
    mdp.state(env).hold_time.fill_(env.cfg.success_hold_s)
    term.compute(env.step_dt)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., 1.]))
    torch.testing.assert_close(term.metrics["completed_transitions"], torch.tensor([0., 1.]))
    assert term.manual_control.tolist() == [True, False]


@pytest.mark.parametrize("mode,enabled", [("stand", True), ("snake", True), ("source", False)])
def test_fixed_direction_or_disabled_switch_retains_target(monkeypatch, mode, enabled):
    env, term = ready_environment(monkeypatch)
    term.cfg.mode = mode
    term.cfg.auto_switch_on_success = enabled
    before = term.command.clone()
    term.compute(env.step_dt)
    torch.testing.assert_close(term.command, before)
    assert not term.metrics["completed_transitions"].any()


def test_terminal_and_reset_scoped_updates_do_not_switch_targets(monkeypatch):
    env, term = ready_environment(monkeypatch)
    env.reset_buf = torch.tensor([True, False])
    term.compute(0., env_ids=torch.tensor([0]))
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., -1.]))
    term.compute(env.step_dt)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., 1.]))


def test_completed_counts_are_logged_and_reset_only_for_selected_worlds(monkeypatch):
    env, term = ready_environment(monkeypatch)
    term.metrics["completed_transitions"][:] = torch.tensor([3., 5.])
    extras = term.reset(torch.tensor([0]))
    assert extras["completed_transitions"] == 3
    torch.testing.assert_close(term.metrics["completed_transitions"], torch.tensor([0., 5.]))
    assert mdp.state(env).hold_time[1] == env.cfg.success_hold_s


def test_partial_reset_balances_current_commands_after_automatic_reversals():
    env, term = environment()
    env._transform_initial_standing.zero_()  # 历史初始状态记录的是站立目标。
    term._command.fill_(-1)  # 此后两个环境均已切换为蛇形目标。
    chosen = commanded.sample_initial_standing(env, torch.tensor([0]))
    assert not chosen.any()  # 在重置槽位中填入站立目标。
    assert not env._transform_initial_standing.any()
    term._command.fill_(1)
    assert commanded.sample_initial_standing(env, torch.tensor([0])).all()


def test_still_pose_with_slight_joint_error_switches_without_relaxing_strict_success():
    env, term = environment()
    term.reset(torch.arange(2))
    term._command.fill_(1)
    env.scene["robot"].data.joint_pos[:, 1] += 0.11
    assert not commanded.final_condition(env).any()
    assert commanded.auto_switch_condition(env).all()
    for _ in range(100):
        env.common_step_counter += 1
        commanded.update_state(env)
    assert mdp.state(env).hold_time.eq(0).all()
    assert not commanded.success(env).any()
    assert term.switch_hold_time.ge(env.cfg.success_hold_s - 1e-5).all()
    term.compute(env.step_dt)
    assert term.command.eq(-1).all()
    assert term.metrics["completed_transitions"].eq(1).all()


def test_switch_condition_retains_velocity_pose_and_contact_requirements():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.joint_pos[:, 1] += 0.11
    data.joint_vel[0] = 0.2
    env.scene["feet_ground_contact"].data.found[1] = 0
    assert not commanded.auto_switch_condition(env).any()
    data.joint_vel.zero_()
    env.scene["feet_ground_contact"].data.found.fill_(1)
    data.joint_pos[:, 1] += 0.1
    assert not commanded.auto_switch_condition(env).any()


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_invalid_switch_joint_tolerance_is_rejected(value):
    env, _term = environment()
    env.cfg.commands["transform"].auto_switch_joint_tolerance = value
    with pytest.raises(ValueError, match="auto_switch_joint_tolerance"):
        env.cfg.commands["transform"].build(env)
