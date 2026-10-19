"""基座提前触地，以及连续停滞（而非累计停滞）触发的重置。"""

from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.experiments.transformation import SnakeToBipedRobust
from zbot_rl_mjlab.tasks.transformation import robust


@pytest.fixture
def failure_env(monkeypatch):
    cfg = SnakeToBipedRobust().build_env()
    data = SimpleNamespace(joint_pos=torch.zeros(2, 6), joint_vel=torch.zeros(2, 6), body_link_lin_vel_w=torch.zeros(2, 1, 3), body_link_ang_vel_w=torch.zeros(2, 1, 3))
    env = SimpleNamespace(cfg=cfg, num_envs=2, device="cpu", step_dt=0.02, common_step_counter=0, episode_length_buf=torch.full((2,), 100), heights=torch.full((2,), 0.0525), targets=torch.zeros(2, 6), contact=torch.ones(2, dtype=torch.bool), scene={"robot": SimpleNamespace(data=data, find_bodies=lambda name: ([0], [name]))}, reward_manager=SimpleNamespace(get_term_cfg=cfg.rewards.__getitem__))
    env.velocity_z = data.body_link_lin_vel_w[:, 0, 2]
    monkeypatch.setattr(robust, "base_height", lambda env: env.heights)
    monkeypatch.setattr(robust, "base_vertical_velocity", lambda env: env.velocity_z)
    monkeypatch.setattr(robust, "commanded_target", lambda env: env.targets)
    monkeypatch.setattr(robust, "base_ground_contact", lambda env: env.contact)
    monkeypatch.setattr(robust, "randomize_initial_snake_roll", lambda *args: None)
    return env


def test_static_initial_snake_contact_is_not_an_early_fall(failure_env):
    env = failure_env
    for step in range(100):
        env.common_step_counter = step
        assert not robust.early_base_ground_contact(env).any()
    assert not robust.fall_retry_state(env).motion_started.any()


def test_recontact_after_tiny_lift_resets_before_reaching_standing_height(failure_env):
    env = failure_env
    env.targets[:] = 0.02
    assert not robust.early_base_ground_contact(env).any()
    env.common_step_counter = 1
    env.contact[:] = False
    env.heights[:] = 0.0530
    assert not robust.early_base_ground_contact(env).any()
    env.common_step_counter = 2
    env.contact[:] = True
    env.heights[:] = 0.0525
    env.velocity_z[:] = -0.01
    assert robust.early_base_ground_contact(env).all()
    assert not robust.fall_retry_state(env).armed.any()
    robust.reset_fall_retry(env, torch.tensor([0]))
    state = robust.fall_retry_state(env)
    assert not state.early_base_fallen[0] and state.early_base_fallen[1]
    assert not state.had_base_contact[0] and not state.base_left_ground[0]


def test_initial_clearance_settling_does_not_count_as_recontact(failure_env):
    env = failure_env
    env.targets[:] = 0.02
    env.contact[:] = False
    assert not robust.early_base_ground_contact(env).any()
    env.common_step_counter = 1
    env.contact[:] = True
    env.heights[:] -= 0.0001
    env.velocity_z[:] = -0.05
    assert not robust.early_base_ground_contact(env).any()


def test_small_downward_impact_is_detected_even_if_soft_contact_never_breaks(failure_env):
    env = failure_env
    env.targets[:] = 0.02
    env.heights[:] = 0.058
    assert not robust.early_base_ground_contact(env).any()
    env.common_step_counter = 1
    env.heights[:] = 0.056
    env.velocity_z[:] = -0.04
    assert robust.early_base_ground_contact(env).all()


def test_stagnation_requires_more_than_one_continuous_second(failure_env):
    env = failure_env
    for step in range(1, 51):
        env.common_step_counter = step
        assert not robust.abandoned_exploration(env).any()
        assert not robust.abandoned_exploration(env).any()
    env.common_step_counter = 51
    assert robust.abandoned_exploration(env).all()
    robust.reset_fall_retry(env, torch.tensor([0]))
    assert robust.fall_retry_state(env).still_time[0] == 0
    assert robust.fall_retry_state(env).still_time[1] > 1


@pytest.mark.parametrize("motion", ["joint", "linear", "angular"])
def test_motion_clears_timer_and_separate_pauses_do_not_accumulate(failure_env, motion):
    env = failure_env
    for step in range(1, 50):
        env.common_step_counter = step
        assert not robust.abandoned_exploration(env).any()
    data = env.scene["robot"].data
    values = {"joint": data.joint_vel, "linear": data.body_link_lin_vel_w, "angular": data.body_link_ang_vel_w}[motion]
    values[:] = 0.1
    env.common_step_counter = 50
    assert not robust.abandoned_exploration(env).any()
    assert robust.fall_retry_state(env).still_time.eq(0).all()
    values[:] = 0
    for step in range(51, 101):
        env.common_step_counter = step
        assert not robust.abandoned_exploration(env).any()
    env.common_step_counter = 101
    assert robust.abandoned_exploration(env).all()


@pytest.mark.parametrize("near_height,near_pose,allowed", [(True, True, True), (False, True, False), (True, False, False)])
def test_only_near_both_target_height_and_configuration_can_stay_still(failure_env, near_height, near_pose, allowed):
    env = failure_env
    if near_height:
        env.heights[:] = env.cfg.standing_base_height
    if near_pose:
        env.scene["robot"].data.joint_pos[:] = torch.tensor(env.cfg.standing_joint_pos)
    for step in range(1, 52):
        env.common_step_counter = step
        result = robust.abandoned_exploration(env)
    assert result.tolist() == [not allowed, not allowed]


def test_initial_grace_does_not_consume_the_stagnation_window(failure_env):
    env = failure_env
    for step in range(1, 15):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        assert not robust.abandoned_exploration(env).any()
    assert robust.fall_retry_state(env).still_time.eq(0).all()
