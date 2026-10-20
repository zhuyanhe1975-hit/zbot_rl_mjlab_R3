"""Keep the successful biped exploration objective inside the joint teacher."""
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from mjlab.sensor import ContactSensorCfg, TerrainHeightSensorCfg
from test_universal_task import state_env

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal import teacher_rewards as rewards
from zbot_rl_mjlab.tasks.universal.env_cfg import universal_env_cfg
from zbot_rl_mjlab.tasks.walking import mdp as walking


def test_biped_tracking_parameters_match_successful_omni_archive():
    path = Path(__file__).resolve().parents[1] / 'bests/omni_velocity_energy_2026-10-10_iter499/env.yaml'
    archived = yaml.load(path.read_text(), Loader=yaml.BaseLoader)['rewards']
    cfg = rewards.configure_teacher_rewards(universal_env_cfg())
    tracking = cfg.rewards['velocity_tracking']
    names = ('track_forward_speed', 'track_lateral_speed', 'track_yaw_rate')
    assert tracking.params['std'] == tuple(float(archived[name]['params']['std']) for name in names)
    assert tracking.params['axis_weights'] == tuple(float(archived[name]['weight']) for name in names)
    assert tracking.weight == 1
    assert cfg.rewards['biped_upright'].weight == float(archived['upright']['weight'])
    for name in ('joint_limits', 'joint_acceleration', 'soft_landing', 'swing_clearance'):
        assert cfg.rewards[name].weight == float(archived[name]['weight'])
    assert cfg.terminations['biped_fallen'].params == {'height': .22, 'tilt': .75}


@pytest.mark.parametrize('axis', range(3))
def test_biped_tracking_is_signed_and_has_broad_initial_feedback(monkeypatch, axis):
    env, term = state_env()
    term.command[:, 0] = 1
    term.transition_mask[:] = False
    term.velocity_command[:] = 0
    term.velocity_command[:, axis] = .04
    actual = torch.zeros(2, 3)
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda env: actual)
    stationary = rewards.teacher_velocity_tracking(env)
    assert stationary.min() > 3.5  # Dense archived feedback even before learning balance.
    actual[0, axis], actual[1, axis] = .04, -.04
    result = rewards.teacher_velocity_tracking(env)
    assert result[0] == pytest.approx(4.)
    assert result[0] > stationary[0] > result[1]
    term.velocity_command[:, axis] = -.04
    assert rewards.teacher_velocity_tracking(env)[1] > rewards.teacher_velocity_tracking(env)[0]


def test_tracking_keeps_snake_signed_progress_and_zero_transition(monkeypatch):
    env, term = state_env()
    term.command[:, 0] = -1
    term.transition_mask[:] = False
    term.velocity_command[:] = torch.tensor([.04, 0., 0.])
    actual = torch.zeros(2, 3)
    actual[1, 0] = .04
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda env: actual)
    torch.testing.assert_close(rewards.teacher_velocity_tracking(env), torch.tensor([0., .6]))
    term.transition_mask[:] = True
    assert rewards.teacher_velocity_tracking(env).eq(0).all()


def test_biped_fall_rejects_failed_motion_even_after_losing_readiness(monkeypatch):
    env, term = state_env()
    term.cfg.training = True
    term.command[:, 0] = 1
    term.standing_achieved[:] = True
    term.transition_mask[:] = torch.tensor([False, True])
    monkeypatch.setattr(walking, 'fallen', lambda env, **kw: torch.ones(2, dtype=torch.bool))
    assert rewards.biped_fallen(env).all()
    term.standing_achieved[0] = False
    term.command[1, 0] = -1  # Intentional descent remains legal even after previously standing.
    assert not rewards.biped_fallen(env).any()


def test_biped_fall_uses_real_archived_height_and_tilt_thresholds():
    env, term = state_env()
    term.cfg.training = True
    term.command[:, 0] = 1
    term.standing_achieved[:] = True
    data = env.scene['robot'].data
    base_id = env.scene['robot'].find_bodies('base')[0][0]
    data.body_link_pos_w[:, base_id, 2] = torch.tensor([.219, .264])
    assert rewards.biped_fallen(env).tolist() == [True, False]
    data.body_link_quat_w[1, base_id] = torch.tensor([1., 0., 0., 0.])
    assert rewards.biped_fallen(env).all()  # Body-Z points upward in this fallen orientation.


def test_manual_play_and_non_biped_upright_gates(monkeypatch):
    env, term = state_env()
    term.cfg.training = False
    term.command[:, 0] = torch.tensor([1., -1.])
    term.standing_achieved[:] = True
    term.manual_control[:] = True
    term.transition_mask[:] = False
    monkeypatch.setattr(walking, 'fallen', lambda env, **kw: torch.ones(2, dtype=torch.bool))
    monkeypatch.setattr(walking, 'upright', lambda env: torch.ones(2))
    assert not rewards.biped_fallen(env).any()
    torch.testing.assert_close(rewards.biped_upright(env), torch.tensor([1., 0.]))
    term.transition_mask[:] = True
    assert rewards.biped_upright(env).eq(0).all()


def test_biped_sensor_recipe_reuses_sole_sites_and_contact_history():
    cfg = rewards.configure_teacher_rewards(universal_env_cfg())
    rewards.configure_teacher_rewards(cfg)  # Reconfiguration does not duplicate sensors.
    feet = cast(ContactSensorCfg, next(sensor for sensor in cfg.scene.sensors if sensor.name == 'feet_ground_contact'))
    heights = [sensor for sensor in cfg.scene.sensors if sensor.name == 'foot_height_scan']
    assert feet.track_air_time and feet.history_length == cfg.decimation == 4
    assert len(heights) == 1
    height = cast(TerrainHeightSensorCfg, heights[0])
    assert isinstance(height.frame, tuple)
    assert tuple((frame.type, frame.name) for frame in height.frame) == (('site', 'foot_0_sole'), ('site', 'foot_1_sole'))
    assert not height.debug_vis


def test_joint_acceleration_uses_archived_absolute_sum_and_motion_gate():
    env, term = state_env()
    term.command[:, 0] = torch.tensor([1., -1.])
    term.transition_mask[:] = False
    env.scene['robot'].data.joint_acc = torch.tensor([[1., -2., 3., -4., 5., -6.], [10., 10., 10., 10., 10., 10.]])
    torch.testing.assert_close(rewards.biped_joint_acceleration(env), torch.tensor([21., 0.]))
    term.transition_mask[:] = True
    assert rewards.biped_joint_acceleration(env).eq(0).all()


def test_joint_limits_and_landing_use_original_functions_with_phase_gates(monkeypatch):
    env, term = state_env()
    term.command[:, 0] = torch.tensor([1., -1.])
    term.transition_mask[:] = False
    for wrapper, source in ((rewards.biped_joint_limits, 'joint_pos_limits'), (rewards.biped_soft_landing, 'soft_landing')):
        monkeypatch.setattr(walking, source, lambda env, **kw: torch.tensor([2., 3.]))
        torch.testing.assert_close(wrapper(env), torch.tensor([2., 0.]))
        term.transition_mask[:] = True
        assert wrapper(env).eq(0).all()
        term.transition_mask[:] = False


def test_action_rate_uses_archived_biped_weight_and_retains_other_modes():
    env, term = state_env()
    env.action_manager.action[:] = 1
    term.command[:, 0] = torch.tensor([1., -1.])
    term.transition_mask[:] = False
    cfg = rewards.configure_teacher_rewards(universal_env_cfg())
    penalty = cfg.rewards['action_rate']
    torch.testing.assert_close(penalty.func(env) * penalty.weight, torch.tensor([-.3, -.06]))
    term.transition_mask[:] = True
    torch.testing.assert_close(penalty.func(env) * penalty.weight, torch.tensor([-.06, -.06]))


def test_clearance_uses_observed_clock_and_archive_kernel_with_stop_gate():
    env, term = state_env()
    term.command[:, 0] = 1
    term.transition_mask[:] = False
    term.requested_velocity[:, 0] = .05
    env.scene['foot_height_scan'] = SimpleNamespace(data=SimpleNamespace(heights=torch.tensor([[.05, .09], [.05, .09]])))
    # Clock sin > 0 means foot_1 swings: 0.5*exp(-1), not foot_0's 0.5.
    env.episode_length_buf[:] = 50
    term.gait_frequency[:] = .25
    torch.testing.assert_close(rewards.biped_swing_clearance(env), torch.full((2,), .5 * torch.exp(torch.tensor(-1.)).item()))
    term.gait_frequency[:] = .75  # Change the same clock consumed by the actor.
    torch.testing.assert_close(rewards.biped_swing_clearance(env), torch.full((2,), .5))
    term.requested_velocity[:] = 0
    assert rewards.biped_swing_clearance(env).eq(0).all()
    term.requested_velocity[:, 0] = .05
    term.command[:, 0] = -1
    assert rewards.biped_swing_clearance(env).eq(0).all()
    term.command[:, 0] = 1
    term.transition_mask[:] = True
    assert rewards.biped_swing_clearance(env).eq(0).all()


def test_landing_keeps_first_contact_peak_force_units():
    env, term = state_env()
    term.command[:, 0] = 1
    term.transition_mask[:] = False
    env.physics_dt = .005
    force = torch.zeros(2, 2, 4, 3)
    force[:, 0, :, 2] = torch.tensor([1., 5., 3., 2.])
    force[:, 1, :, 2] = torch.tensor([2., 8., 4., 1.])
    env.scene['feet_ground_contact'] = SimpleNamespace(
        data=SimpleNamespace(force=force[:, :, -1], force_history=force),
        compute_first_contact=lambda dt: torch.tensor([[True, False], [False, True]]),
    )
    torch.testing.assert_close(rewards.biped_soft_landing(env), torch.tensor([5., 8.]))
    term.transition_mask[1] = True
    torch.testing.assert_close(rewards.biped_soft_landing(env), torch.tensor([5., 0.]))
