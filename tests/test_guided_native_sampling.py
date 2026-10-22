"""Archived motion command distributions stay distinct in guided distillation."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml

from zbot_rl_mjlab.tasks.universal.guided_transitions import GuidedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.teacher_sampling import sample_motion_commands


def make_command(n=64, **overrides):
    cfg = GuidedOmniCommandCfg(
        resampling_time_range=(6., 12.), task_observation_encoding='scalar',
        aligned_teacher_sampling=True, **overrides,
    )
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0)
    return cfg.build(env)


def test_defaults_match_native_archived_ranges_and_timing():
    root = Path(__file__).resolve().parents[1] / 'bests'
    snake = yaml.load((root / 'snake_site_omni_2026-10-20_iter599/env.yaml').read_text(), Loader=yaml.BaseLoader)['commands']['transform']
    biped_env = yaml.load((root / 'omni_velocity_energy_2026-10-10_iter499/env.yaml').read_text(), Loader=yaml.BaseLoader)
    biped = biped_env['commands']['target_speed']
    cfg = cast(GuidedOmniCommandCfg, make_command().cfg)
    assert cfg.snake_velocity_limits == tuple(map(float, snake['velocity_limits']))
    assert cfg.snake_resampling_time_range == tuple(map(float, snake['resampling_time_range']))
    assert cfg.snake_motion_axis_min_fraction == float(snake['motion_axis_min_fraction'])
    assert cfg.snake_gait_frequency == float(snake['snake_gait_frequency'])
    assert cfg.smoothing_s == float(snake['smoothing_s'])
    assert cfg.biped_resampling_time_range == tuple(map(float, biped['resampling_time_range']))
    assert cfg.biped_velocity_ranges == tuple(tuple(map(float, biped['ranges'][key])) for key in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'))
    assert cfg.biped_gait_frequency_range == (float(biped_env['step_frequency_min']), float(biped_env['step_frequency_max']))
    assert float(biped['rel_standing_envs']) == 0


def test_biped_independent_continuous_uniform_has_no_stop_or_axis_categories():
    command = make_command(12000)
    torch.manual_seed(38)
    command._sample_velocity(torch.arange(command.num_envs))
    samples = command.requested_velocity
    bounds = samples.new_tensor(cast(GuidedOmniCommandCfg, command.cfg).biped_velocity_ranges)
    normalized = (samples - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    assert torch.all((normalized >= 0) & (normalized <= 1))
    assert torch.all(normalized.amin(0) < .002)
    assert torch.all(normalized.amax(0) > .998)
    torch.testing.assert_close(normalized.mean(0), torch.full((3,), .5), atol=.012, rtol=0)
    assert not (samples == 0).any()
    assert command.gait_frequency.min() >= .25
    assert command.gait_frequency.max() <= 1.


def test_snake_reuses_original_sampler_exactly_with_native_caps():
    command = make_command(12000)
    command._command[:, 0] = -1
    ids = torch.arange(command.num_envs)
    torch.manual_seed(19)
    expected = sample_motion_commands(torch.zeros(command.num_envs, dtype=torch.bool), (.2, .2, .5), axis_min_fraction=.4, stop_probability=1 / 8, x_only_probability=None)
    torch.manual_seed(19)
    command._sample_velocity(ids)
    torch.testing.assert_close(command.requested_velocity, expected, atol=0, rtol=0)
    assert .11 < float((expected == 0).all(-1).float().mean()) < .14
    single = (expected != 0).sum(-1) == 1
    normalized = expected.abs() / expected.new_tensor((.2, .2, .5))
    assert normalized[single].amax(-1).min() >= .4
    assert normalized.max() <= 1
    assert torch.all(command.gait_frequency == .7)


def test_per_role_timers_and_transition_zero_preserve_manual_control():
    command = make_command()
    command.manual_control[0] = True
    command.time_left[0] = torch.inf
    command.requested_velocity[0] = .123
    command._resample(torch.arange(command.num_envs))
    roles = command.training_roles(torch.arange(command.num_envs))
    automatic = ~command.manual_control
    standing = command.standing_mode & automatic
    snake = ~command.standing_mode & automatic
    assert torch.all((command.time_left[standing] >= 5) & (command.time_left[standing] <= 10))
    assert torch.all((command.time_left[snake] >= 5) & (command.time_left[snake] <= 9))
    assert (command.requested_velocity[roles >= 2] == 0).all()
    assert command.time_left[0].isinf()
    torch.testing.assert_close(command.requested_velocity[0], torch.full((3,), .123))
    task = command.current_task.clone()
    command._resample(torch.arange(command.num_envs))
    torch.testing.assert_close(command.current_task, task)


def test_manual_velocity_uses_target_mode_bounds_and_stays_pending_during_transition():
    command = make_command(2)
    command._command[:, 0] = torch.tensor([1., -1.])
    command.set_velocity(9, -9, 9)
    torch.testing.assert_close(command.requested_velocity, torch.tensor([[.5, -.2, .5], [.2, -.2, .5]]))
    command.set_velocity(-9, 9, -9)
    torch.testing.assert_close(command.requested_velocity, torch.tensor([[-.3, .2, -.5], [-.2, .2, -.5]]))
    assert command.manual_control.all() and command.time_left.isinf().all()
    assert command.transition_mask.all()
    assert (command.velocity_command == 0).all()


def test_legacy_opt_out_keeps_shared_manual_clipping():
    command = make_command(2)
    command.cfg.velocity_limits = (.15, .08, .4)
    cast(GuidedOmniCommandCfg, command.cfg).aligned_teacher_sampling = False
    command.set_velocity(1, 1, 1)
    torch.testing.assert_close(command.requested_velocity, torch.tensor([[.15, .08, .4]]).expand(2, -1))


@pytest.mark.parametrize('overrides', [
    {'biped_velocity_ranges': ((0., 1.),)},
    {'biped_velocity_ranges': ((1., -1.), (-.2, .2), (-.5, .5))},
    {'snake_velocity_limits': (.2, float('nan'), .5)},
    {'biped_resampling_time_range': (0., 10.)},
    {'snake_resampling_time_range': (9., 5.)},
    {'snake_motion_axis_min_fraction': 0.},
    {'snake_motion_stop_probability': 1.1},
])
def test_invalid_native_sampling_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        make_command(**overrides)


def test_native_biped_is_direct_snake_remains_smoothed_and_partial_updates_unchanged(monkeypatch):
    from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommand

    command = make_command(3)
    command._command[:, 0] = torch.tensor([1., -1., 1.])
    command.task_fsm.task_state[:] = torch.tensor([0, 2, 3])
    command.transition_mask[:] = torch.tensor([False, False, True])
    command.requested_velocity[:] = torch.tensor([.2, .1, .3])

    def inherited_update(self, env_ids):
        if env_ids is None:
            target = torch.where(self.transition_mask[:, None], 0., self.requested_velocity)
            alpha = self._env.step_dt / self.cfg.smoothing_s
            self.velocity_command.add_(alpha * (target - self.velocity_command))

    monkeypatch.setattr(PrivilegedOmniCommand, '_update_command', inherited_update)
    command._update_command(torch.tensor([0]))
    assert (command.velocity_command == 0).all()
    command._update_command(None)
    torch.testing.assert_close(command.velocity_command[0], command.requested_velocity[0])
    torch.testing.assert_close(command.velocity_command[1], command.requested_velocity[1] * (.02 / .25))
    assert (command.velocity_command[2] == 0).all()


def test_biped_frequency_is_sampled_per_episode_not_velocity_resample(monkeypatch):
    from zbot_rl_mjlab.tasks.universal import mdp

    command = make_command()
    ids = torch.arange(command.num_envs)
    # Command.reset uses physical initial-mode assignments supplied by reset events.
    command._env._omni_initial_standing = torch.ones(command.num_envs, dtype=torch.bool)
    monkeypatch.setattr(mdp, 'configuration_ready', lambda _, relaxed=False: torch.ones(command.num_envs, dtype=torch.bool))
    monkeypatch.setattr(command, '_update_command', lambda env_ids: None)
    torch.manual_seed(167)
    command.reset(ids)
    frequency = command._biped_gait_frequency.clone()
    assert frequency.min() >= .25 and frequency.max() <= 1.
    assert frequency.unique().numel() > 1
    command._sample_velocity(ids)
    command._resample(ids)
    command._sample_velocity(ids)
    torch.testing.assert_close(command._biped_gait_frequency, frequency, atol=0, rtol=0)
    command.reset(ids)
    refreshed = command._biped_gait_frequency
    assert refreshed.min() >= .25 and refreshed.max() <= 1.
    assert not torch.equal(refreshed, frequency)
