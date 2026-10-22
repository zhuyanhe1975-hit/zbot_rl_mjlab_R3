"""Diagnostics count collected frames rather than episodes that happened to end."""

import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner

from zbot_rl_mjlab.experiments.omni_teacher import omni_teacher_ppo_cfg
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner, teacher_rollout_metrics


def algorithm(observations):
    return SimpleNamespace(storage=SimpleNamespace(observations={'actor': observations}))


def test_rollout_counts_are_not_reset_episode_counts():
    observations = torch.zeros(3, 10, 59)
    for phase, ids in enumerate((slice(0, 4), slice(4, 5), slice(5, 9), slice(9, 10))):
        observations[:, ids, 39 + phase] = 1.
    # Reset-heavy ascent episodes do not affect the actual 40/10/40/10 allocation.
    alg = algorithm(observations)
    alg.storage.dones = torch.zeros(3, 10)
    alg.storage.dones[:, 9] = 1.
    metrics = teacher_rollout_metrics(alg)
    for name, count in (('biped', 12), ('descending', 3), ('snake', 12), ('ascending', 3)):
        assert metrics[f'rollout_{name}_frames'] == count
        assert metrics[f'rollout_{name}_fraction'] == pytest.approx(count / 30)


def test_motion_metrics_use_effective_commands_and_selected_task_frames():
    observations = torch.zeros(1, 4, 59)
    observations[0, :2, 39] = 1.
    observations[0, 2:, 41] = 1.
    observations[0, 0, 43:46] = torch.tensor([.2, -.1, .4])
    observations[0, 0, 53:56] = torch.tensor([.1, -.05, -.2])
    observations[0, 1, 53:56] = 100.  # stopped biped is excluded
    observations[0, 2, 43:46] = torch.tensor([-.1, 0., 0.])
    observations[0, 2, 53:56] = torch.tensor([-.05, .02, .03])
    observations[0, 3, 56:59] = 1.  # pending request is not an effective command
    metrics = teacher_rollout_metrics(algorithm(observations))
    assert metrics['rollout_biped_active_frames'] == 1
    assert metrics['rollout_biped_velocity_rmse'] == pytest.approx(math.sqrt((.1**2 + .05**2 + .6**2) / 3))
    assert metrics['rollout_biped_vx_rmse'] == pytest.approx(.1)
    assert metrics['rollout_biped_signed_vx'] == pytest.approx(.1)
    assert metrics['rollout_biped_signed_vy'] == pytest.approx(.05)
    assert metrics['rollout_biped_signed_wz'] == pytest.approx(-.2)
    assert metrics['rollout_biped_planar_progress'] == pytest.approx((.1 * .2 + .05 * .1) / math.sqrt(.2**2 + .1**2))
    assert metrics['rollout_snake_active_frames'] == 1
    assert metrics['rollout_snake_planar_progress'] == pytest.approx(.05)
    assert metrics['rollout_snake_signed_vx'] == pytest.approx(.05)
    assert metrics['rollout_snake_signed_vy'] == 0.
    assert metrics['rollout_snake_wz_rmse'] == pytest.approx(.03)


def test_empty_task_subsets_return_finite_zero_metrics():
    observations = torch.zeros(2, 3, 59)
    observations[:, :, 42] = 1.
    metrics = teacher_rollout_metrics(algorithm(observations))
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics['rollout_ascending_fraction'] == 1.
    assert all(value == 0. for key, value in metrics.items() if 'ascending' not in key)


def test_yaw_only_commands_have_signed_yaw_and_no_planar_progress():
    observations = torch.zeros(1, 1, 59)
    observations[:, :, 41] = 1.
    observations[:, :, 45] = -.4
    observations[:, :, 55] = -.3
    metrics = teacher_rollout_metrics(algorithm(observations))
    assert metrics['rollout_snake_signed_wz'] == pytest.approx(.3)
    assert metrics['rollout_snake_planar_progress'] == 0.


@pytest.mark.parametrize('shape', [(2, 59), (2, 3, 58), (0, 3, 59), (2, 0, 59)])
def test_wrong_observation_schema_fails_clearly(shape):
    with pytest.raises(ValueError, match='teacher rollout metrics require'):
        teacher_rollout_metrics(algorithm(torch.zeros(shape)))


def test_missing_storage_is_the_only_mock_skip():
    assert teacher_rollout_metrics(SimpleNamespace()) == {}
    with pytest.raises(KeyError):
        teacher_rollout_metrics(SimpleNamespace(storage=SimpleNamespace(observations={})))


def test_runner_snapshots_metrics_before_update_clears_storage(monkeypatch, capsys):
    observations = torch.zeros(2, 3, 59)
    observations[:, :, 41] = 1.
    alg = algorithm(observations)

    def update():
        observations.zero_()
        return {'value': 2.}

    alg.update = update

    def base_init(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.alg = env, cfg, alg

    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', base_init)
    cfg = asdict(omni_teacher_ppo_cfg()) | {'console_mode': 'full'}
    env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=object()))
    OmniTeacherRunner(cast(Any, env), cfg, log_dir='run')
    losses = alg.update()
    assert losses['rollout_snake_fraction'] == 1.
    assert losses['rollout_snake_frames'] == 6.
    assert losses['teacher_ppo_iterations'] == 1.
    assert losses['value'] == 2.
    assert capsys.readouterr().out == ''


def test_scalar_and_site_layouts_preserve_all_task_and_motion_metrics():
    legacy = torch.zeros(2, 4, 59)
    legacy[..., 39:43] = torch.eye(4)
    legacy[..., 43:46] = torch.tensor([.2, -.1, .3])
    legacy[..., 53:56] = torch.tensor([.1, -.03, .2])
    compact = torch.cat((legacy[..., :39], torch.arange(4).expand(2, 4)[..., None], legacy[..., 43:]), dim=-1)
    assert compact.shape[-1] == 56
    # Large sentinels make decoding any extra site features as motion observable.
    legacy_site = torch.cat((
        compact[..., :46], torch.full((2, 4, 2), 11.), compact[..., 47:], torch.full((2, 4, 6), 99.),
    ), dim=-1)
    site = torch.cat((compact[..., :46], compact[..., 47:], torch.full((2, 4, 6), 99.)), dim=-1)
    assert legacy_site.shape[-1] == 63
    assert site.shape[-1] == 61
    expected = teacher_rollout_metrics(algorithm(legacy))
    assert teacher_rollout_metrics(algorithm(compact)) == expected
    assert teacher_rollout_metrics(algorithm(legacy_site)) == expected
    assert teacher_rollout_metrics(algorithm(site)) == expected
