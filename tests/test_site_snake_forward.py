"""Forward ablation changes command exposure and exactly two tracking weights."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from test_site_snake_task import reward_env

from zbot_rl_mjlab.experiments.site_snake import (
    site_forward_env_cfg as _recipe_site_forward_env_cfg,
)
from zbot_rl_mjlab.experiments.site_snake import (
    site_snake_env_cfg as _recipe_site_snake_env_cfg,
)
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunnerCfg
from zbot_rl_mjlab.site_forward_teacher import (
    ForwardSnakeTeacherRunner,
    RelaxedForwardSnakeTeacherRunner,
)
from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake import site_forward as forward
from zbot_rl_mjlab.tasks.snake import site_omni as omni

BASE = 'Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher'
RELAXED = 'Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher'


@pytest.mark.parametrize('task,relaxed,runner', [
    (BASE, False, ForwardSnakeTeacherRunner),
    (RELAXED, True, RelaxedForwardSnakeTeacherRunner),
])
@pytest.mark.parametrize('play', [False, True])
def test_registered_pair_preserves_physics_and_separates_checkpoints(task, relaxed, runner, play):
    cfg = load_env_cfg(task, play=play)
    original = _recipe_site_snake_env_cfg(play)
    assert asdict(cfg.actions['joint_pos']) == asdict(original.actions['joint_pos'])
    assert cfg.sim == original.sim and cfg.terminations == original.terminations
    assert cast(forward.ForwardSnakeCommandCfg, cfg.commands['transform']).play_velocity == (.1, 0., 0.)
    assert cfg.rewards['track_forward_velocity'].weight == 4.
    assert cfg.rewards['track_lateral_velocity'].weight == (0. if relaxed else 4.)
    assert cfg.rewards['track_angular_velocity'].weight == (0. if relaxed else 1.)
    assert load_runner_cls(task) is runner
    assert cast(OmniTeacherRunnerCfg, load_rl_cfg(task)).class_name.endswith(runner.__name__)
    assert runner._metadata_schema()['reward_recipe'] != SiteSnakeTeacherRunner._metadata_schema()['reward_recipe']


def test_sampling_is_forward_only_and_respects_partial_ids_and_manual_control():
    term = object.__new__(forward.ForwardSnakeCommand)
    term.cfg = forward.ForwardSnakeCommandCfg()
    term.manual_control = torch.tensor([False, True, False, False])
    term.requested_velocity = torch.full((4, 3), -.5)
    term._sample_velocity(torch.tensor([0, 1, 2]))
    torch.testing.assert_close(term.requested_velocity[[0, 2]], torch.tensor([[.1, 0., 0.]]*2))
    torch.testing.assert_close(term.requested_velocity[[1, 3]], torch.full((2, 3), -.5))


def test_baseline_split_preserves_total_and_relaxed_exposes_exploration_cost(monkeypatch):
    env = reward_env(monkeypatch, [[.05, .04, .2], [0., 0., 0.], [-.05, 0., 0.]], [[.1, 0., 0.]]*3)
    original = _recipe_site_snake_env_cfg()
    original.rewards['track_forward_velocity'].weight = 4.
    original.rewards['track_lateral_velocity'].weight = 4.
    original.rewards['track_angular_velocity'].weight = 1.
    baseline, relaxed = _recipe_site_forward_env_cfg(), _recipe_site_forward_env_cfg(relaxed=True)
    def total(cfg):
        return torch.stack([term.weight*term.func(env, **term.params) for name, term in cfg.rewards.items() if name.startswith('track_')]).sum(0)
    torch.testing.assert_close(total(baseline), total(original))
    limits = omni.mdp.command_term(env).cfg.velocity_limits
    torch.testing.assert_close(total(relaxed), torch.tensor([0., -1., -2.])*4)
    assert total(baseline)[0] < total(relaxed)[0]
    torch.testing.assert_close(forward.tracking_component(env, 1), torch.tensor([-.04, 0., 0.])/limits[1])


def test_saturation_metric_distinguishes_stillness_from_limit_contact():
    cfg = _recipe_site_forward_env_cfg()
    limit = cfg.actions['joint_pos'].snake_target_limit
    action = SimpleNamespace(commanded_target=torch.tensor([[0.]*6, [limit]*6]))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: action),
                          cfg=cfg)
    torch.testing.assert_close(forward.target_hard_limit_fraction(env), torch.tensor([0., 1.]))


@pytest.mark.parametrize('runner,other', [
    (ForwardSnakeTeacherRunner, RelaxedForwardSnakeTeacherRunner),
    (RelaxedForwardSnakeTeacherRunner, ForwardSnakeTeacherRunner),
])
def test_checkpoint_rejects_other_ablation_and_omni(runner, other, tmp_path):
    import math

    metadata = runner._metadata_schema() | {
        'velocity_limits': (.2, .2, .5),
        'joint_target_limit': math.pi/2, 'max_target_speed': 2*math.pi,
    }
    saved = {'infos': {'privileged_omni_teacher': metadata}}
    assert runner._validate_checkpoint(saved) is metadata
    for incompatible in (other, SiteSnakeTeacherRunner):
        with pytest.raises(ValueError, match='compatible'):
            incompatible._validate_checkpoint(saved)
    path = tmp_path/'forward.pt'
    torch.save(saved, path)
    cfg = _recipe_site_forward_env_cfg(play=True)
    runner.prepare_playback_env_cfg(cfg, path)
    assert cast(forward.ForwardSnakeCommandCfg, cfg.commands['transform']).velocity_limits == (.2, .2, .5)


def test_main_task_axes_have_independent_weights_and_equal_weights_match_linear_total(monkeypatch):
    env = reward_env(monkeypatch, [[.05, -.03, 0.], [-.02, .01, 0.]], [[.1, -.1, 0.]]*2)
    cfg = _recipe_site_snake_env_cfg()
    x, y = cfg.rewards['track_forward_velocity'], cfg.rewards['track_lateral_velocity']
    x.weight = y.weight = 4.
    rx, ry = x.func(env, **x.params), y.func(env, **y.params)
    torch.testing.assert_close(x.weight*rx+y.weight*ry, 4*omni.linear_tracking(env))
    x.weight = 8.
    torch.testing.assert_close(x.weight*rx+y.weight*ry, 8*rx+4*ry)
    assert y.weight == 4.
    y.weight = 0.
    torch.testing.assert_close(x.weight*rx+y.weight*ry, 8*rx)
    assert x.weight == 8.
