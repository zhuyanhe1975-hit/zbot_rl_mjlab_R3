"""The iter599 task must retain its archived policy inputs and training recipe."""
import math
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import yaml
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.archived_site_snake import ArchivedSiteSnakeTeacherRunner
from zbot_rl_mjlab.experiments.cards import load_card
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunnerCfg
from zbot_rl_mjlab.tasks.snake.site_omni import SiteSnakeCommandCfg, SiteSnakeEnvCfg
from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal.privileged import privileged_state

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'
ARCHIVE = Path(__file__).resolve().parents[1] / 'bests/snake_site_omni_2026-10-20_iter599'


@pytest.mark.parametrize('play', [False, True])
def test_restored_recipe_matches_archived_configuration(play):
    cfg = cast(SiteSnakeEnvCfg, load_env_cfg(TASK, play=play))
    agent = cast(OmniTeacherRunnerCfg, load_rl_cfg(TASK))
    action = cast(SnakeCrawlJointTargetRateActionCfg, cfg.actions['joint_pos'])
    command = cast(SiteSnakeCommandCfg, cfg.commands['transform'])
    saved = yaml.load((ARCHIVE / 'env.yaml').read_text(), Loader=yaml.BaseLoader)
    assert cfg.archive_site_checkpoint
    assert cfg.observations['actor'].terms['privileged'].func is privileged_state
    assert agent.actor.distribution_cfg == {'class_name': 'BetaDistribution', 'action_range': (-1., 1.)}
    assert tuple(agent.actor.hidden_dims) == (256, 128)
    assert agent.algorithm.learning_rate == .001 and agent.algorithm.entropy_coef == .002
    assert set(cfg.rewards) == (set(saved['rewards']) - {'track_linear_velocity'}) | {
        'track_forward_velocity', 'track_lateral_velocity', 'wz_velocity', 'site_tilt_angular_velocity'}
    card = load_card('snake_site_omni_teacher')
    for name in set(saved['rewards']) - {'track_linear_velocity'}:
        assert cfg.rewards[name].weight == card['rewards'][name]['weight']
    assert action.target_limit == float(saved['actions']['joint_pos']['target_limit'])
    assert action.max_target_speed == float(saved['actions']['joint_pos']['max_target_speed'])
    assert command.velocity_limits == tuple(float(v) for v in saved['commands']['transform']['velocity_limits'])
    assert cfg.auto_reset is not play
    assert set(cfg.terminations) == (set() if play else {'self_collision', 'site_tilt', 'time_out'})
    state = torch.load(ARCHIVE / 'model_599.pt', map_location='cpu', weights_only=False)
    ArchivedSiteSnakeTeacherRunner._validate_checkpoint(state)


def test_archived_heading_input_survives_site_frame(monkeypatch):
    term = SimpleNamespace(velocity_command=torch.zeros(1, 3), initial_heading=torch.tensor([[1., 0.]]))
    env = SimpleNamespace(cfg=SimpleNamespace(site_snake_omni=True, archive_site_checkpoint=True))
    monkeypatch.setattr(mdp, 'command_term', lambda _: term)
    monkeypatch.setattr(mdp, 'heading', lambda _: torch.tensor([[0., 1.]]))
    torch.testing.assert_close(mdp.heading_error(env), torch.tensor([-math.pi / 2]))
    env.cfg.archive_site_checkpoint = False
    torch.testing.assert_close(mdp.heading_error(env), torch.zeros(1))
