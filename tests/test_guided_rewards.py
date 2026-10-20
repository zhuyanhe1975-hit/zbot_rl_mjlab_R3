"""Archived transforms/crawl must not pay the walking-only power objective."""

from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.universal import guided_rewards as rewards
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg


def test_energy_and_rate_regularizers_match_their_archived_skill_domains(monkeypatch):
    term = SimpleNamespace(standing_mode=torch.tensor([True, False, False, True]),
                           transition_mask=torch.tensor([False, True, False, True]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    monkeypatch.setattr(rewards.walking, 'joint_energy_penalty', lambda env, energy_scale: torch.ones(4))
    monkeypatch.setattr(rewards, 'teacher_action_rate', lambda env: torch.tensor([5., 1., 1., 1.]))
    torch.testing.assert_close(rewards.guided_joint_energy(env), torch.tensor([1., 0., 0., 0.]))
    torch.testing.assert_close(rewards.guided_action_rate(env), torch.tensor([5., 0., 1., 0.]))


def test_small_nonzero_motion_has_no_competing_standstill_reward(monkeypatch):
    term = SimpleNamespace(requested_velocity=torch.tensor([[0., 0., 0.], [0., .008, 0.]]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    monkeypatch.setattr(rewards.mdp, 'standstill_reward', lambda env: torch.ones(2))
    torch.testing.assert_close(rewards.guided_standstill(env), torch.tensor([1., 0.]))


def test_guided_recipe_is_isolated_and_retains_walking_weights():
    base = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'
    original = load_env_cfg(base + '-Task-Routed')
    guided = load_env_cfg(base + '-Guided')
    assert {'mode_posture', 'transition_progress'} <= original.rewards.keys()
    assert not {'mode_posture', 'transition_progress'} & guided.rewards.keys()
    for name, term in guided.rewards.items():
        assert term.weight == original.rewards[name].weight
    command = guided.commands['transform']
    assert isinstance(command, PrivilegedOmniCommandCfg)
    assert command.training_task_fractions == (.25, .25, .25, .25)
    assert command.transition_timeout_s == 16.
    original_command = original.commands['transform']
    assert isinstance(original_command, PrivilegedOmniCommandCfg)
    assert original_command.transition_timeout_s == 8.
