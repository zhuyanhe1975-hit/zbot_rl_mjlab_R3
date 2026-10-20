"""Reward decomposition preserves the scalar objective and pre-reset identity."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.grouped_teacher import GroupedTeacherRunner
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunnerCfg
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.reward_groups import (
    GROUP_NAMES,
    REWARD_GROUPS,
    group_mask_from_tasks,
    install_reward_group_capture,
    reward_assignment,
    reward_group_mask,
)


def test_every_current_reward_has_exactly_one_group_and_total_is_preserved():
    cfg = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Wide-Biped')
    names = list(cfg.rewards)
    assert set(names) == {name for group in REWARD_GROUPS.values() for name in group}
    assignment = reward_assignment(names, device='cpu')
    assert assignment.shape == (23, 4)
    assert assignment.sum(-1).eq(1).all()
    rates = torch.randn(16, len(names))
    torch.testing.assert_close((rates @ assignment).sum(-1), rates.sum(-1), atol=2e-6, rtol=1e-5)


def test_new_experiment_preserves_biped_physics_rewards_and_hyperparameters():
    base = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'
    for play in (False, True):
        before = load_env_cfg(base + '-Wide-Biped', play=play)
        after = load_env_cfg(base + '-Multi-Critic', play=play)
        old_command, new_command = before.commands['transform'], after.commands['transform']
        assert isinstance(old_command, PrivilegedOmniCommandCfg)
        assert isinstance(new_command, PrivilegedOmniCommandCfg)
        assert old_command.mode == 'stand'
        assert new_command.mode == 'source'
        old_command.mode = 'source'
        left, right = asdict(before), asdict(after)
        left['scene']['terrain'].pop('spec_fn')
        right['scene']['terrain'].pop('spec_fn')
        assert left == right
    before, after = load_rl_cfg(base + '-Wide-Biped'), load_rl_cfg(base + '-Multi-Critic')
    assert isinstance(before, OmniTeacherRunnerCfg)
    assert isinstance(after, OmniTeacherRunnerCfg)
    assert before.experiment_name != after.experiment_name
    assert after.actor == before.actor
    before.experiment_name = after.experiment_name
    before.class_name = after.class_name
    before.algorithm.class_name = after.algorithm.class_name
    before.critic.class_name = after.critic.class_name
    assert asdict(before) == asdict(after)
    assert load_runner_cls(base + '-Multi-Critic') is GroupedTeacherRunner


def test_unknown_or_duplicate_rewards_fail_instead_of_silently_dropping_signal():
    with pytest.raises(ValueError, match='ungrouped'):
        reward_assignment(['new_reward'], device='cpu')
    with pytest.raises(ValueError, match='unique'):
        reward_assignment(['standstill', 'standstill'], device='cpu')


@pytest.mark.parametrize('dimension', [56, 59])
def test_semantic_activity_uses_task_not_nonzero_reward(dimension):
    obs = torch.zeros(4, dimension)
    if dimension == 56:
        obs[:, 39] = torch.arange(4)
    else:
        obs[:, 39:43] = torch.eye(4)
    assert reward_group_mask(obs).tolist() == [
        [True, False, True, True],
        [False, True, False, True],
        [True, False, False, True],
        [False, True, False, True],
    ]


@pytest.mark.parametrize('scale_by_dt', [False, True])
def test_capture_calls_stateful_reward_once_and_snapshots_before_task_reset(scale_by_dt):
    calls = []
    manager = SimpleNamespace(
        active_terms=['velocity_tracking', 'joint_standing', 'action_rate'],
        _scale_by_dt=scale_by_dt,
        _step_reward=torch.tensor([[2., 0., -.1], [0., -3., -.2]]),
    )

    def original(dt):
        calls.append(dt)
        return manager._step_reward.sum(-1) * (dt if scale_by_dt else 1.)

    manager.compute = original
    term = SimpleNamespace(current_task=torch.tensor([0, 3]))
    env = SimpleNamespace(
        reward_manager=manager, device='cpu', extras={},
        command_manager=SimpleNamespace(get_term=lambda _: term),
    )
    install_reward_group_capture(env)
    install_reward_group_capture(env)
    total = manager.compute(.02)
    groups = env.extras['teacher_reward_groups']
    expected = torch.tensor([[2., 0., 0., -.1], [0., -3., 0., -.2]])
    torch.testing.assert_close(groups, expected * (.02 if scale_by_dt else 1.))
    torch.testing.assert_close(groups.sum(-1), total)
    assert calls == [.02]
    term.current_task[:] = 2
    manager._step_reward.zero_()
    # Env auto-reset cannot overwrite the reward or task identity for this step.
    assert env.extras['teacher_reward_group_mask'].tolist() == [
        [True, False, True, True], [False, True, False, True],
    ]
    assert groups.abs().sum() > 0
    assert GROUP_NAMES == ('motion_tracking', 'configuration', 'contact_gait', 'regularization')
    assert group_mask_from_tasks(term.current_task)[:, 0].all()
