"""Do not transfer transformation survival incentives and failure fees into locomotion."""

from types import SimpleNamespace

import torch
from test_universal_task import state_env

from zbot_rl_mjlab.experiments.omni_teacher import (
    configure_teacher_rewards as _recipe_configure_teacher_rewards,
)
from zbot_rl_mjlab.experiments.universal import universal_env_cfg
from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal import teacher_rewards as rewards


def test_static_posture_and_potential_rewards_are_zero_for_both_motion_modes(monkeypatch):
    env, term = state_env()
    term.command[:, 0] = torch.tensor([1., -1.])
    term.transition_mask[:] = False
    monkeypatch.setattr(mdp, 'posture_reward', lambda env: torch.ones(2))
    monkeypatch.setattr(mdp, 'transition_progress_reward', lambda env: torch.full((2,), 100.))
    cfg = _recipe_configure_teacher_rewards(universal_env_cfg())
    for name in ('mode_posture', 'transition_progress'):
        assert cfg.rewards[name].func(env).eq(0).all()
    term.transition_mask[:] = torch.tensor([True, False])
    torch.testing.assert_close(rewards.transition_posture(env), torch.tensor([1., 0.]))
    torch.testing.assert_close(rewards.transition_progress(env), torch.tensor([100., 0.]))


def test_failure_fee_is_transform_only_and_does_not_change_termination_state():
    env, term = state_env()
    env.termination_manager = SimpleNamespace(terminated=torch.ones(2, dtype=torch.bool))
    term.transition_mask[:] = torch.tensor([False, True])
    cfg = _recipe_configure_teacher_rewards(universal_env_cfg())
    fee = cfg.rewards['reset_failure']
    torch.testing.assert_close(fee.func(env) * fee.weight * env.step_dt, torch.tensor([0., -10.]))
    assert env.termination_manager.terminated.all()
    env.termination_manager.terminated[:] = False
    assert fee.func(env).eq(0).all()
