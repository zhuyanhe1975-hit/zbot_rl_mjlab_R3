"""The editable episode duration must actually truncate site-snake training."""
from types import SimpleNamespace

import torch
from mjlab.tasks.registry import load_env_cfg

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'


def test_training_truncates_at_exactly_fifteen_seconds():
    cfg = load_env_cfg(TASK)
    assert cfg.episode_length_s == 15.
    assert cfg.auto_reset
    steps = round(cfg.episode_length_s / (cfg.decimation * cfg.sim.mujoco.timestep))
    assert steps == 750
    term = cfg.terminations['time_out']
    assert term.time_out
    env = SimpleNamespace(episode_length_buf=torch.tensor([0, steps - 1, steps, steps + 1]), max_episode_length=steps)
    assert term.func(env).tolist() == [False, False, True, True]


def test_interactive_play_remains_manually_reset():
    cfg = load_env_cfg(TASK, play=True)
    assert cfg.episode_length_s == 15.
    assert not cfg.auto_reset
    assert 'time_out' not in cfg.terminations
