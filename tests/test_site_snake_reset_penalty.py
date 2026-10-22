"""Failure resets carry one fixed cost, regardless of dt or overlapping causes."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.experiments.site_snake import (
    site_snake_env_cfg as _recipe_site_snake_env_cfg,
)


@pytest.mark.parametrize('step_dt', [.005, .02, .1])
def test_failure_reset_penalty_is_fixed_and_excludes_timeout_only(step_dt):
    env = SimpleNamespace(
        num_envs=6, device='cpu', step_dt=step_dt,
        tilt=torch.tensor([False, True, False, True, False, False]),
        collision=torch.tensor([False, False, True, True, False, True]),
        timeout=torch.tensor([False, False, False, False, True, True]),
    )
    env.termination_manager = TerminationManager({
        'site_tilt': TerminationTermCfg(func=lambda env: env.tilt),
        'self_collision': TerminationTermCfg(func=lambda env: env.collision),
        'timeout': TerminationTermCfg(func=lambda env: env.timeout, time_out=True),
    }, env)
    env.termination_manager.compute()
    term = _recipe_site_snake_env_cfg().rewards['failure_reset']
    contribution = term.weight * term.func(env) * step_dt
    torch.testing.assert_close(contribution, torch.tensor([0., -5., -5., -5., 0., -5.]))
    # Reading diagnostics twice must not consume the event or add another charge.
    torch.testing.assert_close(term.weight * term.func(env) * step_dt, contribution)
    # The subsequent, healthy control step has no termination cost.
    env.tilt.zero_()
    env.collision.zero_()
    env.timeout.zero_()
    env.termination_manager.compute()
    assert not term.func(env).any()


def test_training_timeout_triggers_at_1500_control_steps():
    cfg = _recipe_site_snake_env_cfg()
    cfg.episode_length_s = 30.
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    maximum = round(cfg.episode_length_s / step_dt)
    assert maximum == 1500
    env = SimpleNamespace(episode_length_buf=torch.tensor([1499, 1500, 4501]), max_episode_length=maximum)
    assert cfg.terminations['time_out'].func(env).tolist() == [False, True, True]
    assert 'time_out' not in _recipe_site_snake_env_cfg(play=True).terminations


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_training_timeout_resets_without_failure_penalty_and_preserves_other_robots():
    from mjlab.envs import ManagerBasedRlEnv

    cfg = _recipe_site_snake_env_cfg()
    cfg.episode_length_s = 30.
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset()
            assert env.max_episode_length == 1500
            robot = env.scene['robot']
            root = robot.data.default_root_state.clone()
            root[:, :3] = env.scene.env_origins + root.new_tensor([0., 0., .8])
            canonical = root.new_tensor([2**-.5, 0., -2**-.5, 0.])
            root[:, 3:7] = canonical
            root[3, 3:7] = quat_mul(root.new_tensor([2**-.5, 0., 2**-.5, 0.]), canonical)
            root[:, 7:] = 0.
            robot.write_root_state_to_sim(root)
            env.command_manager.get_term('transform').stop()
            env.episode_length_buf[:] = torch.tensor([1498, 1499, 100, 1499], device='cuda:0')
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(4, 6, device='cuda:0'))
            assert terminated.tolist() == [False, False, False, True]
            assert truncated.tolist() == [False, True, False, True]
            assert env.episode_length_buf.tolist() == [1499, 0, 101, 0]
            index = env.reward_manager.active_terms.index('failure_reset')
            torch.testing.assert_close(env.reward_manager._step_reward[:, index] * env.step_dt,
                                       root.new_tensor([0., 0., 0., -5.]))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            # Robot 0 now reaches the limit; the reset robots do not time out again.
            _, _, terminated, truncated, _ = env.step(torch.zeros(4, 6, device='cuda:0'))
            assert not terminated.any()
            assert truncated.tolist() == [True, False, False, False]
            assert env.episode_length_buf.tolist() == [0, 1, 102, 1]
            assert not env.reward_manager._step_reward[:, index].any()
    finally:
        env.close()
