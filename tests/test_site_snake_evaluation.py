"""Qualification rejects missing skills and world-axis shortcuts independently."""

import pytest
import torch

from zbot_rl_mjlab.site_snake_evaluation import (
    CASES,
    SPEEDS,
    SiteSnakeEvaluator,
    score_site_snake,
    targets,
)


def trace(replicas=1):
    velocity = targets(torch.empty(0)).repeat_interleave(replicas, 0).repeat(20, 1, 1)
    return velocity, torch.zeros(72 * replicas, dtype=torch.bool)


def test_complete_two_speed_four_heading_tracking_passes():
    metrics, report = score_site_snake(*trace())
    assert metrics['snake_omni_passed'] == 1.
    assert metrics['snake_omni_scenarios_passed'] == 9.
    assert len(report['cases']) == 72
    assert all(item['passed'] for item in report['cases'])


@pytest.mark.parametrize('fault', ('zero', 'world_x', 'wrong_sign', 'missing_yaw', 'low_speed', 'one_heading', 'stop_drift', 'nonfinite', 'failure', 'burst'))
def test_cannot_hide_missing_capability(fault):
    velocity, failed = trace()
    if fault == 'zero':
        velocity.zero_()
    elif fault == 'world_x':
        velocity[:] = torch.tensor([[.1, 0., 0.], [0., -.1, 0.], [-.1, 0., 0.], [0., .1, 0.]]).repeat(18, 1)
    elif fault == 'wrong_sign':
        velocity.neg_()
    elif fault == 'missing_yaw':
        velocity[..., 2] = 0
    elif fault == 'low_speed':
        velocity[:, 8:12] = 0
    elif fault == 'one_heading':
        velocity[:, 11] = 0
    elif fault == 'stop_drift':
        velocity[:, :8, 1] = .04
    elif fault == 'nonfinite':
        velocity[0, 8] = float('nan')
    elif fault == 'failure':
        failed[8] = True
    else:
        velocity[:10, 8] = 0
    assert score_site_snake(velocity, failed)[0]['snake_omni_passed'] == 0.


def test_each_speed_heading_requires_three_quarters_replicates():
    velocity, failed = trace(4)
    failed[32] = True
    assert score_site_snake(velocity, failed)[0]['snake_omni_passed'] == 1.
    failed[33] = True
    metrics, report = score_site_snake(velocity, failed)
    assert metrics['snake_omni_passed'] == 0.
    assert not report['cases'][8]['passed']


def test_stop_rejects_oscillatory_all_axis_drift():
    velocity, failed = trace()
    velocity[:10, :8, 2] = .3
    velocity[10:, :8, 2] = -.3
    assert score_site_snake(velocity, failed)[0]['snake_stop_success'] == 0.


def test_yaw_tracking_has_no_cumulative_heading_constraint():
    velocity, failed = trace()
    # Sixty seconds at the requested yaw rate would turn many complete circles.
    velocity = velocity.repeat(300, 1, 1)
    assert score_site_snake(velocity, failed)[0]['snake_omni_passed'] == 1.


@pytest.mark.parametrize('num_envs', (0, -1, True, 3.5))
def test_invalid_environment_count(num_envs):
    with pytest.raises(ValueError):
        SiteSnakeEvaluator(num_envs=num_envs)


def test_explicit_experiment_and_case_order():
    with pytest.raises(ValueError, match='independent site snake'):
        SiteSnakeEvaluator()
    assert CASES[:4] == tuple(('stop', 0, yaw) for yaw in (0, 90, 180, 270))


@pytest.mark.parametrize('shape', ((0, 72, 3), (10, 71, 3), (10, 72, 2)))
def test_invalid_trace_shape(shape):
    with pytest.raises(ValueError):
        score_site_snake(torch.empty(shape), torch.zeros(shape[1], dtype=torch.bool))


def test_real_call_protocol_preserves_rng_and_latches_early_failure(monkeypatch):
    import random
    from types import SimpleNamespace
    from typing import Any, cast

    import numpy as np
    from tensordict import TensorDict

    from zbot_rl_mjlab import site_snake_evaluation as evaluation

    events = []
    ticks = [0]
    requested = torch.zeros(72, 3)

    def set_velocity(x, y, z, ids):
        assert all(isinstance(value, float) for value in (x, y, z))
        requested[ids] = torch.tensor((x, y, z))
        events.append((ticks[0], ids[0].item(), x, y, z))

    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=set_velocity)
    obs = TensorDict({'actor': torch.zeros(72, 61)}, batch_size=[72])
    data = SimpleNamespace(root_link_pos_w=torch.zeros(72, 3), joint_pos=torch.zeros(72, 6))

    class Scene(dict):
        env_origins = torch.zeros(72, 3)

    env = SimpleNamespace(step_dt=.5, cfg=SimpleNamespace(site_tilt_limit=1.),
                          scene=Scene(robot=SimpleNamespace(data=data)),
                          observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))

    def step(actions):
        ticks[0] += 1
        return obs, torch.zeros(72), torch.zeros(72, dtype=torch.bool), {}

    wrapper = SimpleNamespace(reset=lambda: None, get_observations=lambda: obs, step=step)
    evaluator = SiteSnakeEvaluator(seconds=10., env_cfg=cast(Any, SimpleNamespace(site_snake_omni=True)))
    cast(Any, evaluator).env, cast(Any, evaluator).wrapper = env, wrapper
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: requested.clone())
    monkeypatch.setattr(evaluation.mdp, 'normalized_force', lambda env: torch.ones(72))
    monkeypatch.setattr(evaluation.site_snake, 'site_tilt', lambda env: torch.zeros(72))

    def collision(env):
        result = torch.zeros(72, dtype=torch.bool)
        if ticks[0] == 1:
            result[8] = True
        return result

    monkeypatch.setattr(evaluation.snake, 'self_collision', collision)

    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            assert not stochastic_output
            random.random()
            np.random.rand()
            return torch.rand(72, 6)

    actor = Actor().train()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert evaluator(actor)['snake_omni_passed'] == 0.
    assert evaluator.last_report['cases'][8]['failure_rate'] == 1.
    assert actor.training
    assert random.getstate() == python_rng
    assert np.array_equal(cast(Any, np.random.get_state())[1], cast(Any, numpy_rng)[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert len(events) == 144
    assert events[0] == (0, 0, pytest.approx(SPEEDS[0][0]), 0., 0.)
    assert events[72] == (2, 0, 0., 0., 0.)
    assert evaluator.last_report['duration_eligible']
    evaluator.seconds = 2.
    ticks[0] = 0
    monkeypatch.setattr(evaluation.snake, 'self_collision', lambda env: torch.zeros(72, dtype=torch.bool))
    assert evaluator(actor)['snake_omni_passed'] == 0.
    assert not evaluator.last_report['duration_eligible']


def test_reset_event_cache_survives_subsequent_real_command_reset(monkeypatch):
    from types import SimpleNamespace

    from mjlab.tasks.registry import load_env_cfg

    from zbot_rl_mjlab import site_snake_evaluation as evaluation
    from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg

    class Scene(dict):
        env_origins = torch.zeros(72, 3)

    root = torch.zeros(72, 13)
    root[:, 3] = 1.
    joints = torch.zeros(72, 6)
    robot = SimpleNamespace(
        data=SimpleNamespace(default_root_state=root.clone(), default_joint_pos=joints.clone()),
        write_joint_state_to_sim=lambda pos, vel, env_ids: joints.__setitem__(env_ids, pos),
        write_root_state_to_sim=lambda pos, env_ids: root.__setitem__(env_ids, pos))
    cfg = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher', play=True)
    env = SimpleNamespace(cfg=cfg, num_envs=72, device='cpu', step_dt=.02,
                          common_step_counter=0, scene=Scene(robot=robot))
    command = PrivilegedOmniCommandCfg(mode='snake', training=False, randomize_play_modes=False, resampling_time_range=(6., 12.)).build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    env._omni_play_standing = torch.ones(72, dtype=torch.bool)
    # This is the ManagerBasedRlEnv order: event reset, then command reset.
    evaluation.reset_site_snake_evaluation(env, None)
    command.reset(torch.arange(72))
    assert not env._omni_initial_standing.any()
    assert not env._omni_play_standing.any()
    assert command.phase.eq(2).all()
    assert command.current_morphology.eq(-1).all()
    assert not command.transition_mask.any()
    assert joints.eq(0).all()
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env, **kwargs: torch.ones(72, dtype=torch.bool))
    command.set_velocity(.1, .05, .3)
    env.common_step_counter = 1
    command._update_command(None)
    assert command.velocity_command[:, 0].gt(0).all()
    assert command.velocity_command[:, 1].gt(0).all()
    assert command.velocity_command[:, 2].gt(0).all()
