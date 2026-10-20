"""Acceptance rejects world-axis tricks, missing axes, and brief bursts."""

import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from zbot_rl_mjlab import universal_omni_evaluation as evaluation


def trace():
    velocity = torch.tensor([item[2] for item in evaluation.SCENARIOS]).repeat_interleave(4, 0)
    return velocity.repeat(20, 1, 1), torch.ones(20, 80, dtype=torch.bool), torch.ones(80, dtype=torch.bool)


def test_every_signed_axis_and_transition_passes():
    result = evaluation.evaluate_omni_capabilities(*trace())
    assert result['validation_passed'] == 1.
    assert all(result[f'{name}_success'] == 1. for name, _, _ in evaluation.SCENARIOS)


@pytest.mark.parametrize('failure', ('zero', 'world_x', 'wrong_sign', 'missing_y', 'missing_yaw', 'stop_drift', 'spike', 'unready', 'unsafe', 'one_yaw'))
def test_missing_capability_cannot_hide_in_other_groups(failure):
    velocity, ready, finite = trace()
    if failure == 'zero':
        velocity.zero_()
    elif failure == 'world_x':
        # Fixed world +X appears as +X,-Y,-X,+Y in the four site headings.
        velocity[:] = torch.tensor([[.08, 0., 0.], [0., -.08, 0.], [-.08, 0., 0.], [0., .08, 0.]]).repeat(20, 1)
    elif failure == 'wrong_sign':
        velocity.neg_()
    elif failure == 'missing_y':
        velocity[..., 1] = 0
    elif failure == 'missing_yaw':
        velocity[..., 2] = 0
    elif failure == 'stop_drift':
        velocity[:, :4, 0] = .08
    elif failure == 'spike':
        velocity[0] *= 20
        velocity[1:] = 0
    elif failure == 'unready':
        ready[:, 8:12] = False
    elif failure == 'unsafe':
        finite[8:12] = False
    elif failure == 'one_yaw':
        # Three good headings do not conceal complete failure at the fourth.
        velocity[:, 7, 0] = 0
    assert evaluation.evaluate_omni_capabilities(velocity, ready, finite)['validation_passed'] == 0.


def test_fall_and_nonfinite_fail():
    velocity, ready, finite = trace()
    no_fall = finite.clone()
    no_fall[:4] = False
    assert evaluation.evaluate_omni_capabilities(velocity, ready, finite, no_fall=no_fall)['validation_passed'] == 0.
    velocity[:, 4:8] = float('nan')
    assert evaluation.evaluate_omni_capabilities(velocity, ready, finite)['validation_passed'] == 0.


def test_keyboard_changes_preserve_physics_and_observation_history(monkeypatch):
    events = []
    obs = TensorDict({'actor': torch.zeros(80, 3)}, batch_size=[80])
    tick = [0]

    def record(kind):
        return lambda *args, **kwargs: events.append((tick[0], kind, args))

    command = SimpleNamespace(cfg=SimpleNamespace(initialize_matched_motion=False), set_target=record('target'), set_velocity=record('velocity'))
    env = SimpleNamespace(step_dt=.5, observation_manager=SimpleNamespace(reset=record('history_reset'), compute=record('history_compute')))

    def step(actions):
        tick[0] += 1
        return obs, torch.zeros(80), None, None

    wrapper = SimpleNamespace(reset=record('physics_reset'), get_observations=lambda: obs, step=step)
    evaluator = evaluation.UniversalOmniEvaluator(seconds=2.)
    cast(Any, evaluator).env, cast(Any, evaluator).wrapper = env, wrapper
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: trace()[0][0])
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(80, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(80, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'base_height', lambda env: torch.full((80,), .3))

    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.rand(80, 6)

    actor = Actor().train()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert evaluator(actor)['validation_passed'] == 1.
    assert actor.training and evaluator.foundation_samples is None
    assert random.getstate() == python_rng
    assert np.array_equal(cast(Any, np.random.get_state())[1], cast(Any, numpy_rng)[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for kind in ('physics_reset', 'history_reset', 'history_compute'):
        assert [(t, k) for t, k, _ in events if k == kind] == [(0, kind)]
    assert len([event for event in events if event[:2] == (2, 'velocity')]) == 20
    assert len([event for event in events if event[:2] == (2, 'target')]) == 20
    assert any(t == 2 and k == 'velocity' and args[:3] == (-.08, -.05, -.3) for t, k, args in events)
    # The final two scenarios change real target mode at the keyboard event.
    targets = [(t, args[0]) for t, k, args in events if k == 'target']
    assert targets[18:20] == [(0, 'stand'), (0, 'snake')]
    assert targets[-2:] == [(2, 'snake'), (2, 'stand')]


def test_config_retains_site_semantics_and_independent_reset():
    from zbot_rl_mjlab.tasks.universal.env_cfg import universal_env_cfg

    source = universal_env_cfg()
    evaluator = evaluation.UniversalOmniEvaluator(env_cfg=source, num_envs=16)
    assert evaluator.num_envs == 80
    assert evaluator.env_cfg is not source
    assert evaluator.foundation_samples is None


@pytest.mark.parametrize('num_envs', (0, -1, True, 1.5))
def test_bad_environment_count(num_envs):
    with pytest.raises(ValueError):
        evaluation.UniversalOmniEvaluator(num_envs=num_envs)


def test_reset_uses_four_yaws_and_collision_envelope(monkeypatch):
    from mjlab.utils.lab_api.math import quat_apply

    written = {}
    root = torch.zeros(80, 13)
    root[:, 3] = 1
    origins = torch.zeros(80, 3)
    origins[:, 0] = torch.arange(80) * 2.

    def joint_write(q, dq, env_ids):
        written['q'] = q
        assert not dq.any()

    def root_write(states, env_ids):
        written['root'] = states

    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=root), write_joint_state_to_sim=joint_write, write_root_state_to_sim=root_write)
    env = SimpleNamespace(num_envs=80, device='cpu', scene={'robot': robot})

    class Scene(dict):
        env_origins = origins

    env.scene = Scene(robot=robot)
    # Support box is sufficient to verify arbitrary quaternion clearance math.
    monkeypatch.setattr(evaluation, 'collision_radial_envelope', lambda pose: ([(-.1, -.1), (-.1, .1), (.1, -.1), (.1, .1)], -.2, .2))
    evaluation.reset_omni_evaluation_pose(env, None)
    states = written['root']
    torch.testing.assert_close(states[:, 0], origins[:, 0])
    assert not states[:, 7:].any()
    assert torch.all(states[:, 2] > 0)
    axis = torch.zeros(80, 3)
    axis[:, 0] = 1
    axis[36:72] = torch.tensor([0., 0., -1.])
    axis[76:] = torch.tensor([0., 0., -1.])
    direction = quat_apply(states[:, 3:7], axis)
    expected = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]).repeat(20, 1)
    torch.testing.assert_close(direction[:, :2], expected, atol=1e-6, rtol=1e-6)
    assert not written['q'][36:72].any()
    assert not written['q'][76:].any()
    assert written['q'][72:76].abs().sum() > 0
