"""Bounded CPU DAgger tests: collection, labels, optimization and resumption."""

import copy
from typing import Any, cast

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn

from zbot_rl_mjlab.gaussian_action_models import expand_mirror
from zbot_rl_mjlab.student import TASK_NAMES, StudentDAgger
from zbot_rl_mjlab.student_models import (
    HistoryGaussianStudentMLP,
    HistoryTaskRoutedNetwork,
)


class FrozenTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.1), requires_grad=False)
        self.queries = []

    def forward(self, obs):
        raw = obs['teacher']
        self.queries.append(raw.clone())
        first = raw[:, :3] * self.scale + .7
        full = torch.cat((first, first + .2), -1)
        tasks = raw[:, 39].round().long()
        return torch.where(((tasks == 1) | (tasks == 3))[:, None], expand_mirror(first), full)


def observations(tasks=(0, 1, 2, 3) * 2) -> Any:
    raw = torch.randn(len(tasks), 56) * .1
    raw[:, 0] = torch.arange(len(tasks)).float()
    raw[:, 39] = torch.tensor(tasks).float()
    history = raw[:, None].expand(-1, 5, -1).clone()
    return TensorDict({'actor': history.flatten(1), 'teacher': raw.clone()}, batch_size=[len(tasks)])


def algorithm(obs=None, **kwargs) -> Any:
    obs = observations() if obs is None else obs
    sets = {'actor': ['actor'], 'critic': ['teacher']}
    actor = HistoryGaussianStudentMLP(obs, sets, 'actor', 6, hidden_dims=(8,), obs_normalization=True)
    critic = MLPModel(obs, sets, 'critic', 1, hidden_dims=(8,))
    storage = RolloutStorage('rl', len(obs), 2, obs, [6])
    defaults = {'dagger_updates': 1, 'dagger_batch_size': 128, 'dagger_max_samples_per_task': 128,
                'dagger_replay_iterations': 1, 'schedule': 'fixed',
                'teacher_warmup_iterations': 0, 'teacher_mix_iterations': 0,
                'dagger_anchor_fraction': 0.}
    alg: Any = StudentDAgger(actor, critic, storage, **(defaults | kwargs))
    alg.teacher = FrozenTeacher().eval()
    return alg


def effective_labels(alg, obs):
    labels = alg.teacher(obs)
    return torch.where((obs['teacher'][:, 39].round() == 0)[:, None], labels, labels.clamp(-1, 1))


def collect(alg, obs):
    with torch.inference_mode():
        action = alg.act(obs)
        alg.process_env_step(obs, torch.zeros(len(obs)), torch.zeros(len(obs)), {})
    return action


def state_equal(left, right):
    return all(torch.equal(value, right[key]) for key, value in left.items())


def test_collection_executes_student_mean_and_copies_prestep_state():
    obs = observations()
    original = obs.clone()
    alg = algorithm(obs)
    expected = alg.actor(obs).detach()
    action = collect(alg, obs)
    assert torch.equal(action, expected)
    assert not alg.teacher.queries
    obs['actor'].fill_(77)
    obs['teacher'].fill_(88)
    assert torch.equal(alg.storage.observations[0]['actor'], original['actor'])
    assert torch.equal(alg.storage.observations[0]['teacher'], original['teacher'])
    teacher_before = copy.deepcopy(alg.teacher.state_dict())
    alg.update()
    replay = alg.replay[0]
    raw = torch.cat(alg.teacher.queries)
    assert torch.equal(replay['tasks'], raw[:, 39].round().long())
    assert torch.equal(replay['actor'].reshape(-1, 5, 56)[:, -1, 0], raw[:, 0])
    labels = effective_labels(alg, TensorDict({'teacher': raw}, batch_size=[len(raw)]))
    assert torch.equal(replay['labels'], labels)
    assert not torch.equal(replay['labels'], action)
    assert state_equal(teacher_before, alg.teacher.state_dict())
    assert all(parameter.grad is None for parameter in alg.teacher.parameters())


def test_all_tasks_optimize_intrinsic_dims_and_keep_critic_and_std_frozen():
    obs = observations((0,) * 17 + (1,) * 2 + (2,) * 3 + (3,))
    alg = algorithm(obs)
    actor = cast(HistoryGaussianStudentMLP, alg.actor)
    router = cast(HistoryTaskRoutedNetwork, actor.mlp)
    before = [copy.deepcopy(head.state_dict()) for head in router.heads]
    critic_before = copy.deepcopy(alg.critic.state_dict())
    assert actor.distribution is not None
    std_before = copy.deepcopy(actor.distribution.state_dict())
    actor.update_normalization(obs)
    with torch.no_grad():
        expected = []
        labels = effective_labels(alg, obs)
        prediction = actor(obs)
        for task in range(4):
            mask = obs['teacher'][:, 39].round() == task
            width = 3 if task in (1, 3) else 6
            expected.append((prediction[mask, :width] - labels[mask, :width]).square().mean().item())
    collect(alg, obs)
    metrics = alg.update()
    for task, name in enumerate(TASK_NAMES):
        assert metrics[f'dagger_{name}_mse'] == pytest.approx(expected[task], rel=1e-5)
        assert not state_equal(before[task], router.heads[task].state_dict())
    assert [metrics[f'rollout_{name}_frames'] for name in TASK_NAMES] == [17, 2, 3, 1]
    assert state_equal(critic_before, alg.critic.state_dict())
    assert state_equal(std_before, actor.distribution.state_dict())
    assert metrics['student_updates'] == 1
    assert alg.storage.step == 0
    assert all(not batch['labels'].is_inference() for batch in alg.replay)


def test_inactive_heads_and_adam_momentum_remain_unchanged():
    alg = algorithm()
    collect(alg, observations())
    alg.update()
    router = cast(HistoryTaskRoutedNetwork, alg.actor.mlp)
    before = [copy.deepcopy(head.state_dict()) for head in router.heads]
    states = [{p: copy.deepcopy(alg.optimizer.state[p]) for p in head.parameters()} for head in router.heads]
    assert all(any(state[p]['exp_avg'].abs().sum() > 0 for p in state) for state in states)
    collect(alg, observations((1,) * 8))
    alg.update()
    for task in (0, 2, 3):
        assert state_equal(before[task], router.heads[task].state_dict())
        for parameter in router.heads[task].parameters():
            assert parameter.grad is None
            assert all(torch.equal(value, alg.optimizer.state[parameter][key])
                       for key, value in states[task][parameter].items())
    assert not state_equal(before[1], router.heads[1].state_dict())


def test_full_resume_restores_replay_progress_and_optimizer():
    alg = algorithm(dagger_replay_iterations=2)
    for _ in range(2):
        collect(alg, observations())
        alg.update()
    saved = copy.deepcopy(alg.save())
    resumed = algorithm(dagger_replay_iterations=2)
    assert resumed.load(saved, None, True)
    assert resumed.completed_updates == 2
    assert len(resumed.replay) == 2
    assert state_equal(resumed.actor.state_dict(), alg.actor.state_dict())
    for left, right in zip(resumed.replay, alg.replay, strict=True):
        assert state_equal(left, right)
    torch.manual_seed(51)
    collect(alg, observations())
    metrics = alg.update()
    torch.manual_seed(51)
    collect(resumed, observations())
    assert resumed.update() == pytest.approx(metrics)
    assert state_equal(resumed.actor.state_dict(), alg.actor.state_dict())


def test_wrong_teacher_hash_rejected_before_actor_mutation():
    alg = algorithm()
    saved = copy.deepcopy(alg.save())
    saved['student_dagger']['contract']['teacher_sha256'] = 'wrong'
    for tensor in saved['actor_state_dict'].values():
        if tensor.is_floating_point():
            tensor.add_(3)
    before = copy.deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError, match='contract'):
        alg.load(saved, None, True)
    assert state_equal(before, alg.actor.state_dict())


def test_actor_only_load_requires_no_teacher_or_replay_metadata(monkeypatch):
    source = algorithm()
    saved = source.save()
    saved.pop('student_dagger')
    restored = algorithm()
    restored.teacher = None
    monkeypatch.setattr('zbot_rl_mjlab.student.load_universal_teacher',
                        lambda *args: pytest.fail('playback must not read teacher checkpoint'))
    assert restored.load(saved, {'actor': True}, True) is False
    obs = observations()
    assert torch.equal(restored.actor(obs), source.actor(obs))


def test_unequal_task_cohorts_have_equal_weight_in_actual_gradient(monkeypatch):
    obs = observations((0,) * 20 + (1,) * 2 + (2,) * 3 + (3,))
    alg = algorithm(obs)
    alg.actor.update_normalization(obs)
    prediction = alg.actor(obs)
    with torch.no_grad():
        labels = effective_labels(alg, obs)
    terms = []
    for task in range(4):
        mask = obs['teacher'][:, 39].round() == task
        width = 3 if task in (1, 3) else 6
        terms.append((prediction[mask, :width] - labels[mask, :width]).square().mean())
    parameters = [parameter for parameter in alg.actor.parameters() if parameter.requires_grad]
    expected = torch.autograd.grad(torch.stack(terms).mean(), parameters)
    observed = []

    def capture_gradients(parameters, max_norm):
        observed.extend(parameter.grad.detach().clone() for parameter in parameters if parameter.requires_grad)
        return torch.tensor(0.)

    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture_gradients)
    collect(alg, obs)
    alg.update()
    assert len(observed) == len(expected)
    assert all(torch.allclose(left, right, atol=2e-6, rtol=2e-5)
               for left, right in zip(observed, expected, strict=True))
