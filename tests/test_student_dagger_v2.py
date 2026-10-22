"""DAgger v2 collection support, action targets and replay coverage regressions."""

import copy

import pytest
import torch
from test_student_dagger import algorithm, collect, observations, state_equal


def test_epoch_mode_visits_entire_replay_and_excludes_recovery_from_budget():
    obs = observations((0,) * 13 + (1,) * 5 + (2,) * 7 + (3,) * 3 + (3.25,) * 8)
    alg = algorithm(obs, dagger_updates=0, dagger_epochs=1, dagger_batch_size=8,
                    dagger_max_samples_per_task=13)
    visited = []

    def record(module, args):
        batch = args[0]
        visited.extend(batch['actor'][:, -56].detach().tolist())

    hook = alg.actor.register_forward_pre_hook(record)
    collect(alg, obs)
    alg.update()
    hook.remove()
    ordinary = set(obs['teacher'][:28, 0].tolist())
    recovery = set(obs['teacher'][28:, 0].tolist())
    assert set(visited) == ordinary
    assert set(visited).isdisjoint(recovery)
    replay = alg.replay[0]
    assert len(replay) == 28
    assert torch.equal(torch.bincount(replay['tasks'], minlength=4), torch.tensor([13, 5, 7, 3]))
    assert all(raw[:, 39].ne(3.25).all() for raw in alg.teacher.queries)


def test_recovery_only_rollout_cannot_update_ascent_head():
    alg = algorithm()
    collect(alg, observations())
    alg.update()
    before = copy.deepcopy(alg.actor.state_dict())
    collect(alg, observations((3.25,) * 8))
    alg.update()
    # No optimizer gradients or Adam momentum may turn scripted recovery into
    # ascent-policy supervision. Normalization buffers may still be updated.
    parameters = dict(alg.actor.named_parameters())
    assert all(torch.equal(value, before[name]) for name, value in parameters.items())
    assert all(parameter.grad is None for parameter in parameters.values())
    assert sum(map(len, alg.replay)) == 0


def test_rate_labels_match_executed_clamp_but_biped_keeps_unbounded_native_action():
    obs = observations((0, 1, 2, 3))
    obs['teacher'][:, :3] = torch.tensor([40., -40., 30.])
    obs['actor'] = obs['teacher'][:, None].expand(-1, 5, -1).clone().flatten(1)
    alg = algorithm(obs)
    collect(alg, obs)
    alg.update()
    replay = alg.replay[0]
    biped = replay['labels'][replay['tasks'] == 0]
    rates = replay['labels'][replay['tasks'] != 0]
    assert biped.abs().max() > 1
    assert rates.abs().max() == 1
    assert torch.equal(biped[0, :3], torch.tensor([4.7, -3.3, 3.7]))
    for task in (1, 3):
        value = replay['labels'][replay['tasks'] == task]
        assert torch.equal(value[:, 3:], -value[:, :3].flip(-1))


def test_version_one_full_resume_rejected_before_mutation_but_actor_play_allowed():
    alg = algorithm()
    saved = copy.deepcopy(alg.save())
    saved['student_dagger']['contract']['algorithm'] = 'universal_teacher_dagger_v1'
    for tensor in saved['actor_state_dict'].values():
        if tensor.is_floating_point():
            tensor.add_(2)
    before = copy.deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError, match='contract'):
        alg.load(saved, None, True)
    assert state_equal(before, alg.actor.state_dict())
    assert alg.load(saved, {'actor': True}, True) is False
    assert state_equal(saved['actor_state_dict'], alg.actor.state_dict())


def test_teacher_warmup_and_per_rollout_mixture_use_exact_prestep_labels():
    obs = observations((0, 1, 2, 3) * 2)
    alg = algorithm(obs, teacher_warmup_iterations=2, teacher_mix_iterations=4)
    assert alg.teacher_beta() == 1
    before = copy.deepcopy(alg.teacher.state_dict())
    with torch.no_grad():
        expected = alg.teacher(obs)
    alg.teacher.queries.clear()
    original = obs.clone()
    first = collect(alg, obs)
    assert torch.equal(first, expected)
    assert alg._teacher_mask.all()
    obs['teacher'][:, 0] += 1
    obs['actor'][:, -56] += 1
    second = collect(alg, obs)
    assert torch.equal(second, alg.teacher(obs))
    assert torch.equal(alg.storage.observations[0]['teacher'], original['teacher'])
    # Exactly two collection queries; aggregate reuses those labels.
    alg.teacher.queries.pop()
    alg.update()
    assert len(alg.teacher.queries) == 2
    replay = alg.replay[0]
    biped = replay['tasks'] == 0
    markers = replay['actor'][:, -56]
    assert torch.allclose(replay['labels'][biped, 0], markers[biped] * .1 + .7)
    assert state_equal(before, alg.teacher.state_dict())
    assert alg.teacher.scale.grad is None

    alg.completed_updates = 2
    assert alg.teacher_beta() == 1
    alg.completed_updates = 4
    assert alg.teacher_beta() == .5
    obs = observations()
    with torch.no_grad():
        student = alg.actor(obs)
        teacher = alg.teacher(obs)
    torch.manual_seed(8)
    mixed = collect(alg, obs)
    mask = alg._teacher_mask.clone()
    assert mask.any() and (~mask).any()
    assert torch.equal(mixed, torch.where(mask[:, None], teacher, student))
    torch.manual_seed(99)
    next_action = collect(alg, obs)
    assert torch.equal(alg._teacher_mask, mask)
    assert torch.equal(next_action, mixed)
    alg.completed_updates = 6
    assert alg.teacher_beta() == 0


def test_teacher_anchor_survives_replay_eviction_and_full_resume():
    opts = {'teacher_warmup_iterations': 1, 'teacher_mix_iterations': 0,
            'dagger_anchor_fraction': .25, 'dagger_anchor_max_samples_per_task': 2}
    alg = algorithm(**opts)
    warmup = observations()
    collect(alg, warmup)
    alg.update()
    assert alg.anchor is not None
    anchor = alg.anchor.clone()
    assert torch.equal(torch.bincount(anchor['tasks'], minlength=4), torch.tensor([2, 2, 2, 2]))
    later = observations()
    later['teacher'][:, 0] += 100
    later['actor'][:, -56] += 100
    collect(alg, later)
    metrics = alg.update()
    assert state_equal(alg.anchor, anchor)
    assert (alg.replay[0]['actor'][:, -56] >= 100).all()
    assert (alg.anchor['actor'][:, -56] < 100).all()
    assert metrics['dagger_anchor_samples'] == 8
    assert metrics['dagger_supervised_rows'] > len(later)
    saved = copy.deepcopy(alg.save())
    resumed = algorithm(**opts)
    resumed.load(saved, None, True)
    assert resumed.completed_updates == 2
    assert resumed.teacher_beta() == 0
    assert state_equal(resumed.anchor, anchor)
    assert state_equal(resumed.replay[0], alg.replay[0])


def test_recovery_frames_do_not_consume_ascent_sample_cap():
    obs = observations((3.25,) * 20 + (3,) * 4)
    alg = algorithm(obs, dagger_max_samples_per_task=2)
    collect(alg, obs)
    metrics = alg.update()
    replay = alg.replay[0]
    assert len(replay) == 2
    assert (replay['tasks'] == 3).all()
    assert (replay['actor'][:, -17] == 3).all()
    assert metrics['rollout_recovery_frames'] == 20
    assert all((query[:, 39] == 3).all() for query in alg.teacher.queries)
