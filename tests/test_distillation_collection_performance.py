"""Collection optimizations preserve rollout actions, RNG and scalar metrics."""

from copy import deepcopy

import pytest
import torch
from rsl_rl.algorithms import PPO
from test_universal_distillation import algorithm as algorithm  # noqa: PLC0414


@pytest.mark.parametrize("probe", [False, True])
def test_warmup_reuses_actor_forward_and_preserves_transition(algorithm, probe):
    alg, obs, _ = algorithm
    alg.distillation_iteration = alg.warmup_iterations if probe else 0
    reference = deepcopy(alg)
    torch.manual_seed(27)
    with torch.inference_mode():
        PPO.act(reference, obs)
        expected = reference.actor(obs, stochastic_output=False).detach()
        if probe:
            expected[2:] = 0.7
        else:
            expected[:] = 0.7
    expected_rng = torch.get_rng_state()
    calls = []
    hook = alg.actor.register_forward_hook(lambda *args: calls.append(True))
    # Sampling the replay cache consumes RNG before PPO.act in both versions.
    # Disable cache capacity here to isolate the unchanged actor RNG stream.
    alg._sample_count = alg.distillation_max_samples
    torch.manual_seed(27)
    with torch.inference_mode():
        actual = alg.act(obs)
    hook.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(alg.transition.actions, expected, rtol=0, atol=0)
    torch.testing.assert_close(alg.transition.values, reference.transition.values)
    torch.testing.assert_close(alg.transition.actions_log_prob, reference.transition.actions_log_prob)
    for actual_param, expected_param in zip(alg.transition.distribution_params, reference.transition.distribution_params, strict=True):
        torch.testing.assert_close(actual_param, expected_param)
    torch.testing.assert_close(alg.actor.output_std, reference.actor.output_std)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert len(calls) == 1


def test_collection_defers_scalar_counter_reads_until_update(algorithm, monkeypatch):
    alg, obs, _ = algorithm

    def scalar_read(*args, **kwargs):
        raise AssertionError("collection must not read device counter scalars")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__int__", scalar_read)
        patch.setattr(torch.Tensor, "item", scalar_read)
        with torch.inference_mode():
            alg.act(obs)
    assert alg._sample_count == 6


def test_collection_metrics_remain_python_scalars_and_reset(algorithm):
    alg, obs, _ = algorithm
    from test_universal_distillation import rollout

    for _ in range(2):
        rollout(alg, obs)
        metrics = alg.update()
        assert metrics["distillation_coverage"] == 1.0
        assert metrics["warmup_student_fraction"] == 0.0
        assert metrics["teacher_walking_candidates"] == 4.0
        assert metrics["teacher_omni_candidates"] == 4.0
        assert metrics["teacher_transform_candidates"] == 8.0
        assert all(isinstance(value, float) for value in metrics.values())


def test_deferred_counters_keep_finite_filtering_and_teacher_ids(algorithm):
    alg, obs, _ = algorithm
    from test_universal_distillation import rollout

    targets = alg.teachers.query()
    targets.actions = targets.actions.detach().clone()
    targets.weights[0] = 0
    targets.weights[1] = float("nan")
    targets.actions[2, 0] = float("nan")
    targets.teacher_ids[3] = -1
    targets.weights[4] = float("inf")
    targets.teacher_ids[6] = 99
    alg.teachers.query = lambda: targets
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["distillation_coverage"] == 0.25
    assert metrics["warmup_student_fraction"] == 0.75
    assert metrics["teacher_walking_candidates"] == 0.0
    assert metrics["teacher_omni_candidates"] == 0.0
    assert metrics["teacher_transform_candidates"] == 4.0
    assert alg._valid_count == alg._warmup_student_count == 0
    assert not alg._teacher_counts
