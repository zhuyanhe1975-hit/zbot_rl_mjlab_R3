"""Per-task diagnostics retain every episode and pre-reset task identity."""

import pytest
import torch

from zbot_rl_mjlab.student_metrics import StudentTaskMetrics


def observation(tasks):
    state = torch.zeros(len(tasks), 56)
    state[:, 39] = torch.as_tensor(tasks)
    return state


def step(metrics, tasks, rewards, dones, **reasons):
    metrics.observe(observation(tasks), torch.tensor(rewards), torch.tensor(dones),
                    {name: torch.tensor(mask) for name, mask in reasons.items()})


def test_rewards_are_rates_per_current_task_and_episode_uses_initial_task():
    metrics = StudentTaskMetrics(1, 'cpu', .5)
    step(metrics, [0], [2.], [False])
    step(metrics, [2], [4.], [True], unsafe=[True])
    # The next episode starts as a different task after reset.
    step(metrics, [3.25], [3.], [True], skill_complete=[True], time_out=[True])
    result = metrics.flush()
    assert result['task_biped_reward_rate'] == 4.
    assert result['task_snake_reward_rate'] == 8.
    assert result['task_biped_completed_episodes'] == 1
    assert result['task_biped_mean_episode_return'] == 6.
    assert result['task_biped_mean_episode_s'] == 1.
    assert result['task_biped_unsafe_count'] == 1
    assert result['task_biped_early_failure_count'] == 1
    assert result['task_snake_completed_episodes'] == 0
    assert 'task_snake_mean_episode_s' not in result
    assert result['task_ascending_success_count'] == 1
    assert result['task_ascending_time_out_count'] == 1
    assert result['task_ascending_early_failure_count'] == 0


def test_ongoing_episode_keeps_return_age_and_initial_task_across_flushes():
    metrics = StudentTaskMetrics(1, 'cpu', 1.)
    step(metrics, [1], [2.], [False])
    first = metrics.flush()
    assert first['task_descending_ongoing_mean_age_s'] == 1.
    assert first['task_descending_completed_episodes'] == 0
    assert 'task_descending_mean_episode_return' not in first
    step(metrics, [2], [3.], [False])
    second = metrics.flush()
    assert second['task_descending_ongoing_mean_age_s'] == 2.
    assert second['task_descending_frames'] == 0
    assert 'task_descending_reward_rate' not in second
    step(metrics, [3], [4.], [True], transition_failure=[True])
    final = metrics.flush()
    assert final['task_descending_mean_episode_s'] == 3.
    assert final['task_descending_mean_episode_return'] == 9.
    assert final['task_descending_early_failure_count'] == 0
    assert final['task_descending_ongoing_episodes'] == 0
    empty = metrics.flush()
    assert empty['task_descending_completed_episodes'] == 0
    assert 'task_descending_mean_episode_s' not in empty


def test_overlapping_reasons_count_individually_and_early_failure_once():
    metrics = StudentTaskMetrics(2, 'cpu', .02)
    step(metrics, [0, 2], [1., 2.], [True, False],
         biped_fallen=[True, True], unsafe=[True, True], skill_complete=[False, True])
    result = metrics.flush()
    assert result['task_biped_early_failure_count'] == 1
    assert result['task_biped_biped_fallen_count'] == 1
    assert result['task_biped_unsafe_count'] == 1
    assert result['task_snake_unsafe_count'] == 0
    assert result['task_snake_success_count'] == 0


def test_all_environments_count_and_results_are_environment_order_invariant():
    n = 256
    tasks = torch.arange(n) % 4
    rewards = torch.arange(n).float()
    done = torch.ones(n, dtype=torch.bool)
    fallen = torch.arange(n) < 156  # Deliberately outside the last100 environment IDs.
    first, second = StudentTaskMetrics(n, 'cpu', .02), StudentTaskMetrics(n, 'cpu', .02)
    first.observe(observation(tasks), rewards, done, {'biped_fallen': fallen})
    order = torch.randperm(n)
    second.observe(observation(tasks[order]), rewards[order], done[order], {'biped_fallen': fallen[order]})
    actual, shuffled = first.flush(), second.flush()
    assert actual == shuffled
    assert sum(actual[f'task_{name}_completed_episodes'] for name in ('biped', 'descending', 'snake', 'ascending')) == n
    assert sum(actual[f'task_{name}_biped_fallen_count'] for name in ('biped', 'descending', 'snake', 'ascending')) == 156


def test_no_frames_or_episodes_do_not_emit_false_zero_averages():
    result = StudentTaskMetrics(2, 'cpu', .02).flush()
    assert result['task_biped_completed_episodes'] == 0
    assert result['task_biped_frames'] == 0
    assert not any('mean_' in key or 'reward_rate' in key for key in result)


def test_dynamic_reason_allocated_during_inference_can_flush_outside_it():
    metrics = StudentTaskMetrics(2, 'cpu', .02)
    with torch.inference_mode():
        step(metrics, [0, 1], [1., 2.], [True, False],
             demonstration_time_out=[True, False])
    result = metrics.flush()
    assert result['task_biped_demonstration_time_out_count'] == 1
    assert metrics.flush()['task_biped_demonstration_time_out_count'] == 0


@pytest.mark.parametrize('count,dt', [(0, .02), (2, 0.), (2, float('nan'))])
def test_invalid_dimensions_or_time_step_rejected(count, dt):
    with pytest.raises(ValueError):
        StudentTaskMetrics(count, 'cpu', dt)
