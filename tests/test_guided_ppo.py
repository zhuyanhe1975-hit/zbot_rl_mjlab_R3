"""Real tiny-model PPO and task-balanced privileged teacher guidance contracts."""

import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from zbot_rl_mjlab.guided_ppo import GuidedTaskPPO, GuidedTaskPpoCfg
from zbot_rl_mjlab.task_models import RoutedTaskNormalizedMLP
from zbot_rl_mjlab.task_ppo import IndependentTaskCritics, TaskPPO


def observations(tasks=(0, 1, 2, 3)):
    raw = torch.randn(len(tasks), 56) * .1
    raw[:, 39] = torch.tensor(tasks)
    return TensorDict({'actor': raw.clone(), 'critic': raw.clone()}, batch_size=[len(tasks)])


def algorithm(obs=None, *, steps=2, cls: type[TaskPPO] = GuidedTaskPPO, actor_cls=MLPModel, **kwargs) -> Any:
    obs = observations() if obs is None else obs
    sets = {'actor': ['actor'], 'critic': ['critic']}
    actor = actor_cls(obs, sets, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
                     distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)})
    critic = IndependentTaskCritics(obs, sets, 'critic', 1, hidden_dims=(16,), obs_normalization=True)
    storage = RolloutStorage('rl', obs.batch_size[0], steps, obs, [6])
    if issubclass(cls, GuidedTaskPPO):
        kwargs.setdefault('guidance_tasks', (1, 2, 3))
    return cls(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
               schedule='fixed', gamma=.9, lam=.8, **kwargs)


def teacher(obs, action=.6, weights=None):
    weights = torch.ones(len(obs)) if weights is None else weights
    targets = SimpleNamespace(actions=torch.full((len(obs), 6), action), weights=weights, counts={})
    return SimpleNamespace(query=lambda: targets)


def collect(alg, obs, steps=2):
    for step in range(steps):
        with torch.inference_mode():
            alg.act(obs)
            alg.process_env_step(obs, torch.arange(len(obs)).float() + step, torch.zeros(len(obs)), {})
    with torch.inference_mode():
        alg.compute_returns(obs)


def state_equal(left, right):
    return all(torch.equal(left[key], right[key]) for key in left)


@pytest.mark.parametrize('guided_tasks', [(1, 2, 3), (0, 1, 2, 3)])
def test_labels_never_replace_on_policy_actions_or_log_probabilities(guided_tasks):
    obs = observations()
    guided = algorithm(obs, guidance_tasks=guided_tasks)
    plain = algorithm(obs, cls=TaskPPO)
    plain.actor.load_state_dict(guided.actor.state_dict())
    plain.critic.load_state_dict(guided.critic.state_dict())
    guided.teachers = teacher(obs)
    torch.manual_seed(81)
    expected = plain.act(obs)
    expected_log = plain.transition.actions_log_prob.clone()
    torch.manual_seed(81)
    actual = guided.act(obs)
    assert torch.equal(actual, expected)
    assert torch.equal(guided.transition.actions_log_prob, expected_log)
    assert not torch.equal(actual, guided.teachers.query().actions)
    assert guided._guidance_counts[0] == int(0 in guided_tasks)
    assert all(guided._guidance_counts[task] > 0 for task in (1, 2, 3))


def test_real_update_runs_ppo_and_bc_from_first_iteration_and_normalizes_once():
    obs = observations(tuple([0, 1, 2, 3] * 8))
    guided = algorithm(obs, guidance_updates=2)
    plain = algorithm(obs, cls=TaskPPO)
    plain.actor.load_state_dict(guided.actor.state_dict())
    plain.critic.load_state_dict(guided.critic.state_dict())
    guided.teachers = teacher(obs)
    torch.manual_seed(40)
    collect(guided, obs)
    torch.manual_seed(40)
    collect(plain, obs)
    before = copy.deepcopy(guided.critic.state_dict())
    torch.manual_seed(41)
    metrics = guided.update()
    torch.manual_seed(41)
    plain.update()
    assert {'value', 'surrogate', 'guidance'} <= metrics.keys()
    assert metrics['guidance'] > 0
    assert not state_equal(before, guided.critic.state_dict())
    # Guidance uses the shared optimizer but gives every critic parameter grad=None.
    assert state_equal(plain.critic.state_dict(), guided.critic.state_dict())
    assert all(parameter.grad is None for parameter in guided.critic.parameters())
    assert not state_equal(plain.actor.state_dict(), guided.actor.state_dict())
    assert torch.equal(plain.actor.obs_normalizer.count, guided.actor.obs_normalizer.count)
    assert guided.guidance_iteration == 1
    assert guided._guidance_counts == [0] * 4


def test_disabled_guidance_is_exact_task_ppo_update():
    obs = observations()
    guided = algorithm(obs, guidance_enabled=False)
    plain = algorithm(obs, cls=TaskPPO)
    plain.actor.load_state_dict(guided.actor.state_dict())
    plain.critic.load_state_dict(guided.critic.state_dict())
    guided.teachers = SimpleNamespace(query=lambda: pytest.fail('disabled teacher queried'))
    torch.manual_seed(19)
    collect(guided, obs)
    torch.manual_seed(19)
    collect(plain, obs)
    torch.manual_seed(22)
    left = guided.update()
    torch.manual_seed(22)
    right = plain.update()
    assert left == right
    assert state_equal(guided.actor.state_dict(), plain.actor.state_dict())
    assert state_equal(guided.critic.state_dict(), plain.critic.state_dict())


def test_cache_is_bounded_inference_safe_and_spans_rollout():
    obs = observations(tuple([1, 2, 3] * 20))
    alg = algorithm(obs, steps=4, guidance_max_samples_per_task=8)
    alg.teachers = teacher(obs)
    for step in range(4):
        obs['actor'][:, 0] = step
        with torch.inference_mode():
            alg.act(obs)
    assert alg._guidance_counts == [0, 8, 8, 8]
    for chunks in alg._guidance_samples.values():
        assert len(chunks) == 4
        assert [chunk[0]['actor'][0, 0].item() for chunk in chunks] == list(range(4))
        assert not chunks[0][0]['actor'].is_inference()
        assert not chunks[0][1].is_inference()
        assert not chunks[0][1].requires_grad
    metrics = alg._guidance_update()
    assert metrics['guidance'] > 0
    assert all(parameter.grad is None for parameter in alg.critic.parameters())


def test_invalid_labels_and_biped_are_excluded_before_task_balancing():
    obs = observations((0, 1, 1, 2, 2, 3, 3, 3))
    alg = algorithm(obs, steps=1)
    targets = torch.zeros(8, 6)
    targets[2, 0] = float('nan')
    targets[4, 0] = 1.1
    weights = torch.tensor([1., 1., 1., 0., 1., -1., float('inf'), 1.])
    alg._cache_guidance(obs, targets, weights)
    assert alg._guidance_counts == [0, 1, 0, 1]
    cast(torch.Tensor, obs['actor']).fill_(float('nan'))
    assert torch.isfinite(alg._guidance_samples[1][0][0]['actor']).all()


def test_mean_per_task_objective_keeps_minority_guidance_equal():
    # Compare the actual pre-update objective to equal per-task means; raw
    # environment-proportional weighting would strongly favor the snake cohort.
    torch.manual_seed(20)
    obs = observations(tuple([1] * 10 + [2] * 100 + [3] * 10))
    alg = algorithm(obs, steps=1, guidance_updates=1, guidance_batch_size=300)
    alg._cache_guidance(obs, torch.full((120, 6), .5), torch.ones(120))
    with torch.no_grad():
        expected = []
        for task in (1, 2, 3):
            cached = alg._guidance_samples[task][0][0]
            predicted = alg.actor(cached, stochastic_output=False)
            errors = (predicted - .5).square().mean(-1)
            expected.append(errors[torch.arange(len(errors)) % 5 != 0].mean())
    metrics = alg._guidance_update()
    assert metrics['guidance'] == pytest.approx(float(torch.stack(expected).mean()))
    assert metrics['guidance_descent_optimized'] == 8
    assert metrics['guidance_snake_motion_optimized'] == 80
    assert metrics['guidance_ascent_optimized'] == 8
    # Every nonempty task contributes one weighted mean to the objective; small
    # ascent/descent cohorts cannot disappear in an environment-proportional draw.
    assert metrics['guidance'] > 0


def test_held_out_agreement_anneals_only_covered_tasks_and_resumes():
    obs = observations(tuple([1] * 200 + [2] * 10))
    alg = algorithm(obs, steps=1, guidance_updates=1, guidance_agreement_passes=2,
                    guidance_decay=.8, guidance_agreement_mse=1.)
    for _ in range(2):
        with torch.no_grad():
            labels = alg.actor(obs, stochastic_output=False)
        alg._cache_guidance(obs, labels, torch.ones(len(obs)))
        alg._guidance_update()
        alg._clear_guidance()
    assert alg.guidance_coefficients == [0., .8, 1., 1.]
    assert alg.guidance_passes == [0, 2, 0, 0]
    saved = copy.deepcopy(alg.save())
    resumed = algorithm(obs, steps=1, guidance_updates=1, guidance_agreement_passes=2,
                        guidance_decay=.8, guidance_agreement_mse=1.)
    assert resumed.load(saved, None, True)
    assert resumed.guidance_coefficients == alg.guidance_coefficients
    assert resumed.guidance_agreement_ema == alg.guidance_agreement_ema
    assert resumed.guidance_passes == alg.guidance_passes


@pytest.mark.parametrize('field,value', [('version', 3), ('settings', {}), ('provenance', {'changed': 1}),
                                         ('coefficients', [0, float('nan'), 1, 1]), ('agreement_passes', [0, -1, 0, 0])])
def test_full_resume_contract_rejected_before_upstream_mutation(field, value):
    alg = algorithm()
    saved = copy.deepcopy(alg.save())
    saved['guided_task_ppo'][field] = value
    for tensor in saved['actor_state_dict'].values():
        if tensor.is_floating_point():
            tensor.add_(3)
    before = copy.deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError, match='guidance'):
        alg.load(saved, None, True)
    assert state_equal(before, alg.actor.state_dict())


def test_actor_only_load_needs_no_teachers_or_guidance_metadata():
    alg = algorithm()
    saved = alg.save()
    saved.pop('guided_task_ppo')
    alg.teacher_provenance = {'unavailable': 'no teacher files needed for playback'}
    assert alg.load(saved, {'actor': True}, True) is False


@pytest.mark.parametrize('kwargs', [{'guidance_updates': 0}, {'guidance_batch_size': 2},
                                     {'guidance_max_samples_per_task': -1}, {'guidance_decay': 1.1},
                                     {'guidance_ema_beta': 1}, {'guidance_start_coef': .01},
                                     {'guidance_agreement_mse': float('nan')}, {'guidance_enabled': 'yes'},
                                     {'guidance_start_coef': 0, 'guidance_floor_coef': 0},
                                     {'teacher_checkpoints': {'walking': 'model.pt'}},
                                     {'teacher_checkpoints': {'snake': ''}}])
def test_invalid_guidance_configuration(kwargs):
    with pytest.raises(ValueError):
        algorithm(**kwargs)


@pytest.mark.parametrize('actions,weights', [(torch.zeros(4, 3), torch.ones(4)),
                                             (torch.zeros(4, 6), torch.ones(4, 1))])
def test_target_shape_validation(actions, weights):
    alg = algorithm()
    with pytest.raises(ValueError, match='targets'):
        alg._cache_guidance(observations(), actions, weights)


@pytest.mark.parametrize('model_key', ['actor_state_dict', 'critic_state_dict'])
def test_architecture_mismatch_is_preflighted_before_any_model_mutation(model_key):
    alg = algorithm()
    saved = copy.deepcopy(alg.save())
    for tensor in saved['actor_state_dict'].values():
        if tensor.is_floating_point():
            tensor.add_(3)
    first = next(iter(saved[model_key]))
    saved[model_key][first] = torch.zeros(12345)
    before = copy.deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError, match='architecture'):
        alg.load(saved, None, True)
    assert state_equal(before, alg.actor.state_dict())


def test_routed_bc_preserves_inactive_head_parameters_and_existing_adam_momentum():
    torch.manual_seed(105)
    obs = observations(tuple([0, 1, 2, 3] * 20))
    alg = algorithm(obs, actor_cls=RoutedTaskNormalizedMLP, guidance_updates=4)
    # A real PPO update gives every branch nonzero Adam momentum. Guidance must
    # preserve inactive branches even when the optimizer already has their state.
    collect(alg, obs)
    TaskPPO.update(alg)
    heads = alg._raw_actor.mlp.heads
    assert all(any(alg.optimizer.state[p]['exp_avg'].abs().sum() > 0 for p in head.parameters())
               for head in heads)
    parameters = [copy.deepcopy(head.state_dict()) for head in heads]
    states = [{p: copy.deepcopy(alg.optimizer.state[p]) for p in head.parameters()} for head in heads]
    subset = observations(tuple([1] * 40))
    alg._cache_guidance(subset, torch.full((40, 6), .6), torch.ones(40))
    alg._guidance_update()
    for task in (0, 2, 3):
        assert state_equal(parameters[task], heads[task].state_dict())
        for parameter in heads[task].parameters():
            assert parameter.grad is None
            for key, value in states[task][parameter].items():
                actual = alg.optimizer.state[parameter][key]
                assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
    assert not state_equal(parameters[1], heads[1].state_dict())
    for parameter in heads[1].parameters():
        assert alg.optimizer.state[parameter]['step'] == states[1][parameter]['step'] + 4


def test_four_task_guidance_default_optimizes_biped_and_balances_minority_tasks():
    assert GuidedTaskPpoCfg().guidance_tasks == (0, 1, 3)
    obs = observations(tuple([0] * 100 + [1] * 10 + [2] * 10 + [3] * 10))
    alg = algorithm(obs, steps=1, actor_cls=RoutedTaskNormalizedMLP,
                    guidance_tasks=(0, 1, 2, 3), guidance_updates=1, guidance_batch_size=400,
                    teacher_checkpoints={'omni': 'omni.pt', 'transform': 'transform.pt', 'snake': 'snake.pt'})
    alg._cache_guidance(obs, torch.full((130, 6), .6), torch.ones(130))
    before = copy.deepcopy(alg._raw_actor.mlp.heads[0].state_dict())
    assert alg._guidance_counts == [100, 10, 10, 10]
    with torch.no_grad():
        expected = []
        for task in range(4):
            cached = alg._guidance_samples[task][0][0]
            predicted = alg.actor(cached, stochastic_output=False)
            errors = (predicted - .6).square().mean(-1)
            expected.append(errors[torch.arange(len(errors)) % 5 != 0].mean())
    metrics = alg._guidance_update()
    assert metrics['guidance'] == pytest.approx(float(torch.stack(expected).mean()))
    assert metrics['guidance_biped_motion_optimized'] == 80
    assert metrics['guidance_descent_optimized'] == 8
    assert metrics['guidance_snake_motion_optimized'] == 8
    assert metrics['guidance_ascent_optimized'] == 8
    assert not state_equal(before, alg._raw_actor.mlp.heads[0].state_dict())


def test_four_task_guidance_resumes_biped_schedule_and_requires_version_two():
    alg = algorithm(guidance_tasks=(0, 1, 2, 3))
    alg.guidance_coefficients[0] = .4
    alg.guidance_agreement_ema[0] = .01
    alg.guidance_passes[0] = 7
    alg.guidance_iteration = 23
    saved = copy.deepcopy(alg.save())
    contract = saved['guided_task_ppo']
    assert contract['version'] == 2
    assert contract['guided_tasks'] == [0, 1, 2, 3]
    assert contract['settings']['guidance_tasks'] == (0, 1, 2, 3)
    resumed = algorithm(guidance_tasks=(0, 1, 2, 3))
    resumed.load(saved, None, True)
    assert resumed.guidance_coefficients == alg.guidance_coefficients
    assert resumed.guidance_agreement_ema == alg.guidance_agreement_ema
    assert resumed.guidance_passes == alg.guidance_passes
    assert resumed.guidance_iteration == 23
    saved['guided_task_ppo']['version'] = 1
    with pytest.raises(ValueError, match='contract'):
        resumed.load(saved, None, True)


@pytest.mark.parametrize('tasks', [(), (0, 0), (4,), (-1,), (False,), (1.,), [0, 1]])
def test_guidance_task_configuration_rejects_invalid_ids(tasks):
    with pytest.raises(ValueError, match='guidance_tasks'):
        algorithm(guidance_tasks=tasks)


def test_configured_task_subset_preserves_excluded_coefficient_contract():
    alg = algorithm(guidance_tasks=(0, 2), guidance_batch_size=2)
    assert alg.guidance_coefficients == [1., 0., 1., 0.]
    saved = alg.save()
    saved['guided_task_ppo']['coefficients'][1] = .5
    with pytest.raises(ValueError, match='coefficients'):
        alg.load(saved, None, True)


def test_default_guidance_excludes_unqualified_snake_teacher():
    defaults = GuidedTaskPpoCfg()
    obs = observations(tuple([0, 1, 2, 3] * 10))
    alg = algorithm(obs, steps=1, guidance_tasks=defaults.guidance_tasks,
                    actor_cls=RoutedTaskNormalizedMLP, guidance_updates=1)
    assert alg.guidance_coefficients == [1., 1., 0., 1.]
    alg._cache_guidance(obs, torch.full((40, 6), .6), torch.ones(40))
    assert alg._guidance_counts == [10, 10, 0, 10]
    assert 2 not in alg._guidance_samples
    snake_before = copy.deepcopy(alg._raw_actor.mlp.heads[2].state_dict())
    metrics = alg._guidance_update()
    assert 'guidance_snake_motion_optimized' not in metrics
    assert state_equal(snake_before, alg._raw_actor.mlp.heads[2].state_dict())
    assert all(parameter.grad is None for parameter in alg._raw_actor.mlp.heads[2].parameters())
    assert alg.save()['guided_task_ppo']['guided_tasks'] == [0, 1, 3]
