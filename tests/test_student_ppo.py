"""Student PPO transfer, warmup, frozen retention and resume regressions."""

import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from zbot_rl_mjlab.grouped_ppo import GroupedRolloutStorage
from zbot_rl_mjlab.student_models import HistoryGaussianStudentMLP
from zbot_rl_mjlab.student_ppo import (
    MjlabOnPolicyRunner,
    PPOHistoryGaussianStudentMLP,
    StudentPPO,
    UniversalStudentPPORunner,
)
from zbot_rl_mjlab.task_models import LocalFrameTaskNormalizedCritics


def observations():
    raw = torch.randn(16, 56) * .1
    raw[:, 39] = torch.arange(16) % 4
    return TensorDict({'actor': raw.repeat(1, 5), 'teacher': raw}, batch_size=[16])


def algorithm(warmup=1, **kwargs):
    obs = observations()
    sets = {'actor': ['actor'], 'critic': ['teacher']}
    actor = PPOHistoryGaussianStudentMLP(
        obs, sets, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
        distribution_cfg={'init_std': .1, 'rate_init_std': .1, 'learn_std': False},
    )
    critic = LocalFrameTaskNormalizedCritics(obs, sets, 'critic', 1, hidden_dims=(16,),
                                            obs_normalization=True)
    return StudentPPO(actor, critic, RolloutStorage('rl', 16, 4, obs, [6]),
                      critic_warmup_iterations=warmup, num_learning_epochs=2, num_mini_batches=1,
                      learning_rate=.001, schedule='fixed', entropy_coef=0., **kwargs)


def rollout(alg, reward_sign=1.):
    obs = observations()
    with torch.inference_mode():
        for step in range(4):
            actions = alg.act(obs)
            reward = -((actions[:, 0] - .5).square()) + step * .1
            alg.process_env_step(obs, reward * reward_sign, torch.zeros(16), {})
        alg.compute_returns(obs)


def equal_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def test_real_ppo_warmup_updates_only_critic_then_actor():
    torch.manual_seed(12)
    alg = algorithm()
    before = copy.deepcopy(alg.actor.state_dict())
    critic = copy.deepcopy(alg.critic.state_dict())
    flags = [p.requires_grad for p in alg.actor.parameters()]
    rollout(alg)
    metrics = alg.update()
    assert metrics['student_ppo_critic_warmup'] == 1
    assert equal_state(before, alg.actor.state_dict())
    assert not equal_state(critic, alg.critic.state_dict())
    assert flags == [p.requires_grad for p in alg.actor.parameters()]
    rollout(alg)
    metrics = alg.update()
    assert metrics['student_ppo_critic_warmup'] == 0
    assert metrics['student_ppo_completed_updates'] == 2
    assert not equal_state(before, alg.actor.state_dict())
    for key, value in before.items():
        if key.startswith(('distribution.', 'obs_normalizer.')):
            assert torch.equal(value, alg.actor.state_dict()[key])


def test_reward_changes_actor_update():
    torch.manual_seed(2)
    first = algorithm(0)
    second = algorithm(0)
    second.actor.load_state_dict(first.actor.state_dict())
    second.critic.load_state_dict(first.critic.state_dict())
    torch.manual_seed(3)
    rollout(first)
    torch.manual_seed(3)
    rollout(second, reward_sign=-1.)
    torch.manual_seed(4)
    first.update()
    torch.manual_seed(4)
    second.update()
    assert not equal_state(first.actor.state_dict(), second.actor.state_dict())


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_warmup_rejects_invalid_iterations(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        algorithm(value)


def test_algorithm_resume_preserves_next_update_and_rejects_dagger():
    alg = algorithm(2)
    rollout(alg)
    alg.update()
    saved = alg.save()
    restored = algorithm(2)
    assert restored.load(saved, None, True)
    assert restored.completed_updates == 1
    rollout(restored)
    assert restored.update()['student_ppo_critic_warmup'] == 1
    rollout(restored)
    assert restored.update()['student_ppo_critic_warmup'] == 0
    state = copy.deepcopy(restored.actor.state_dict())
    with pytest.raises(ValueError, match='PPO phase'):
        restored.load(saved | {'student_ppo': None}, None, True)
    assert equal_state(state, restored.actor.state_dict())


class Environment:
    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(student_history_length=5)
        self.common_step_counter = 17
        self.reset_calls = 0
        self.start_step = 0

    def reset(self):
        self.reset_calls += 1
        self.start_step = self.common_step_counter


def runner() -> Any:
    result: Any = UniversalStudentPPORunner.__new__(UniversalStudentPPORunner)
    result.env = Environment()
    result.alg = algorithm()
    result.device = 'cpu'
    result.student_source = None
    result.current_learning_iteration = 0
    return result


def dagger_checkpoint(instance):
    actor = HistoryGaussianStudentMLP(
        observations(), {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,),
        obs_normalization=True, distribution_cfg={'init_std': .7, 'rate_init_std': .6},
    )
    actor.update_normalization(observations())
    return actor, {'actor_state_dict': actor.state_dict(), 'critic_state_dict': {'unusable': None},
                   'optimizer_state_dict': {'unusable': None}, 'student_dagger': {}, 'iter': 150,
                   'infos': {'universal_student': instance._schema(),
                             'env_state': {'common_step_counter': 999999}}}


def test_cold_transfer_preserves_mean_and_stats_but_not_std_critic_or_clock(tmp_path, monkeypatch):
    monkeypatch.setattr('zbot_rl_mjlab.student.load_universal_teacher',
                        lambda *args, **kwargs: pytest.fail('must never load a teacher'))
    instance = runner()
    actor, saved = dagger_checkpoint(instance)
    path = tmp_path / 'dagger.pt'
    torch.save(saved, path)
    critic = copy.deepcopy(instance.alg.critic.state_dict())
    instance.initialize_student(path)
    obs = observations()
    assert torch.equal(actor(obs), instance.alg.actor(obs))
    for key, value in actor.state_dict().items():
        if key.startswith('obs_normalizer.'):
            assert torch.equal(value, instance.alg.actor.state_dict()[key])
    assert equal_state(critic, instance.alg.critic.state_dict())
    assert not instance.alg.optimizer.state
    assert instance.env.common_step_counter == instance.env.start_step == 0
    assert instance.alg.completed_updates == instance.current_learning_iteration == 0
    assert instance.student_source['iteration'] == 150
    assert equal_state(instance.alg.actor.state_dict(), instance.alg.reference_actor.state_dict())
    instance.alg.actor(obs, stochastic_output=True)
    assert torch.allclose(instance.alg.actor.output_std, torch.full((16, 6), .1))
    instance.alg.actor.update_normalization(observations())
    assert torch.equal(actor(obs), instance.alg.actor(obs))


def test_resume_rejects_dagger_before_upstream_mutation(tmp_path, monkeypatch):
    instance = runner()
    _, saved = dagger_checkpoint(instance)
    path = tmp_path / 'dagger.pt'
    torch.save(saved, path)
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load',
                        lambda *args, **kwargs: pytest.fail('must reject before upstream load'))
    with pytest.raises(ValueError, match='full resume'):
        instance.load(path)
    instance.load(path, {'actor': True})
    assert instance.env.reset_calls == 0
    assert instance.env.common_step_counter == 17


@pytest.mark.parametrize('warmup', [1, 20])
def test_full_resume_rebases_clock_and_restores_provenance(tmp_path, monkeypatch, warmup):
    instance = runner()
    instance.alg.critic_warmup_iterations = warmup
    saved = instance.alg.save() | {'iter': 11, 'infos': {
        'universal_student': instance._schema(),
        'student_ppo': instance._phase() | {'source': {'iteration': 150}},
    }}
    path = tmp_path / 'ppo.pt'
    torch.save(saved, path)

    def upstream(self, path, cfg, strict, map_location):
        self.alg.completed_updates = 12
        self.current_learning_iteration = 11
        self.env.common_step_counter = 8000
        return saved['infos']

    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', upstream)
    instance.load(path)
    assert instance.current_learning_iteration == 12
    assert instance.env.start_step == 8000
    assert instance.student_source == {'iteration': 150}
    assert instance.env._student_pretraining is (12 < warmup)
    assert instance.env._student_rl_iteration == max(0, 12 - warmup)


def test_cold_transfer_checks_hash_and_exact_interface_before_import(tmp_path):
    instance = runner()
    _, saved = dagger_checkpoint(instance)
    path = tmp_path / 'dagger.pt'
    torch.save(saved, path)
    state = copy.deepcopy(instance.alg.actor.state_dict())
    with pytest.raises(ValueError, match='SHA256'):
        instance.initialize_student(path, 'wrong')
    saved['infos']['universal_student']['action_contract'] = 'beta'
    torch.save(saved, path)
    with pytest.raises(ValueError, match='matching universal'):
        instance.initialize_student(path)
    assert equal_state(state, instance.alg.actor.state_dict())


def test_fresh_training_requires_source_before_constructing_algorithm(monkeypatch):
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__',
                        lambda *args, **kwargs: pytest.fail('reject before building algorithm'))
    with pytest.raises(ValueError, match='initial_student_checkpoint'):
        UniversalStudentPPORunner(cast(Any, Environment()), {}, log_dir='logs')


def test_resume_and_playback_do_not_access_configured_source(monkeypatch):
    def upstream(self, env, cfg, log_dir, device):
        self.env = env
        self.alg = algorithm()
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', upstream)
    monkeypatch.setattr(UniversalStudentPPORunner, 'initialize_student',
                        lambda *args, **kwargs: pytest.fail('resume/playback must not read source'))
    # Empty termination mapping is sufficient for metrics wiring without stepping.
    env: Any = Environment()
    env.num_envs = 16
    env.step_dt = .02
    env.cfg.terminations = {}
    cfg = {'resume': True, 'initial_student_checkpoint': 'missing.pt', 'console_mode': 'full'}
    UniversalStudentPPORunner(env, cfg, log_dir='logs')
    UniversalStudentPPORunner(env, cfg | {'resume': False})


def test_training_publishes_schedule_before_collection_and_after_warmup(monkeypatch):
    def upstream(self, env, cfg, log_dir, device):
        self.env = env
        self.alg = algorithm(1)
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', upstream)
    env: Any = Environment()
    env.num_envs = 16
    env.step_dt = .02
    env.cfg.terminations = {}
    instance: Any = UniversalStudentPPORunner(env, {'resume': True, 'console_mode': 'full'}, log_dir='logs')
    assert env._student_pretraining is True
    assert env._student_rl_iteration == 0
    # Run actual PPO updates with metrics wrapper, without environment stepping.
    # Populate storage through the unwrapped parent step method; the synthetic
    # environment has no termination manager, which this test does not need.
    from zbot_rl_mjlab.task_ppo import TaskPPO
    instance.alg.process_env_step = lambda *args: TaskPPO.process_env_step(instance.alg, *args)
    rollout(instance.alg)
    instance.alg.update()
    assert env._student_pretraining is False
    assert env._student_rl_iteration == 0
    rollout(instance.alg)
    instance.alg.update()
    assert env._student_rl_iteration == 1


def test_playback_does_not_publish_or_replace_environment_schedule(monkeypatch):
    def upstream(self, env, cfg, log_dir, device):
        self.env = env
        self.alg = algorithm()
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', upstream)
    env: Any = Environment()
    env._student_pretraining = 'playback'
    env._student_rl_iteration = 456
    UniversalStudentPPORunner(env, {'console_mode': 'full'})
    assert env._student_pretraining == 'playback'
    assert env._student_rl_iteration == 456


def reference_groups(alg):
    obs = observations()
    return {task: obs[obs['teacher'][:, 39] == task] for task in (0, 2)}


def mean_distance(alg, groups):
    with torch.no_grad():
        return sum((alg.actor(obs) - alg.reference_actor(obs)).square().mean().item()
                   for obs in groups.values())


def test_reference_step_pulls_motion_means_toward_frozen_student():
    alg = algorithm(0)
    frozen = copy.deepcopy(alg.reference_actor.state_dict())
    with torch.no_grad():
        for task in (0, 2):
            for parameter in cast(Any, alg._raw_actor.mlp).heads[task].parameters():
                parameter.add_(.03)
    groups = reference_groups(alg)
    before = mean_distance(alg, groups)
    metrics = alg._reference_update(groups)
    assert mean_distance(alg, groups) < before
    assert metrics['student_ppo_reference_loss'] > 0
    assert metrics['student_ppo_reference_task0_kl'] > 0
    assert metrics['student_ppo_reference_task2_kl'] > 0
    assert equal_state(frozen, alg.reference_actor.state_dict())
    assert not any(parameter.requires_grad or parameter.grad is not None
                   for parameter in alg.reference_actor.parameters())
    assert not alg.reference_actor.training


def test_reference_step_cannot_move_transform_heads_or_critic_with_adam_momentum():
    alg = algorithm(0)
    # Seed Adam state for every trainable actor/critic parameter. Zero gradients
    # during retention would still move these using their accumulated momentum.
    for parameter in list(alg.actor.parameters()) + list(alg.critic.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    alg.optimizer.step()
    heads = {task: copy.deepcopy(cast(Any, alg._raw_actor.mlp).heads[task].state_dict()) for task in (1, 3)}
    critic = copy.deepcopy(alg.critic.state_dict())
    alg._reference_update(reference_groups(alg))
    for task in (1, 3):
        assert equal_state(heads[task], cast(Any, alg._raw_actor.mlp).heads[task].state_dict())
        assert all(parameter.grad is None for parameter in cast(Any, alg._raw_actor.mlp).heads[task].parameters())
    assert equal_state(critic, alg.critic.state_dict())
    assert all(parameter.grad is None for parameter in alg.critic.parameters())


def test_reference_warmup_is_immutable_and_decay_counts_only_policy_updates():
    alg = algorithm(2, reference_decay_iterations=2)
    frozen = copy.deepcopy(alg.reference_actor.state_dict())
    for _ in range(2):
        rollout(alg)
        metrics = alg.update()
        assert metrics['student_ppo_reference_updates'] == 0
        assert metrics['student_ppo_reference_coefficient'] == pytest.approx(.05)
        assert equal_state(frozen, alg.reference_actor.state_dict())
    for coefficient in (.05, .0275, .005, .005):
        rollout(alg)
        metrics = alg.update()
        assert metrics['student_ppo_reference_coefficient'] == pytest.approx(coefficient)
        assert equal_state(frozen, alg.reference_actor.state_dict())
    assert alg.reference_updates == 4


def test_v2_resume_restores_original_reference_independent_of_source():
    alg = algorithm(0)
    frozen = copy.deepcopy(alg.reference_actor.state_dict())
    rollout(alg)
    alg.update()
    saved = copy.deepcopy(alg.save())
    restored = algorithm(0)
    assert not equal_state(frozen, restored.reference_actor.state_dict())
    restored.load(saved)
    assert equal_state(frozen, restored.reference_actor.state_dict())
    assert restored.reference_updates == 1
    assert restored.reference_coefficient() == alg.reference_coefficient()
    before = copy.deepcopy(restored.actor.state_dict())
    saved['student_ppo']['contract']['version'] = 1
    with pytest.raises(ValueError, match='PPO phase'):
        restored.load(saved)
    assert equal_state(before, restored.actor.state_dict())


@pytest.mark.parametrize('version,phase', [(1, 'student_reward_only_ppo'), (2, 'student_ppo_reference')])
def test_actor_playback_accepts_old_and_new_ppo_without_reference_source(tmp_path, version, phase):
    instance = runner()
    saved = instance.alg.save() | {'infos': {'universal_student': instance._schema(),
                                           'student_ppo': {'version': version, 'phase': phase}}}
    saved.pop('student_ppo')
    path = tmp_path / 'playback.pt'
    torch.save(saved, path)
    instance.load(path, {'actor': True})
    assert instance.env.reset_calls == 0


def test_runner_rejects_v1_before_actor_mutation(tmp_path, monkeypatch):
    instance = runner()
    saved = instance.alg.save() | {'infos': {'universal_student': instance._schema(),
                                           'student_ppo': {'version': 1, 'phase': 'student_reward_only_ppo'}}}
    path = tmp_path / 'old.pt'
    torch.save(saved, path)
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load',
                        lambda *args, **kwargs: pytest.fail('reject before upstream mutation'))
    with pytest.raises(ValueError, match='full resume'):
        instance.load(path)


def test_reference_sampling_is_bounded_per_motion_task_only():
    alg = algorithm(0, reference_batch_size=2)
    rollout(alg)
    groups = alg._reference_samples()
    assert set(groups) == {0, 2}
    for task, batch in groups.items():
        assert batch.batch_size[0] == 2
        assert torch.all(batch['teacher'][:, 39] == task)


def seed_ppo_momentum(alg):
    for parameter in list(alg.actor.parameters()) + list(alg.critic.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter)
    alg.optimizer.step()


def assert_optimizer_unchanged(before, optimizer):
    after = optimizer.state_dict()
    assert before['param_groups'] == after['param_groups']
    assert before['state'].keys() == after['state'].keys()
    for parameter, state in before['state'].items():
        for key, value in state.items():
            other = after['state'][parameter][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


def test_zero_reference_kl_cannot_move_active_heads_with_existing_ppo_momentum():
    alg = algorithm(0)
    seed_ppo_momentum(alg)
    alg.actor.load_state_dict(alg.reference_actor.state_dict())
    before = copy.deepcopy(alg.actor.state_dict())
    optimizer = copy.deepcopy(alg.optimizer.state_dict())
    metrics = alg._reference_update(reference_groups(alg))
    assert metrics['student_ppo_reference_loss'] == 0
    assert equal_state(before, alg.actor.state_dict())
    assert_optimizer_unchanged(optimizer, alg.optimizer)


def test_reference_reduces_displacement_without_changing_existing_ppo_moments():
    alg = algorithm(0)
    seed_ppo_momentum(alg)
    groups = reference_groups(alg)
    before = mean_distance(alg, groups)
    optimizer = copy.deepcopy(alg.optimizer.state_dict())
    alg._reference_update(groups)
    assert mean_distance(alg, groups) < before
    assert_optimizer_unchanged(optimizer, alg.optimizer)


def test_reference_decay_scales_actual_gradient_correction_and_zero_disables_it():
    full = algorithm(0)
    half = algorithm(0, reference_start_coef=.025, reference_floor_coef=.025)
    disabled = algorithm(0, reference_start_coef=0., reference_floor_coef=0.)
    with torch.no_grad():
        for parameter in full.actor.parameters():
            if parameter.requires_grad:
                parameter.add_(.001)
    for alg in (half, disabled):
        alg.actor.load_state_dict(full.actor.state_dict())
        alg.reference_actor.load_state_dict(full.reference_actor.state_dict())
    before = copy.deepcopy(full.actor.state_dict())
    groups = reference_groups(full)
    for alg in (full, half, disabled):
        alg._reference_update(groups)
    assert equal_state(before, disabled.actor.state_dict())
    assert disabled.reference_updates == 0
    full_delta = torch.cat([(value - before[key]).flatten() for key, value in full.actor.state_dict().items()
                            if key.startswith('mlp.')])
    half_delta = torch.cat([(value - before[key]).flatten() for key, value in half.actor.state_dict().items()
                            if key.startswith('mlp.')])
    assert full_delta.norm() > 0
    assert torch.allclose(full_delta, 2 * half_delta, atol=3e-8, rtol=.01)


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True])
def test_reference_learning_rate_must_be_finite_positive(value):
    with pytest.raises(ValueError, match='reference_learning_rate'):
        algorithm(0, reference_learning_rate=value)


def test_all_task_reference_corrects_each_head_without_touching_critic_or_optimizer():
    alg = algorithm(0, reference_tasks=(0, 1, 2, 3))
    seed_ppo_momentum(alg)
    obs = observations()
    groups = {task: obs[obs['teacher'][:, 39] == task] for task in range(4)}
    before = {task: mean_distance(alg, {task: batch}) for task, batch in groups.items()}
    critic = copy.deepcopy(alg.critic.state_dict())
    optimizer = copy.deepcopy(alg.optimizer.state_dict())
    metrics = alg._reference_update(groups)
    for task, batch in groups.items():
        assert mean_distance(alg, {task: batch}) < before[task]
        # Reference KL uses intrinsic dimensions, including mirrored heads.
        width = 3 if task in (1, 3) else 6
        assert metrics[f'student_ppo_reference_task{task}_kl'] == pytest.approx(
            before[task] * width / (2 * .1 ** 2), rel=1e-4,
        )
    assert equal_state(critic, alg.critic.state_dict())
    assert_optimizer_unchanged(optimizer, alg.optimizer)
    rollout(alg)
    assert set(alg._reference_samples()) == {0, 1, 2, 3}


@pytest.mark.parametrize('tasks', [[0] * 8 + [1] * 4 + [2] * 3 + [3], [2] * 16])
def test_balanced_policy_advantages_use_actual_task_frames_and_preserve_gae(tasks):
    from zbot_rl_mjlab.task_ppo import TaskPPO

    alg = algorithm(0, task_balanced_advantages=True)
    obs = observations()
    obs['teacher'][:, 39] = torch.tensor(tasks)
    obs['actor'] = cast(torch.Tensor, obs['teacher']).repeat(1, 5)
    with torch.inference_mode():
        for step in range(4):
            alg.act(obs)
            alg.process_env_step(obs, torch.arange(16).float() + step, torch.zeros(16), {})
        TaskPPO.compute_returns(alg, obs)
        returns = alg.storage.returns.clone()
        unweighted = alg.storage.advantages.clone()
        alg.compute_returns(obs)
    assert torch.equal(returns, alg.storage.returns)
    count = torch.bincount(torch.tensor(tasks), minlength=4)
    for task in range(4):
        prefix = f'student_ppo_task{task}'
        assert alg._rollout_metrics[f'{prefix}_sample_fraction'] == pytest.approx(float(count[task] / 16))
        if count[task] == 0:
            continue
        expected_weight = 16 / (int((count > 0).sum()) * count[task])
        mask = cast(GroupedRolloutStorage, alg.storage).group_active_masks[..., task].bool()
        assert torch.allclose(alg.storage.advantages[mask], unweighted[mask] * expected_weight)
        assert alg._rollout_metrics[f'{prefix}_policy_weight'] == pytest.approx(float(expected_weight))
    assert torch.isfinite(alg.storage.advantages).all()


def test_policy_kl_telemetry_is_read_only_and_reports_rollout_drift():
    alg = algorithm(0)
    rollout(alg)
    actor = copy.deepcopy(alg.actor.state_dict())
    rng = torch.get_rng_state().clone()
    initial = alg._ppo_kl_metrics()
    assert initial['student_ppo_sampled_policy_kl'] == pytest.approx(0., abs=1e-5)
    assert equal_state(actor, alg.actor.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad():
        for parameter in cast(Any, alg._raw_actor.mlp).heads[3].parameters():
            parameter.add_(.01)
    metrics = alg._ppo_kl_metrics()
    assert metrics['student_ppo_task3_sampled_policy_kl'] > 0
    assert metrics['student_ppo_sampled_policy_kl'] == pytest.approx(
        sum(metrics[f'student_ppo_task{task}_sampled_policy_kl'] for task in range(4)) / 4,
    )
    losses = alg.update()
    for task in range(4):
        for name in ('return_mean', 'value_mean', 'advantage_mean', 'advantage_std',
                     'policy_advantage_std', 'policy_weight', 'sample_fraction', 'sampled_policy_kl'):
            assert torch.isfinite(torch.tensor(losses[f'student_ppo_task{task}_{name}']))


def test_policy_kl_telemetry_caps_forward_batch_per_task(monkeypatch):
    alg = algorithm(0)
    rollout(alg)
    storage = cast(GroupedRolloutStorage, alg.storage)
    # Installed TensorDict stubs disagree with its variadic repeat signature.
    storage.observations = cast(Any, storage.observations).repeat(80, 1)
    storage.group_active_masks = storage.group_active_masks.repeat(80, 1, 1)
    assert storage.distribution_params is not None
    storage.distribution_params = tuple(value.repeat(80, 1, 1) for value in storage.distribution_params)
    forward_batches = []
    get_latent = alg._raw_actor.get_latent

    def tracked(obs):
        forward_batches.append(obs.batch_size[0])
        return get_latent(obs)

    monkeypatch.setattr(alg._raw_actor, 'get_latent', tracked)
    alg._ppo_kl_metrics()
    assert forward_batches == [1024] * 4


def test_v2_contract_is_unchanged_and_balancing_requires_matching_resume():
    alg = algorithm(0)
    contract = alg._contract()
    assert contract == {
        'version': 2, 'critic_warmup_iterations': 0,
        'actor_normalization': 'frozen_supervised_v1', 'objective': 'ppo_student_reference',
        'reference_tasks': [0, 2], 'reference_start_coef': .05, 'reference_floor_coef': .005,
        'reference_decay_iterations': 200, 'reference_batch_size_per_task': 1024,
        'reference_learning_rate': .001, 'reference_optimizer': 'sgd_without_momentum',
    }
    balanced = algorithm(0, task_balanced_advantages=True)
    assert balanced._contract() == contract | {'policy_task_weighting': 'inverse_rollout_frequency_v1'}
    with pytest.raises(ValueError, match='PPO phase'):
        balanced.load(alg.save())
    restored = algorithm(0, task_balanced_advantages=True)
    assert restored.load(balanced.save())


@pytest.mark.parametrize('value', [0, 1, None, 'true'])
def test_task_balancing_requires_boolean(value):
    with pytest.raises(ValueError, match='task_balanced_advantages'):
        algorithm(task_balanced_advantages=value)


@pytest.mark.parametrize('value', [[], [0, 0], [-1], [4], [True]])
def test_reference_tasks_require_unique_valid_task_ids(value):
    with pytest.raises(ValueError, match='reference_tasks'):
        algorithm(reference_tasks=value)
