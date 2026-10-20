"""Independent numerical checks for the opt-in reward-group PPO."""

import copy
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from zbot_rl_mjlab.grouped_ppo import (
    GroupedPPO,
    GroupedRolloutStorage,
    IndependentRewardCritics,
    normalize_group_advantages,
)
from zbot_rl_mjlab.tasks.universal.reward_groups import GROUP_NAMES, reward_group_mask


def normalization_count(model: MLPModel) -> int:
    normalizer = cast(EmpiricalNormalization, model.obs_normalizer)
    return int(cast(torch.Tensor, normalizer.count).item())


def observations(tasks=(0, 1, 2, 3)):
    raw = torch.randn(len(tasks), 56) * 0.1
    raw[:, 39] = torch.tensor(tasks)
    return TensorDict({'actor': raw.clone(), 'critic': raw.clone()}, batch_size=[len(tasks)])


def algorithm(obs=None, steps=3, **kwargs):
    obs = observations() if obs is None else obs
    groups = {'actor': ['actor'], 'critic': ['critic']}
    actor = MLPModel(obs, groups, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
                     distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1.0, 1.0)})
    critic = IndependentRewardCritics(obs, groups, 'critic', 1, hidden_dims=(16,), obs_normalization=True)
    storage = RolloutStorage('rl', obs.batch_size[0], steps, obs, [6])
    return GroupedPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
                      schedule='fixed', gamma=0.9, lam=0.8, **kwargs)


def test_independent_critics_mask_values_and_gradients():
    obs = observations()
    model = cast(IndependentRewardCritics, algorithm(obs).critic)
    values = model(obs)
    active = reward_group_mask(obs['critic'])
    assert values.shape == (4, 4)
    assert torch.equal(values[active == 0], torch.zeros_like(values[active == 0]))
    values[:, 2].sum().backward()
    layers = [next(layer for layer in child.modules() if isinstance(layer, torch.nn.Linear))
              for child in model.critics]
    assert layers[2].weight.grad is not None
    assert not torch.equal(layers[0].weight, layers[1].weight)


def test_normalization_active_only_empty_and_singleton():
    values = torch.tensor([[[1., 900., 7., 5.]], [[3., -900., 100., 5.]], [[5., 10., 200., 5.]]])
    active = torch.tensor([[[1., 0., 1., 1.]], [[1., 0., 0., 1.]], [[1., 0., 0., 1.]]])
    result = normalize_group_advantages(values, active)
    assert torch.allclose(result[:, 0, 0], torch.tensor([-1., 0., 1.]))
    assert torch.equal(result[:, :, 1:], torch.zeros_like(result[:, :, 1:]))
    assert torch.isfinite(result).all()


def test_gae_independent_reference_breaks_inactive_traces():
    obs = observations((0, 1))
    alg = algorithm(obs, steps=3)
    st = cast(GroupedRolloutStorage, alg.storage)
    masks = torch.tensor([[[1., 0., 1., 1.], [0., 1., 0., 1.]],
                          [[0., 1., 0., 1.], [0., 1., 0., 1.]],
                          [[1., 0., 1., 1.], [1., 0., 1., 1.]]])
    st.group_active_masks.copy_(masks)
    st.group_rewards.copy_(torch.arange(24).reshape(3, 2, 4) * masks / 10)
    st.values.copy_(torch.arange(24).reshape(3, 2, 4) * masks / 100)
    st.dones[1, 1] = 1
    last = alg.critic(obs).detach()
    last_mask = reward_group_mask(obs['critic'])
    expected = torch.zeros_like(st.returns)
    # Independent scalar recurrence for each environment/group.
    for env in range(2):
        for group in range(4):
            gae = 0.0
            for t in (2, 1, 0):
                next_mask = last_mask[env, group] if t == 2 else masks[t + 1, env, group]
                next_value = last[env, group] if t == 2 else st.values[t + 1, env, group]
                continuation = (1 - int(st.dones[t, env, 0])) * float(next_mask)
                delta = float(st.group_rewards[t, env, group]) + .9 * continuation * float(next_value) - float(st.values[t, env, group])
                gae = (delta + .9 * .8 * continuation * gae) * float(masks[t, env, group])
                expected[t, env, group] = gae + st.values[t, env, group]
    alg.compute_returns(obs)
    assert torch.allclose(st.returns, expected, atol=1e-6)
    normalized = normalize_group_advantages(expected - st.values, masks)
    assert torch.allclose(st.advantages, normalized.sum(-1, keepdim=True))


def test_timeout_bootstrap_clones_groups_and_preserves_scalar_log_rewards():
    obs = observations()
    alg = algorithm(obs)
    alg.act(obs)
    assert alg.transition.values is not None
    values = alg.transition.values.clone()
    active = reward_group_mask(obs['actor'])
    groups = active.float().clone()
    before = groups.clone()
    alg.process_env_step(obs, groups.sum(-1), torch.ones(4), {
        'teacher_reward_groups': groups, 'teacher_reward_group_mask': active,
        'time_outs': torch.tensor([1., 0., 1., 0.]),
    })
    expected = before + .9 * values * torch.tensor([1., 0., 1., 0.])[:, None]
    assert torch.equal(groups, before)
    storage = cast(GroupedRolloutStorage, alg.storage)
    assert torch.allclose(storage.group_rewards[0], expected)
    assert torch.allclose(alg.storage.rewards[0, :, 0], expected.sum(-1))
    active.zero_()
    assert storage.group_active_masks[0].sum() > 0


def test_single_active_group_matches_scalar_ppo_gae_and_normalization():
    obs = observations((0, 0))
    alg = algorithm(obs, steps=3)
    # Use the always-active regularization column alone; other groups zero.
    st = cast(GroupedRolloutStorage, alg.storage)
    st.group_active_masks[..., 3] = 1
    rewards = torch.tensor([[1., 2.], [3., 1.], [0., 4.]])
    st.group_rewards[..., 3] = rewards
    st.values.zero_()
    for parameter in alg.critic.parameters():
        parameter.data.zero_()
    alg.compute_returns(obs)
    expected = torch.zeros(3, 2)
    expected[2] = rewards[2]
    expected[1] = rewards[1] + .9 * .8 * expected[2]
    expected[0] = rewards[0] + .9 * .8 * expected[1]
    assert torch.allclose(st.returns[..., 3], expected)
    normalized = (expected - expected.mean()) / (expected.std() + 1e-8)
    assert torch.allclose(st.advantages[..., 0], normalized)


def test_combined_advantage_is_clipped_once_not_group_by_group():
    # Opposite group advantages cancel before clipping. Separately clipping
    # would introduce a nonzero penalty even though the combined objective is zero.
    ratio = torch.tensor([1.4])
    advantages = torch.tensor([[2., -2.]])
    combined = advantages.sum(-1)
    once = torch.maximum(-combined * ratio, -combined * ratio.clamp(.8, 1.2))
    separate = torch.maximum(-advantages * ratio[:, None], -advantages * ratio.clamp(.8, 1.2)[:, None]).sum(-1)
    assert once.item() == 0
    assert separate.item() > 0


@pytest.mark.parametrize('metadata', [None, {'version': 1, 'group_names': list(reversed(GROUP_NAMES))}, {'version': 2, 'group_names': list(GROUP_NAMES)}])
def test_metadata_rejected_before_mutation(metadata):
    alg = algorithm()
    before = copy.deepcopy(alg.actor.state_dict())
    saved = alg.save()
    saved['grouped_ppo'] = metadata
    saved['actor_state_dict'] = {key: torch.ones_like(value) for key, value in before.items()}
    with pytest.raises(ValueError, match='reward-group'):
        alg.load(saved, None, True)
    for key, value in alg.actor.state_dict().items():
        assert torch.equal(value, before[key])


def test_plain_actor_only_load_allowed_but_full_resume_rejected():
    alg = algorithm()
    plain = {'actor_state_dict': copy.deepcopy(alg.actor.state_dict())}
    assert alg.load(plain, {'actor': True}, True) is False
    with pytest.raises(ValueError):
        alg.load(plain, None, True)


@pytest.mark.parametrize('kwargs', [{'rnd_cfg': {}}, {'symmetry_cfg': {}}, {'multi_gpu_cfg': {}}, {'normalize_advantage_per_mini_batch': True}])
def test_unsupported_options_rejected(kwargs):
    with pytest.raises(ValueError):
        algorithm(**kwargs)


def test_required_reward_metadata():
    obs = observations()
    alg = algorithm(obs)
    alg.act(obs)
    with pytest.raises(ValueError, match='reward groups'):
        alg.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
    assert alg.storage.step == 0


def test_cpu_beta_rollout_update_and_roundtrip():
    torch.manual_seed(7)
    obs = observations()
    alg = algorithm(obs, steps=3)
    before = copy.deepcopy(alg.actor.state_dict())
    for step in range(3):
        with torch.inference_mode():
            actions = alg.act(obs)
            assert actions.shape == (4, 6)
            active = reward_group_mask(obs['actor'])
            groups = torch.randn(4, 4) * active
            alg.process_env_step(obs, groups.sum(-1), torch.zeros(4), {
                'teacher_reward_groups': groups, 'teacher_reward_group_mask': active,
            })
    assert normalization_count(alg.actor) == 0
    with torch.inference_mode():
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert normalization_count(alg.actor) == 12
    critic = cast(IndependentRewardCritics, alg.critic)
    assert all(normalization_count(cast(MLPModel, model)) == 12
               for model in critic.critics)
    assert any(not torch.equal(value, before[key]) for key, value in alg.actor.state_dict().items())
    restored = algorithm(obs)
    assert restored.load(alg.save(), None, True)
    assert torch.allclose(alg.critic(obs), restored.critic(obs))
    assert torch.allclose(alg.actor(obs), restored.actor(obs))


def test_standard_factory_resolves_grouped_algorithm_and_critic():
    obs = observations()
    cfg = {
        'algorithm': {'class_name': 'zbot_rl_mjlab.grouped_ppo:GroupedPPO'},
        'actor': {'class_name': 'MLPModel', 'hidden_dims': (16,), 'distribution_cfg': {'class_name': 'BetaDistribution', 'action_range': (-1., 1.)}},
        'critic': {'class_name': 'zbot_rl_mjlab.grouped_ppo:IndependentRewardCritics', 'hidden_dims': (16,)},
        'obs_groups': {'actor': ['actor'], 'critic': ['critic']},
        'num_steps_per_env': 3, 'multi_gpu': None,
    }
    alg = PPO.construct_algorithm(obs, cast(VecEnv, SimpleNamespace(num_actions=6, num_envs=4)), cfg, 'cpu')
    assert isinstance(alg, GroupedPPO)
    assert alg.storage.values.shape == (3, 4, 4)


def test_critic_count_mismatch_rejected_before_actor_mutation():
    alg = algorithm()
    before = copy.deepcopy(alg.actor.state_dict())
    saved = copy.deepcopy(alg.save())
    saved['critic_state_dict'] = {
        key: value for key, value in saved['critic_state_dict'].items()
        if not key.startswith('critics.3.')
    }
    saved['actor_state_dict'] = {key: torch.ones_like(value) for key, value in before.items()}
    with pytest.raises(ValueError, match='critic count'):
        alg.load(saved, None, True)
    for key, value in alg.actor.state_dict().items():
        assert torch.equal(value, before[key])
