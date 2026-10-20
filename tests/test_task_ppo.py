"""Task routing preserves complete scalar rewards and single-task PPO math."""

import copy
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from zbot_rl_mjlab.grouped_ppo import GroupedRolloutStorage
from zbot_rl_mjlab.task_ppo import (
    TASK_NAMES,
    IndependentTaskCritics,
    TaskPPO,
    task_mask,
)


def observations(tasks=(0, 1, 2, 3)) -> TensorDict:
    raw = torch.randn(len(tasks), 56) * .1
    raw[:, 39] = torch.tensor(tasks)
    return TensorDict({'actor': raw.clone(), 'critic': raw.clone()}, batch_size=[len(tasks)])


def algorithm(obs: TensorDict | None = None, steps=3, **kwargs) -> TaskPPO:
    obs = observations() if obs is None else obs
    sets = {'actor': ['actor'], 'critic': ['critic']}
    actor = MLPModel(obs, sets, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
                     distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)})
    critic = IndependentTaskCritics(obs, sets, 'critic', 1, hidden_dims=(16,), obs_normalization=True)
    storage = RolloutStorage('rl', obs.batch_size[0], steps, obs, [6])
    return TaskPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
                   schedule='fixed', gamma=.9, lam=.8, **kwargs)


def storage(alg: TaskPPO) -> GroupedRolloutStorage:
    return cast(GroupedRolloutStorage, alg.storage)


def test_all_four_tasks_route_complete_reward_from_source_before_reset():
    obs = observations()
    alg = algorithm(obs)
    # Motion, contact/gait, configuration and negative regularization all remain
    # in the original scalar physical reward, without separately normalizing them.
    terms = torch.tensor([[2., .5, 0., -.1], [0., 0., 3., -.2],
                          [1., 0., 0., -.3], [0., 0., 4., -.4]])
    rewards = terms.sum(-1)
    alg.act(obs)
    following = observations((3, 2, 1, 0))
    # Routing was captured in act(), even if the external source buffer mutates.
    obs['critic'][:, 39] = 3
    alg.process_env_step(following, rewards, torch.zeros(4), {})
    st = storage(alg)
    assert torch.equal(st.group_active_masks[0], torch.eye(4))
    assert torch.allclose(st.group_rewards[0], torch.diag(rewards))
    assert torch.allclose(st.rewards[0, :, 0], rewards)


def test_value_gating_and_raw_layout():
    obs = observations()
    alg = algorithm(obs)
    values = alg.critic(obs)
    assert torch.equal(values[torch.eye(4) == 0], torch.zeros(12))
    with pytest.raises(ValueError, match='56-dimensional'):
        task_mask(torch.zeros(4, 59))


def test_task_switch_and_done_break_independent_gae():
    obs = observations((0, 0))
    alg = algorithm(obs, steps=3)
    for parameter in alg.critic.parameters():
        parameter.data.zero_()
    st = storage(alg)
    tasks = torch.tensor([[0, 0], [1, 0], [0, 0]])
    masks = torch.nn.functional.one_hot(tasks, 4).float()
    st.group_active_masks.copy_(masks)
    st.group_rewards.copy_(masks * torch.tensor([[1., 2.], [10., 3.], [100., 4.]])[..., None])
    st.dones[1, 1] = 1
    alg.compute_returns(obs)
    # Env0 changes task every step, so none of its traces can bridge the switch.
    assert st.returns[0, 0, 0].item() == 1
    assert st.returns[1, 0, 1].item() == 10
    assert st.returns[2, 0, 0].item() == 100
    # Env1's explicit done cuts the later value of4 from its earlier return.
    assert st.returns[1, 1, 0].item() == 3
    assert st.returns[0, 1, 0].item() == pytest.approx(2 + .9 * .8 * 3)
    assert torch.equal(st.returns[masks == 0], torch.zeros_like(st.returns[masks == 0]))
    assert torch.isfinite(st.advantages).all()


def scalar_counterpart(alg: TaskPPO, obs: TensorDict, steps: int) -> PPO:
    actor = copy.deepcopy(alg.actor)
    critic = copy.deepcopy(cast(IndependentTaskCritics, alg.critic).critics[0])
    return PPO(actor, cast(MLPModel, critic), RolloutStorage('rl', obs.batch_size[0], steps, obs, [6]),
               num_learning_epochs=1, num_mini_batches=1, schedule='fixed', gamma=.9, lam=.8,
               value_loss_coef=alg.value_loss_coef / 4)


def test_constant_task_matches_scalar_ppo_returns_timeout_advantages_and_actor_gradient():
    torch.manual_seed(11)
    obs = observations((0, 0, 0, 0))
    alg = algorithm(obs)
    scalar = scalar_counterpart(alg, obs, 3)
    for step in range(3):
        # Identical actors, actions, scalar values and physical reward sum.
        torch.manual_seed(100 + step)
        alg.act(obs)
        torch.manual_seed(100 + step)
        scalar.act(obs)
        rewards = torch.tensor([.3, .8, -.1, 1.2]) + step * .2
        extras = {'time_outs': torch.tensor([0., 0., 0., 1.])} if step == 1 else {}
        dones = torch.tensor([0., 0., 0., 1.]) if step == 1 else torch.zeros(4)
        alg.process_env_step(obs, rewards, dones, extras)
        scalar.process_env_step(obs, rewards, dones, extras)
    alg.compute_returns(obs)
    scalar.compute_returns(obs)
    assert torch.allclose(storage(alg).returns[..., 0:1], scalar.storage.returns, atol=1e-6)
    assert torch.allclose(storage(alg).advantages, scalar.storage.advantages, atol=1e-6)
    assert torch.allclose(storage(alg).rewards, scalar.storage.rewards)
    # Compute policy gradients from actual stored actions/log probabilities,
    # rather than merely comparing the advantage arrays.
    for current in (alg, scalar):
        st = current.storage
        flat = st.observations.flatten(0, 1)
        current.actor(flat, stochastic_output=True)
        log_prob = current.actor.get_output_log_prob(st.actions.flatten(0, 1))
        ratio = (log_prob - st.actions_log_prob.flatten()).exp()
        advantages = st.advantages.flatten()
        loss = torch.maximum(-advantages * ratio, -advantages * ratio.clamp(.8, 1.2)).mean()
        current.optimizer.zero_grad()
        loss.backward()
    for left, right in zip(alg.actor.parameters(), scalar.actor.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=1e-6)


def test_critic_value_loss_compensation_matches_scalar_gradient():
    obs = observations((0, 0, 0, 0))
    alg = algorithm(obs, value_loss_coef=1.7)
    scalar = scalar_counterpart(alg, obs, 3)
    targets = torch.tensor([.3, 1.2, -.1, .5])
    vector_targets = targets[:, None] * task_mask(obs['critic'])
    loss = alg.value_loss_coef * (alg.critic(obs) - vector_targets).square().mean()
    scalar_loss = scalar.value_loss_coef * (scalar.critic(obs)[:, 0] - targets).square().mean()
    assert torch.allclose(loss, scalar_loss)
    loss.backward()
    scalar_loss.backward()
    head = cast(IndependentTaskCritics, alg.critic).critics[0]
    for left, right in zip(head.parameters(), scalar.critic.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=1e-6)


def test_small_opposing_cost_does_not_cancel_motion_preference():
    obs = observations((0, 0))
    alg = algorithm(obs, steps=1)
    for parameter in alg.critic.parameters():
        parameter.data.zero_()
    # Moving earns1 at cost.05; standing earns0 at zero cost. Complete reward
    # must prefer moving; separately normalized positive/cost groups can cancel.
    rewards = torch.tensor([1., 0.]) + torch.tensor([-.05, 0.])
    alg.act(obs)
    alg.process_env_step(obs, rewards, torch.ones(2), {})
    alg.compute_returns(obs)
    assert storage(alg).returns[0, 0, 0] > storage(alg).returns[0, 1, 0]
    assert storage(alg).advantages[0, 0, 0] > 0
    assert storage(alg).advantages[0, 1, 0] < 0
    assert torch.equal(storage(alg).returns[..., 1:], torch.zeros_like(storage(alg).returns[..., 1:]))


@pytest.mark.parametrize('metadata', [None, {'version': 1, 'task_names': list(reversed(TASK_NAMES)), 'value_loss_active_scaling': 4}, {'version': 1, 'task_names': list(TASK_NAMES), 'value_loss_active_scaling': 1}])
def test_bad_metadata_and_reward_group_v1_rejected_before_mutation(metadata):
    alg = algorithm()
    before = copy.deepcopy(alg.actor.state_dict())
    saved = copy.deepcopy(alg.save())
    saved['task_ppo'] = metadata
    saved['grouped_ppo'] = {'version': 1, 'group_names': ['motion_tracking', 'configuration', 'contact_gait', 'regularization']}
    saved['actor_state_dict'] = {key: torch.ones_like(value) for key, value in before.items()}
    with pytest.raises(ValueError, match='task version'):
        alg.load(saved, None, True)
    assert all(torch.equal(value, before[key]) for key, value in alg.actor.state_dict().items())


def test_bad_critic_count_rejected_before_mutation_and_actor_only_allowed():
    alg = algorithm()
    before = copy.deepcopy(alg.actor.state_dict())
    saved = copy.deepcopy(alg.save())
    saved['critic_state_dict'] = {key: value for key, value in saved['critic_state_dict'].items() if not key.startswith('critics.3.')}
    saved['actor_state_dict'] = {key: torch.ones_like(value) for key, value in before.items()}
    with pytest.raises(ValueError, match='critic count'):
        alg.load(saved, None, True)
    assert all(torch.equal(value, before[key]) for key, value in alg.actor.state_dict().items())
    assert not alg.load({'actor_state_dict': before}, {'actor': True}, True)


@pytest.mark.parametrize('kwargs', [{'rnd_cfg': {}}, {'symmetry_cfg': {}}, {'multi_gpu_cfg': {}}, {'normalize_advantage_per_mini_batch': True}])
def test_unsupported_features_rejected(kwargs):
    with pytest.raises(ValueError):
        algorithm(**kwargs)


def test_actual_factory_cpu_beta_update_and_roundtrip():
    obs = observations()
    cfg = {
        'algorithm': {'class_name': 'zbot_rl_mjlab.task_ppo:TaskPPO', 'num_learning_epochs': 1, 'num_mini_batches': 1},
        'actor': {'class_name': 'MLPModel', 'hidden_dims': (16,), 'distribution_cfg': {'class_name': 'BetaDistribution', 'action_range': (-1., 1.)}},
        'critic': {'class_name': 'zbot_rl_mjlab.task_ppo:IndependentTaskCritics', 'hidden_dims': (16,)},
        'obs_groups': {'actor': ['actor'], 'critic': ['critic']}, 'num_steps_per_env': 3, 'multi_gpu': None,
    }
    alg = PPO.construct_algorithm(obs, cast(VecEnv, SimpleNamespace(num_actions=6, num_envs=4)), cfg, 'cpu')
    assert isinstance(alg, TaskPPO)
    for step in range(3):
        with torch.inference_mode():
            actions = alg.act(obs)
            assert actions.shape == (4, 6)
            alg.process_env_step(obs, torch.randn(4), torch.zeros(4), {})
    with torch.inference_mode():
        alg.compute_returns(obs)
    losses = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    restored = PPO.construct_algorithm(obs, cast(VecEnv, SimpleNamespace(num_actions=6, num_envs=4)), cfg, 'cpu')
    assert restored.load(alg.save(), None, True)
    assert torch.allclose(alg.actor(obs), restored.actor(obs))
    assert torch.allclose(alg.critic(obs), restored.critic(obs))
