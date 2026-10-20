"""One complete scalar reward stream and independent value function per FSM task.

Unlike reward-group normalization, this preserves relative weights of every physical
reward term inside a task. One shared actor receives task-normalized advantages.
"""

from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .grouped_ppo import (
    GroupedRolloutStorage,
    IndependentRewardCritics,
    normalize_group_advantages,
)
from .tasks.universal.observation_layout import teacher_task_ids

TASK_NAMES = ('biped_motion', 'descent', 'snake_motion', 'ascent')


def task_mask(raw: torch.Tensor) -> torch.Tensor:
    """Read the unnormalized scalar FSM code; legacy observations are unsupported."""
    if raw.shape[-1] != 56:
        raise ValueError('TaskPPO requires raw 56-dimensional teacher observations')
    return torch.nn.functional.one_hot(teacher_task_ids(raw), len(TASK_NAMES)).to(raw.dtype)


class IndependentTaskCritics(IndependentRewardCritics):
    """Four independent MLPs; only the current task's scalar prediction is active."""

    def forward(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        return torch.cat([model(obs, **kwargs) for model in self.critics], dim=-1) * task_mask(self._raw(obs))


class TaskPPO(PPO):
    """Task-routed scalar rewards, GAE, value estimates and advantage statistics."""

    def __init__(self, actor: Any, critic: Any, storage: RolloutStorage, **kwargs: Any) -> None:
        if kwargs.get('normalize_advantage_per_mini_batch', False):
            raise ValueError('TaskPPO requires per-mini-batch advantage normalization disabled')
        for name in ('rnd_cfg', 'symmetry_cfg', 'multi_gpu_cfg'):
            if kwargs.get(name) is not None:
                raise ValueError(f'TaskPPO does not support {name}')
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError('TaskPPO does not support recurrent models')
        if not isinstance(critic, IndependentTaskCritics):
            raise TypeError('TaskPPO requires IndependentTaskCritics')
        super().__init__(actor, cast(Any, critic), GroupedRolloutStorage(storage), **kwargs)
        # Upstream loss averages four columns, exactly one of which is active.
        # Undo that factor so a pure-task critic has the scalar PPO loss/gradient.
        self.value_loss_coef *= len(TASK_NAMES)

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        # Snapshot routing before env.step can reset or change the FSM observation.
        assert self.transition.observations is not None
        raw = cast(IndependentTaskCritics, self._raw_critic)._raw(self.transition.observations)
        cast(Any, self.transition).group_active_masks = task_mask(raw).detach().clone()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        if rewards.shape not in ((self.storage.num_envs,), (self.storage.num_envs, 1)):
            raise ValueError('TaskPPO requires one complete scalar reward per environment')
        masks = cast(Any, self.transition).group_active_masks
        routed = rewards.detach().reshape(-1, 1) * masks
        super().process_env_step(obs, routed, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        st = cast(GroupedRolloutStorage, self.storage)
        last_values = self.critic(obs).detach()
        last_active = task_mask(cast(IndependentTaskCritics, self._raw_critic)._raw(obs))
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_active = last_active if step == st.num_transitions_per_env - 1 else st.group_active_masks[step + 1]
            continuation = (1.0 - st.dones[step].float()) * next_active
            active = st.group_active_masks[step]
            delta = st.group_rewards[step] + self.gamma * continuation * next_values - st.values[step]
            advantage = (delta + self.gamma * self.lam * continuation * advantage) * active
            st.returns[step].copy_(advantage + st.values[step])
        normalized = normalize_group_advantages(st.returns - st.values, st.group_active_masks)
        st.advantages = normalized.sum(dim=-1, keepdim=True)

    @staticmethod
    def _metadata() -> dict:
        return {'version': 1, 'task_names': list(TASK_NAMES), 'value_loss_active_scaling': len(TASK_NAMES)}

    def save(self) -> dict:
        result = super().save()
        result['task_ppo'] = self._metadata()
        return result

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        actor_only = load_cfg is not None and load_cfg.get('actor', False) and not any(
            load_cfg.get(key, False) for key in ('critic', 'optimizer', 'iteration', 'rnd')
        )
        if not actor_only and loaded_dict.get('task_ppo') != self._metadata():
            raise ValueError('TaskPPO full resume requires matching task version/order/value-loss scaling')
        if not actor_only and (load_cfg is None or load_cfg.get('critic', False)):
            current = self._raw_critic.state_dict()
            incoming = loaded_dict.get('critic_state_dict', {})
            if current.keys() != incoming.keys() or any(
                current[key].shape != incoming[key].shape for key in current
            ):
                raise ValueError('TaskPPO critic count or architecture does not match checkpoint')
        return super().load(loaded_dict, load_cfg, strict)
