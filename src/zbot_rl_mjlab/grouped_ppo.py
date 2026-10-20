"""Reward-group critics and GAE for the privileged universal teacher only.

The actor uses one PPO ratio and clips the *sum* of normalized group advantages.
The inherited value objective averages squared errors over all four critic columns;
this is a local adaptation, not a claim of an exact reproduction of the paper.
"""

from typing import Any, cast

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn

from .tasks.universal.reward_groups import GROUP_NAMES, reward_group_mask


class IndependentRewardCritics(nn.Module):
    """Four independent scalar critics, with inactive predictions fixed at zero."""

    is_recurrent = False

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, **kwargs: Any,
    ) -> None:
        super().__init__()
        if output_dim != 1 or kwargs.get("distribution_cfg") is not None:
            raise ValueError("IndependentRewardCritics requires deterministic scalar base critics")
        self.critics = nn.ModuleList([
            MLPModel(obs, obs_groups, obs_set, 1, **kwargs) for _ in GROUP_NAMES
        ])
        self.obs_groups = obs_groups[obs_set]
        self.obs_dim = self.critics[0].obs_dim

    def _raw(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[key] for key in self.obs_groups], dim=-1)

    def forward(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        return torch.cat([model(obs, **kwargs) for model in self.critics], dim=-1) * reward_group_mask(self._raw(obs))

    def update_normalization(self, obs: TensorDict) -> None:
        for model in self.critics:
            cast(MLPModel, model).update_normalization(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        pass

    def get_hidden_state(self) -> None:
        return None


class GroupedRolloutStorage(RolloutStorage):
    """Keep scalar rewards for runner metrics and vector targets for the critics."""

    def __init__(self, original: RolloutStorage) -> None:
        super().__init__(
            original.training_type, original.num_envs, original.num_transitions_per_env,
            original.observations[0], original.actions_shape, original.device,
        )
        shape = (self.num_transitions_per_env, self.num_envs, len(GROUP_NAMES))
        self.values = torch.zeros(shape, device=self.device)
        self.returns = torch.zeros_like(self.values)
        self.group_rewards = torch.zeros_like(self.values)
        self.group_active_masks = torch.zeros_like(self.values)

    def add_transition(self, transition: Any) -> None:
        rewards = transition.rewards
        self.group_rewards[self.step].copy_(rewards)
        self.group_active_masks[self.step].copy_(transition.group_active_masks)
        transition.rewards = rewards.sum(dim=-1)
        try:
            super().add_transition(transition)
        finally:
            transition.rewards = rewards


def normalize_group_advantages(advantages: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Normalize each group using only its active samples; empty groups stay zero.

    Sample standard deviation matches upstream PPO when exactly one group is active.
    Singleton groups have zero normalized advantage instead of NaN.
    """
    if advantages.shape != active.shape or advantages.shape[-1] != len(GROUP_NAMES):
        raise ValueError("Expected matching [..., 4] advantages and active masks")
    axes = tuple(range(advantages.ndim - 1))
    count = active.sum(dim=axes, keepdim=True)
    mean = (advantages * active).sum(dim=axes, keepdim=True) / count.clamp_min(1)
    centered = (advantages - mean) * active
    variance = centered.square().sum(dim=axes, keepdim=True) / (count - 1).clamp_min(1)
    return centered / (variance.sqrt() + 1e-8)


class GroupedPPO(PPO):
    """Independent reward-group GAE, combined before ordinary PPO clipping."""

    def __init__(self, actor: Any, critic: Any, storage: RolloutStorage, **kwargs: Any) -> None:
        if kwargs.get("normalize_advantage_per_mini_batch", False):
            raise ValueError("GroupedPPO requires per-mini-batch advantage normalization disabled")
        for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if kwargs.get(name) is not None:
                raise ValueError(f"GroupedPPO does not support {name}")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("GroupedPPO does not support recurrent models")
        if not isinstance(critic, IndependentRewardCritics):
            raise TypeError("GroupedPPO requires IndependentRewardCritics")
        super().__init__(actor, cast(Any, critic), GroupedRolloutStorage(storage), **kwargs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        if "teacher_reward_groups" not in extras or "teacher_reward_group_mask" not in extras:
            raise ValueError("GroupedPPO requires teacher reward groups and active masks")
        groups = extras["teacher_reward_groups"]
        masks = extras["teacher_reward_group_mask"]
        expected = (self.storage.num_envs, len(GROUP_NAMES))
        if groups.shape != expected or masks.shape != expected:
            raise ValueError(f"Teacher reward groups and masks must have shape {expected}")
        # Clone before upstream timeout bootstrapping and before command/reset buffers change.
        cast(Any, self.transition).group_active_masks = masks.detach().clone()
        super().process_env_step(obs, groups.detach().to(dtype=rewards.dtype).clone(), dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        st = cast(GroupedRolloutStorage, self.storage)
        last_values = self.critic(obs).detach()
        last_active = reward_group_mask(cast(IndependentRewardCritics, self._raw_critic)._raw(obs))
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

    def save(self) -> dict:
        result = super().save()
        result["grouped_ppo"] = {"version": 1, "group_names": list(GROUP_NAMES)}
        return result

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        # Validate before *any* actor, critic or optimizer state can be mutated.
        actor_only = load_cfg is not None and load_cfg.get("actor", False) and not any(
            load_cfg.get(key, False) for key in ("critic", "optimizer", "iteration", "rnd")
        )
        expected = {"version": 1, "group_names": list(GROUP_NAMES)}
        if not actor_only and loaded_dict.get("grouped_ppo") != expected:
            raise ValueError("GroupedPPO full resume requires matching reward-group version and order")
        if not actor_only and (load_cfg is None or load_cfg.get("critic", False)):
            current = self._raw_critic.state_dict()
            incoming = loaded_dict.get("critic_state_dict", {})
            if current.keys() != incoming.keys() or any(
                current[key].shape != incoming[key].shape for key in current
            ):
                raise ValueError("GroupedPPO critic count or architecture does not match checkpoint")
        return super().load(loaded_dict, load_cfg, strict)
