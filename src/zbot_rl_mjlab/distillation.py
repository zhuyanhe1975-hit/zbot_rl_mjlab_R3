"""多教师预热与联合 PPO；部署时只需要学生策略。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .anchor import Anchor
from .anchor import refresh as refresh_protection
from .anchor import select as select_protection
from .anchor import strata as protection_strata

if TYPE_CHECKING:
    from .tasks.universal.deployment import DeploymentCommand, DeploymentCommandCfg
    from .teachers import TeacherEnsemble


@dataclass
class MultiTeacherPpoCfg(RslRlPpoAlgorithmCfg):
    """蒸馏参数全部进入算法配置，可从训练命令行覆盖。"""

    class_name: str = "zbot_rl_mjlab.distillation:MultiTeacherPPO"
    distillation_enabled: bool = True
    teacher_checkpoints: dict[str, str] | None = None
    enabled_teachers: tuple[str, ...] | None = None
    teacher_warmup_enabled: bool = True
    warmup_iterations: int = 100
    anneal_iterations: int = 1000
    start_coef: float = 1.0
    floor_coef: float = 0.02
    distillation_batch_size: int = 1024
    distillation_max_samples: int = 8192
    distillation_updates: int = 32
    replay_iterations: int = 8
    anchor_max_samples: int = 4096
    anchor_coef: float = 100.0
    validation_interval_iterations: int = 25
    validation_required_passes: int = 3
    validation_min_samples: int = 32
    validation_max_mse: float = 0.02
    validation_use_mse: bool = True
    validation_max_std: float = 0.15
    distillation_nll_coef: float = 0.01
    warmup_target_std: float = 0.12


@dataclass
class MultiTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    """保留具体算法类型，使命令行能够解析蒸馏字段。"""

    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=MultiTeacherPpoCfg)
    validation_randomized: bool = True
    robust_validation_interval: int = 0
    console_mode: str = "compact"


def _full_resume(load_cfg: dict | None) -> bool:
    return load_cfg is None or all(load_cfg.get(key, False) for key in ("actor", "critic", "optimizer", "iteration"))


class MultiTeacherPPO(PPO):
    """预热只更新学生 actor，随后 PPO 与退火模仿共用一个优化器。"""

    def __init__(
        self, *args, distillation_enabled=True, teacher_checkpoints=None,
        enabled_teachers=None, teacher_warmup_enabled=True,
        warmup_iterations=100, anneal_iterations=1000,
        start_coef=1.0, floor_coef=0.02, distillation_batch_size=1024,
        distillation_max_samples=8192, distillation_updates=32, replay_iterations=8, anchor_max_samples=4096, anchor_coef=100.0,
        validation_interval_iterations=25, validation_required_passes=3,
        validation_min_samples=32, validation_max_mse=0.02, validation_max_std=0.15, validation_use_mse=True,
        distillation_nll_coef=0.01, warmup_target_std=0.12, **kwargs,
    ):
        if kwargs.get("multi_gpu_cfg") is not None:
            raise ValueError("多教师蒸馏目前只支持单卡，请移除分布式训练配置")
        if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise ValueError("多教师蒸馏目前不支持 RND 或对称扩增")
        if kwargs.get("use_mixed_precision"):
            raise ValueError("多教师蒸馏目前不支持混合精度")
        integer_fields = {
            "warmup_iterations": (warmup_iterations, 0), "anneal_iterations": (anneal_iterations, 1),
            "distillation_batch_size": (distillation_batch_size, 6),
            "distillation_max_samples": (distillation_max_samples, 1),
            "distillation_updates": (distillation_updates, 1),
            "replay_iterations": (replay_iterations, 1),
            "anchor_max_samples": (anchor_max_samples, 3),
            "validation_interval_iterations": (validation_interval_iterations, 1),
            "validation_required_passes": (validation_required_passes, 1),
            "validation_min_samples": (validation_min_samples, 1),
        }
        for name, (value, minimum) in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")
        if not all(math.isfinite(v) for v in (start_coef, floor_coef)) or not 0 <= floor_coef <= start_coef:
            raise ValueError("模仿权重必须有限，且 0 <= floor_coef <= start_coef")
        if distillation_enabled and start_coef == 0:
            raise ValueError("启用蒸馏时 start_coef 必须大于零")
        for name, value in (("anchor_coef", anchor_coef), ("validation_max_mse", validation_max_mse), ("validation_max_std", validation_max_std), ("warmup_target_std", warmup_target_std)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        if not math.isfinite(distillation_nll_coef) or distillation_nll_coef < 0:
            raise ValueError("distillation_nll_coef 必须是有限非负数")
        from .teachers import validate_teacher_selection

        enabled_teachers = validate_teacher_selection(enabled_teachers)
        if type(teacher_warmup_enabled) is not bool:
            raise ValueError("teacher_warmup_enabled 必须是布尔值")
        if teacher_checkpoints:
            unknown = set(teacher_checkpoints) - {"walking", "omni", "transform", "snake"}
            if unknown:
                raise ValueError(f"未知教师：{sorted(unknown)}")
            if enabled_teachers is not None and set(teacher_checkpoints) - set(enabled_teachers):
                raise ValueError("检查点覆盖包含未启用教师")
        super().__init__(*args, **kwargs)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("多教师蒸馏目前只支持非循环网络")
        self.distillation_enabled = distillation_enabled
        self.teacher_checkpoints = teacher_checkpoints
        self.enabled_teachers = enabled_teachers
        self.teacher_warmup_enabled = teacher_warmup_enabled
        self.warmup_iterations = warmup_iterations
        self.anneal_iterations = anneal_iterations
        self.start_coef = start_coef
        self.floor_coef = floor_coef
        self.distillation_batch_size = distillation_batch_size
        self.distillation_max_samples = distillation_max_samples
        self.distillation_updates = distillation_updates
        self.replay_iterations = replay_iterations
        self.anchor_max_samples = anchor_max_samples
        self.anchor_coef = anchor_coef
        self.validation_interval_iterations = validation_interval_iterations
        self.validation_required_passes = validation_required_passes
        self.validation_min_samples = validation_min_samples
        self.validation_max_mse = validation_max_mse
        self.validation_use_mse = validation_use_mse
        self.validation_max_std = validation_max_std
        self.distillation_nll_coef = distillation_nll_coef
        self.warmup_target_std = warmup_target_std
        self.validation_callback: Callable[[], dict[str, float | bool]] | None = None
        self.phase_callback: Callable[[bool], TensorDict] | None = None
        self.progress_callback: Callable[[int, bool], None] | None = None
        self.robust_validation_callback: Callable[[], dict[str, float | bool]] | None = None
        self.robust_validation_interval = 0
        self.robust_validation_metrics: dict[str, float] = {}
        self.anchor_samples_callback: Callable[[], tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None] | None = None
        self._anchor_observation_source: str | None = None
        self._pending_observation: TensorDict | None = None
        self.validation_passes = 0
        self.ppo_start_iteration: int | None = None if teacher_warmup_enabled else 0
        self.validation_metrics: dict[str, float] = {}
        self.distillation_iteration = 0
        self.teachers: TeacherEnsemble | None = None
        self.teacher_provenance: dict = {}
        self._samples: list[tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._anchor: tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._anchor_foundation: Anchor | None = None
        self._anchor_recent: Anchor | None = None
        self._anchor_refresh_count = 0
        self._replay_chunk_counts: list[int] = []
        self._current_chunks = 0
        self._sample_count = 0
        self._queried_count = 0
        self._valid_count: int | torch.Tensor = 0
        self._teacher_counts: dict[str, int | torch.Tensor] = {}
        self._cached_counts = [0] * len(self.teacher_names)
        self._warmup_student_count: int | torch.Tensor = 0
        self._warmup_action_count = 0

    @property
    def teacher_names(self) -> tuple[str, ...]:
        default = self.enabled_teachers or ("walking", "omni", "transform")
        if self.enabled_teachers is None and self.teacher_checkpoints and "snake" in self.teacher_checkpoints:
            default += ("snake",)
        return getattr(self.teachers, "teacher_names", default)

    @property
    def is_warmup(self) -> bool:
        return self.distillation_enabled and self.teacher_warmup_enabled and self.ppo_start_iteration is None

    @property
    def rl_iteration(self) -> int:
        """已完成的真实 PPO 更新次数；教师预热和门控等待不推进扰动课程。"""
        return 0 if self.is_warmup else max(0, self.distillation_iteration - (self.ppo_start_iteration or 0))

    @property
    def imitation_coef(self) -> float:
        if not self.distillation_enabled:
            return 0.0
        elapsed = 0 if self.ppo_start_iteration is None else max(0, self.distillation_iteration - self.ppo_start_iteration)
        fraction = min(1.0, elapsed / self.anneal_iterations)
        return self.start_coef + fraction * (self.floor_coef - self.start_coef)

    @staticmethod
    def _balanced_indices(ids: torch.Tensor, limit: int) -> torch.Tensor:
        """各教师等额抽样，少数教师不会被大量变形样本淹没。"""
        groups = [torch.where(ids == index)[0] for index in ids.unique()]
        present = [group for group in groups if len(group)]
        if not present:
            return ids.new_empty(0)
        quota = max(1, limit // len(present))
        return torch.cat([group[torch.randperm(len(group), device=ids.device)[:quota]] for group in present])[:limit]

    def act(self, obs: TensorDict) -> torch.Tensor:
        if not self._current_chunks:
            # 在新 rollout 加入样本前淘汰旧轮，峰值也不超过配置的回放窗口。
            while len(self._replay_chunk_counts) >= self.replay_iterations:
                del self._samples[:self._replay_chunk_counts.pop(0)]
        if self._pending_observation is not None:
            obs = self._pending_observation
            self._pending_observation = None
        targets = None
        valid = None
        if self.distillation_enabled and self.teachers is not None:
            targets = self.teachers.query()
            valid = (targets.weights > 0) & torch.isfinite(targets.weights) & torch.isfinite(targets.actions).all(dim=-1)
            ids = getattr(targets, "teacher_ids", None)
            if ids is None:
                # 兼容只提供单教师的测试/自定义适配器。
                ids = torch.full_like(targets.weights, -1, dtype=torch.long)
                names = [name for name, count in targets.counts.items() if count]
                if len(names) == 1:
                    ids[valid] = self.teacher_names.index(names[0])
            valid &= (ids >= 0) & (ids < len(self.teacher_names))
            self._queried_count += len(valid)
            self._valid_count += valid.sum()
            for index, name in enumerate(self.teacher_names):
                self._teacher_counts[name] = self._teacher_counts.get(name, 0) + (valid & (ids == index)).sum()
            # 各教师分别限制容量；跨整个 rollout 保留覆盖。
            selected = []
            quota = math.ceil(self.distillation_max_samples / (len(self.teacher_names) * self.storage.num_transitions_per_env))
            capacity = math.ceil(self.distillation_max_samples / len(self.teacher_names))
            if len(self._cached_counts) != len(self.teacher_names):
                self._cached_counts = [0] * len(self.teacher_names)
            for index in range(len(self.teacher_names)):
                candidates = torch.where(valid & (ids == index))[0]
                cached = self._cached_counts[index]
                count = min(len(candidates), quota, capacity - cached, self.distillation_max_samples - self._sample_count)
                if count > 0:
                    chosen = candidates[torch.randperm(len(candidates), device=ids.device)[:count]]
                    selected.append(chosen)
                    self._sample_count += count
                    self._cached_counts[index] += count
            if selected:
                indices = torch.cat(selected)
                with torch.inference_mode(False), torch.no_grad():
                    sample_ids = ids[indices].detach().clone()
                    held_out = torch.zeros(len(indices), device=ids.device, dtype=torch.bool)
                    for index in range(len(self.teacher_names)):
                        group = torch.where(sample_ids == index)[0]
                        if len(group) > 1:
                            held_out[group[:max(1, len(group) // 5)]] = True
                    self._samples.append((obs[indices].detach().clone(), targets.actions.detach()[indices].clone(),
                                          targets.weights.detach()[indices].clone(), sample_ids, held_out))
                    self._current_chunks += 1
        actions = super().act(obs)
        if self.is_warmup:
            # PPO.act already populated the same deterministic distribution mean.
            # Keep its stochastic sample/log-prob/critic work to preserve RNG and storage.
            actions = self.actor.output_mean.detach()
            use_teacher = torch.zeros(len(actions), device=actions.device, dtype=torch.bool)
            if targets is not None and valid is not None:
                use_teacher = valid.clone()
                if self.distillation_iteration >= self.warmup_iterations:
                    # 最低预热后保留各能力中的固定学生试跑组，补充自身偏差下的恢复监督。
                    env_ids = torch.arange(len(actions), device=actions.device)
                    capability_count = 5 if "snake" in self.teacher_names else 4
                    probes = env_ids[(env_ids // capability_count) % 4 == 0][:max(1, len(actions) // 4)]
                    use_teacher[probes] = False
                actions = torch.where(use_teacher[:, None], targets.actions.detach(), actions)
            self._warmup_student_count += (~use_teacher).sum()
            self._warmup_action_count += len(actions)
            self.transition.actions = actions
        return actions

    def _capture_anchor(self):
        """固定成功预热的均衡基础样本；留出标记随样本保存，永不转为训练样本。"""
        supplied = self.anchor_samples_callback() if self.anchor_samples_callback is not None else None
        if supplied is not None:
            device = next(self.actor.parameters()).device
            observations = supplied[0].detach().to(device).clone()
            weights, ids, held_out = (value.detach().to(device).clone() for value in supplied[2:])
            self._anchor_observation_source = "validation_trajectory"
        else:
            observations = cast(TensorDict, torch.cat([sample[0] for sample in self._samples], dim=0))
            weights = torch.cat([sample[2] for sample in self._samples], dim=0)
            ids = torch.cat([sample[3] for sample in self._samples], dim=0)
            held_out = torch.cat([sample[4] for sample in self._samples], dim=0)
            self._anchor_observation_source = "rolling_replay"
        groups = [torch.where(ids == index)[0] for index in range(len(self.teacher_names))]
        quota = min(self.anchor_max_samples // len(self.teacher_names), *(len(group) for group in groups))
        selected = []
        for group in groups:
            train = group[~held_out[group]]
            validation = group[held_out[group]]
            if not quota or not len(train):
                raise RuntimeError("基础保护样本缺少教师训练覆盖，不能进入 PPO")
            # 每个教师至少保留训练行；有余量时也保留独立留出行。
            fixed = train[torch.randperm(len(train), device=ids.device)[:1]]
            if quota > 1 and len(validation):
                fixed = torch.cat((fixed, validation[torch.randperm(len(validation), device=ids.device)[:1]]))
            remaining = group[~torch.isin(group, fixed)]
            selected.append(torch.cat((fixed, remaining[torch.randperm(len(remaining), device=ids.device)[:quota - len(fixed)]])))
        indices = torch.cat(selected)
        if observations['actor'].shape[-1] == 185 and 'critic' in observations and observations['critic'].shape[-1] == 53:
            indices = select_protection(observations, ids, held_out, self.anchor_max_samples, required_teacher_training=True, expected_teacher_count=len(self.teacher_names))
        anchor_observations = observations[indices].detach().clone()
        with torch.no_grad():
            # 保护的是刚通过闭环验收的学生控制器，避免继续逼近旧教师而改变成功步态。
            accepted_actions = self.actor(anchor_observations, stochastic_output=False).detach().clone()
        self._anchor = (anchor_observations, accepted_actions, weights[indices].detach().clone(),
                        ids[indices].detach().clone(), held_out[indices].detach().clone())
        self._anchor_foundation = self._anchor_recent = None
        self._anchor_refresh_count = 0

    def refresh_anchor(self, supplied):
        """只接纳独立成功轨迹；永久保存基础半集，近期半集按困难阶段更新。"""
        if self._anchor is None or supplied is None or not len(supplied[1]):
            return
        observations = supplied[0]
        if observations['actor'].shape[-1] != 185 or 'critic' not in observations or observations['critic'].shape[-1] != 53:
            return
        device = next(self.actor.parameters()).device
        accepted = cast(Anchor, tuple(value.detach().to(device).clone() for value in supplied))
        finite = torch.isfinite(accepted[1]).all(-1) & torch.isfinite(accepted[2]) & (accepted[2] > 0)
        finite &= torch.isfinite(accepted[0]['actor']).all(-1) & torch.isfinite(accepted[0]['critic']).all(-1)
        if not finite.any():
            return
        accepted = cast(Anchor, tuple(value[finite] for value in accepted))
        self._anchor, self._anchor_foundation, self._anchor_recent = refresh_protection(
            self._anchor, self._anchor_foundation, self._anchor_recent, accepted, self.anchor_max_samples,
        )
        self._anchor_refresh_count += 1

    def _imitate(self, coefficient: float) -> tuple[float, dict[str, float]]:
        samples = self._samples + ([self._anchor] if self._anchor is not None and not self.is_warmup else [])
        observations = cast(TensorDict, torch.cat([sample[0] for sample in samples], dim=0))
        actions = torch.cat([sample[1] for sample in samples], dim=0)
        weights = torch.cat([sample[2] for sample in samples], dim=0)
        ids = torch.cat([sample[3] for sample in samples], dim=0)
        held_out = torch.cat([sample[4] for sample in samples], dim=0)
        rolling_count = sum(len(sample[1]) for sample in self._samples)
        anchor_mask = torch.arange(len(actions), device=ids.device) >= rolling_count
        anchor_strata = protection_strata(observations, ids)
        train_indices = torch.where(~held_out & ~anchor_mask)[0]
        anchor_indices = torch.where(~held_out & anchor_mask)[0]
        used = torch.zeros(len(self.teacher_names), dtype=torch.long, device=ids.device)
        anchor_used = torch.zeros_like(used)
        total = anchor_total = 0.0
        for _ in range(self.distillation_updates):
            anchor_limit = max(1, self.distillation_batch_size // 2) if len(anchor_indices) else 0
            rolling_limit = self.distillation_batch_size - anchor_limit if len(train_indices) else 0
            if not len(train_indices):
                anchor_limit = self.distillation_batch_size
            rolling = train_indices[self._balanced_indices(ids[train_indices], rolling_limit)] if rolling_limit else train_indices[:0]
            fixed = anchor_indices[self._balanced_indices(anchor_strata[anchor_indices], anchor_limit)] if anchor_limit else anchor_indices[:0]
            indices = torch.cat((rolling, fixed))
            if not len(indices):
                break
            predicted = self.actor(observations[indices], stochastic_output=False)
            squared_error = (predicted - actions[indices]).square().mean(dim=-1)
            batch_weights = weights[indices]
            fixed_mask = anchor_mask[indices]
            teacher_weights = batch_weights * ~fixed_mask
            anchor_weights = batch_weights * fixed_mask
            teacher_mse = (squared_error * teacher_weights).sum() / teacher_weights.sum().clamp_min(1e-8)
            anchor_mse = (squared_error * anchor_weights).sum() / anchor_weights.sum().clamp_min(1e-8)
            # 滚动教师继续退火；已验收学生的均值保护单独计权，不能被教师项重复覆盖。
            self.actor(observations[indices], stochastic_output=True)
            nll = -self.actor.get_output_log_prob(actions[indices].clamp(-0.999, 0.999)) / actions.shape[-1]
            teacher_loss = teacher_mse + self.distillation_nll_coef * (nll * teacher_weights).sum() / teacher_weights.sum().clamp_min(1e-8)
            protected_std = self.actor.output_std if self.is_warmup else self.actor.output_std[fixed_mask]
            std_penalty = 0.1 * (torch.relu(protected_std - self.warmup_target_std) / self.warmup_target_std).square().mean() if protected_std.numel() else teacher_mse.new_zeros(())
            if self.is_warmup:
                objective = coefficient * (teacher_loss + std_penalty)
            else:
                objective = coefficient * teacher_loss + self.anchor_coef * anchor_mse + std_penalty
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total += teacher_mse.item()
            anchor_total += anchor_mse.item()
            used += torch.bincount(ids[indices], minlength=len(self.teacher_names))
            anchor_used += torch.bincount(ids[fixed], minlength=len(self.teacher_names))
        with torch.no_grad():
            errors = (self.actor(observations, stochastic_output=False) - actions).square().mean(dim=-1)
            self.actor(observations, stochastic_output=True)
            std = self.actor.output_std.mean(dim=-1)
        metrics = {"distillation_anchor_refresh_count": float(self._anchor_refresh_count),
                   "distillation_anchor_straightening_samples": float(((anchor_strata == 4) & anchor_mask).sum()) if observations['actor'].shape[-1] == 185 else 0.,
                   "distillation_anchor_optimized": float(anchor_used.sum()),
                   "distillation_rolling_optimized": float(used.sum() - anchor_used.sum()),
                   "distillation_anchor_loss": anchor_total / self.distillation_updates}
        for index, name in enumerate(self.teacher_names):
            group = ids == index
            validation = group & held_out & ~anchor_mask
            anchor_validation = group & held_out & anchor_mask
            metrics[f"teacher_{name}_anchor_validation_samples"] = float(anchor_validation.sum())
            metrics[f"teacher_{name}_anchor_validation_mse"] = float(errors[anchor_validation].mean()) if anchor_validation.any() else -1.0
            metrics[f"teacher_{name}_anchor_validation_std"] = float(std[anchor_validation].mean()) if anchor_validation.any() else -1.0
            metrics[f"teacher_{name}_cached"] = float((group & ~anchor_mask).sum())
            metrics[f"teacher_{name}_anchor_samples"] = float((group & anchor_mask).sum())
            metrics[f"teacher_{name}_anchor_optimized"] = float(anchor_used[index])
            metrics[f"teacher_{name}_optimized"] = float(used[index])
            metrics[f"teacher_{name}_validation_samples"] = float(validation.sum())
            metrics[f"teacher_{name}_validation_std"] = float(std[validation].mean()) if validation.any() else -1.0
            metrics[f"teacher_{name}_validation_mse"] = float(errors[validation].mean()) if validation.any() else -1.0
            metrics[f"teacher_{name}_training_mse"] = float(errors[group & ~held_out & ~anchor_mask].mean()) if (group & ~held_out & ~anchor_mask).any() else -1.0
        return total / self.distillation_updates, metrics

    def update(self) -> dict[str, float]:
        warmup = self.is_warmup
        coefficient = self.imitation_coef
        if warmup and not self._samples:
            self.storage.clear()
            self._clear_samples()
            raise RuntimeError("蒸馏预热没有有效教师样本，请检查教师适用域或显式关闭蒸馏")
        if warmup:
            self.storage.clear()
            losses = {}
        else:
            losses = super().update()
        self._replay_chunk_counts.append(self._current_chunks)
        imitate = self.distillation_enabled and ((bool(self._samples) and coefficient > 0) or (self._anchor is not None and not warmup))
        error, metrics = self._imitate(coefficient) if imitate else (0.0, {})
        losses.update(metrics)
        losses["distillation"] = error
        losses["distillation_coefficient"] = coefficient
        losses["distillation_anchor_coefficient"] = self.anchor_coef if self.distillation_enabled else 0.0
        losses["distillation_warmup"] = float(warmup)
        # Transfer collection counters together once per update, rather than forcing
        # device synchronization for each teacher on every rollout step.
        collection_counts = torch.stack([
            torch.as_tensor(count, device=self.device)
            for count in (self._valid_count, self._warmup_student_count, *self._teacher_counts.values())
        ]).tolist()
        losses["warmup_student_fraction"] = collection_counts[1] / max(1, self._warmup_action_count)
        losses["distillation_samples"] = float(self._sample_count)
        losses["distillation_replay_samples"] = float(sum(len(sample[1]) for sample in self._samples))
        losses["distillation_replay_iterations"] = float(len(self._replay_chunk_counts))
        losses["distillation_coverage"] = collection_counts[0] / max(1, self._queried_count)
        losses.update({f"teacher_{name}_candidates": float(count) for name, count in zip(self._teacher_counts, collection_counts[2:], strict=True)})
        if warmup:
            eligible = self.distillation_iteration + 1 >= self.warmup_iterations
            checked = eligible and (self.distillation_iteration + 1 - self.warmup_iterations) % self.validation_interval_iterations == 0
            for name in self.teacher_names:
                covered = metrics.get(f"teacher_{name}_validation_samples", 0) >= self.validation_min_samples
                mse = metrics.get(f"teacher_{name}_validation_mse", -1)
                std = metrics.get(f"teacher_{name}_validation_std", -1)
                mse_ready = not self.validation_use_mse or 0 <= mse <= self.validation_max_mse
                losses[f"warmup_blocked_{name}"] = float(not (covered and mse_ready and 0 <= std <= self.validation_max_std))
            if checked:
                result = self.validation_callback() if self.validation_callback is not None else {"validation_passed": False}
                self.validation_metrics = {str(key): float(value) for key, value in result.items()}
                passed = bool(result.get("validation_passed", False)) and all(losses[f"warmup_blocked_{name}"] == 0 for name in self.teacher_names)
                self.validation_passes = self.validation_passes + 1 if passed else 0
                if self.validation_passes >= self.validation_required_passes:
                    self._capture_anchor()
                    self.ppo_start_iteration = self.distillation_iteration + 1
                    if self.phase_callback is not None:
                        self._pending_observation = self.phase_callback(False).detach().clone()
            losses.update(self.validation_metrics)
            losses["warmup_blocked_behavior"] = float(not self.validation_metrics.get("validation_passed", 0))
            losses["warmup_validation_passes"] = float(self.validation_passes)
            losses["warmup_blocked_minimum_iterations"] = float(not eligible)
        elif self.validation_callback is not None:
            # 联合训练继续监测基础能力，避免进入 PPO 后遗忘却只看到奖励上涨。
            if (self.distillation_iteration + 1) % self.validation_interval_iterations == 0:
                result = self.validation_callback()
                self.validation_metrics = {str(key): float(value) for key, value in result.items()}
            losses.update(self.validation_metrics)
        losses["distillation_anchor_samples"] = float(len(self._anchor[1])) if self._anchor is not None else 0.0
        self.distillation_iteration += 1
        if self.progress_callback is not None:
            self.progress_callback(self.rl_iteration, self.is_warmup)
        if not self.is_warmup and self.rl_iteration > 0 and self.robust_validation_callback is not None and self.robust_validation_interval > 0:
            if self.rl_iteration % self.robust_validation_interval == 0:
                result = self.robust_validation_callback()
                self.robust_validation_metrics = {f"robust_{key}": float(value) for key, value in result.items()}
            losses.update(self.robust_validation_metrics)
        losses["student_rl_iteration"] = float(self.rl_iteration)
        self._clear_samples(clear_replay=False)
        return losses

    def _clear_samples(self, clear_replay: bool = True):
        """普通更新只清本轮计数；恢复训练或异常时丢弃整个回放缓存。"""
        if clear_replay:
            self._samples.clear()
            self._replay_chunk_counts.clear()
        self._current_chunks = 0
        self._sample_count = self._queried_count = self._valid_count = 0
        self._teacher_counts.clear()
        self._cached_counts = [0] * len(self.teacher_names)
        self._warmup_student_count = self._warmup_action_count = 0

    def save(self) -> dict:
        """基础保护小样本保存到 CPU；大容量滚动回放不入检查点，恢复后重建。"""
        saved = super().save()
        saved["multi_teacher"] = {
            "iteration": self.distillation_iteration,
            "teacher_names": self.teacher_names,
            "teacher_warmup_enabled": self.teacher_warmup_enabled,
            "provenance": self.teacher_provenance,
            "anchor": tuple(value.detach().to("cpu").clone() for value in self._anchor) if self._anchor is not None else None,
            "anchor_kind": "accepted_student_mean",
            "anchor_observation_source": self._anchor_observation_source,
            # 基础集始终位于组合保护集前缀，只存行数，避免重复保存观测张量。
            "anchor_foundation_count": len(self._anchor_foundation[1]) if self._anchor_foundation is not None else 0,
            "anchor_refresh_count": self._anchor_refresh_count,
            "gate": {"ppo_start_iteration": self.ppo_start_iteration, "passes": self.validation_passes, "metrics": self.validation_metrics},
            "schedule": {
                name: getattr(self, name)
                for name in ("warmup_iterations", "anneal_iterations", "start_coef", "floor_coef", "validation_interval_iterations", "validation_required_passes", "validation_min_samples", "validation_max_mse", "validation_use_mse", "validation_max_std", "distillation_nll_coef", "warmup_target_std", "replay_iterations", "anchor_max_samples", "anchor_coef")
            },
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        full_resume = _full_resume(load_cfg)
        metadata = loaded_dict.get('multi_teacher', {}) if full_resume else {}
        if full_resume:
            saved_warmup = metadata.get("teacher_warmup_enabled", True)
            saved_names = metadata.get("teacher_names")
            if type(saved_warmup) is not bool or saved_warmup != self.teacher_warmup_enabled:
                raise ValueError("检查点教师预热语义不兼容，请显式选择 actor 热启动")
            if saved_names is None:
                # Unversioned checkpoints used only the legacy teacher defaults.
                if self.enabled_teachers is not None:
                    raise ValueError("旧检查点缺少教师选择语义，请显式选择 actor 热启动")
            elif tuple(saved_names) != self.teacher_names:
                raise ValueError("检查点教师选择不兼容，请显式选择 actor 热启动")
        foundation_count = metadata.get('anchor_foundation_count', 0)
        source_anchor = metadata.get('anchor')
        if type(foundation_count) is not int or foundation_count < 0 or (foundation_count and (source_anchor is None or foundation_count > len(source_anchor[1]))):
            raise ValueError('检查点基础保护集前缀无效，拒绝恢复')
        if full_resume and "multi_teacher" not in loaded_dict:
            raise ValueError("完整恢复需要综合任务检查点；旧教师请通过蒸馏使用，热启动请显式选择 actor")
        if load_cfg is None or load_cfg.get("actor", False):
            current = self.actor.state_dict()
            if any(key in current and value.shape != current[key].shape for key, value in loaded_dict.get("actor_state_dict", {}).items()):
                raise ValueError("学生观测或网络维度不兼容：旧 49 维、50 维或 185 维策略不能跨观测结构恢复，请重新训练")
        result = super().load(loaded_dict, load_cfg, strict)
        self.distillation_iteration = int(loaded_dict["multi_teacher"]["iteration"]) if full_resume else 0
        if full_resume:
            # 早期综合检查点没有调度字段，兼容时保留本次配置。
            schedule = loaded_dict["multi_teacher"].get("schedule", {})
            for name in ("warmup_iterations", "anneal_iterations", "start_coef", "floor_coef", "validation_interval_iterations", "validation_required_passes", "validation_min_samples", "validation_max_mse", "validation_use_mse", "validation_max_std", "distillation_nll_coef", "warmup_target_std", "replay_iterations", "anchor_max_samples", "anchor_coef"):
                if name in schedule:
                    setattr(self, name, schedule[name])
        gate = loaded_dict.get("multi_teacher", {}).get("gate", {}) if full_resume else {}
        self.ppo_start_iteration = gate.get("ppo_start_iteration") if self.teacher_warmup_enabled else 0
        self.validation_passes = int(gate.get("passes", 0))
        self.validation_metrics = dict(gate.get("metrics", {}))
        metadata = loaded_dict.get("multi_teacher", {})
        anchor = metadata.get("anchor") if full_resume and metadata.get("anchor_kind") == "accepted_student_mean" else None
        device = next(self.actor.parameters()).device
        self._anchor = tuple(value.detach().to(device).clone() for value in anchor) if anchor is not None else None
        self._anchor_foundation = cast(Anchor, tuple(value[:foundation_count].detach().clone() for value in self._anchor)) if self._anchor is not None and foundation_count else None
        self._anchor_recent = cast(Anchor, tuple(value[foundation_count:].detach().clone() for value in self._anchor)) if self._anchor is not None and self._anchor_foundation is not None else None
        self._anchor_refresh_count = int(metadata.get('anchor_refresh_count', 0)) if full_resume else 0
        self._anchor_observation_source = metadata.get("anchor_observation_source", "rolling_replay") if self._anchor is not None else None
        if self.teacher_warmup_enabled and self.ppo_start_iteration is not None and self._anchor is None:
            # 旧联合训练检查点缺少已验收学生的基础保护数据，必须重过预热门控再开始 PPO。
            self.ppo_start_iteration = None
            self.validation_passes = 0
            self.validation_metrics.clear()
        self._pending_observation = None
        self.robust_validation_metrics.clear()
        self._clear_samples()
        if self.progress_callback is not None:
            self.progress_callback(self.rl_iteration, self.is_warmup)
        return result and full_resume


class MultiTeacherRunner(MjlabOnPolicyRunner):
    """训练时装载冻结教师；回放、导出及学生推理不读取教师。"""

    _motion_directions = ("forward", "backward", "left", "right")

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg, log_dir=None, device="cpu"):
        console_mode = train_cfg.get("console_mode", "compact")
        if console_mode not in ("compact", "full"):
            raise ValueError("console_mode 必须为 compact 或 full")
        super().__init__(env, train_cfg, log_dir, device)
        if console_mode == "compact" and log_dir is not None:
            from .training_console import install_compact_logging

            install_compact_logging(self.logger)
        algorithm = cast(MultiTeacherPPO, self.alg)
        self._training = log_dir is not None
        self._training_distillation = self._training and algorithm.distillation_enabled
        self._evaluator = None
        self._robust_evaluator = None
        self._speed_evaluator = None
        self._speed_course_last_iteration: int | None = None
        self._speed_course_metrics: dict[str, float] = {}
        self._motion_recovery_failures = 0
        robust_interval = train_cfg.get("robust_validation_interval", 0)
        if isinstance(robust_interval, bool) or not isinstance(robust_interval, int) or robust_interval < 0:
            raise ValueError("robust_validation_interval 必须是非负整数")
        algorithm.robust_validation_interval = robust_interval
        if env is not None:
            algorithm.progress_callback = self._publish_student_progress
            self._publish_student_progress(algorithm.rl_iteration, algorithm.is_warmup)
        if self._training_distillation:
            from .teachers import TeacherEnsemble

            algorithm.teachers = TeacherEnsemble(env.unwrapped, device=device, checkpoints=algorithm.teacher_checkpoints,
                                                 enabled_teachers=algorithm.enabled_teachers)
            algorithm.teacher_provenance = self._teacher_provenance()
        speed_command = self._forward_speed_command()
        omni_training = self._training and getattr(getattr(env.unwrapped, 'cfg', None), 'omni_site_frame', False)
        if self._training_distillation or speed_command is not None or omni_training:
            from .universal_evaluation import UniversalEvaluator

            source_cfg = getattr(env.unwrapped, "cfg", None)
            evaluator_cls = UniversalEvaluator
            if omni_training:
                from .universal_omni_evaluation import UniversalOmniEvaluator

                evaluator_cls = UniversalOmniEvaluator
            evaluator = evaluator_cls(device=device, env_cfg=source_cfg, randomized=train_cfg.get("validation_randomized", True))
            self._evaluator = evaluator
            algorithm.validation_callback = self._run_validation
            if self._training_distillation:
                algorithm.anchor_samples_callback = lambda: evaluator.foundation_samples
                algorithm.phase_callback = self._set_training_phase
                self._set_training_phase(algorithm.is_warmup)
            if speed_command is not None:
                speed_cfg = cast("DeploymentCommandCfg", speed_command.cfg)
                self._speed_evaluator = UniversalEvaluator(
                    device=device, env_cfg=source_cfg, randomized=False,
                    forward_speed=speed_cfg.forward_speed_levels[0],
                    forward_tolerance=.25, strict_forward=True, keyboard_protocol=True,
                )
            if robust_interval:
                self._robust_evaluator = evaluator_cls(device=device, env_cfg=source_cfg, randomized=True)
                algorithm.robust_validation_callback = self._run_robust_validation

    def _forward_speed_command(self) -> DeploymentCommand | None:
        """基础综合任务和回放不启用部署学生专属速度课程。"""
        if not getattr(self, "_training", False):
            return None
        if getattr(getattr(self.env.unwrapped, 'cfg', None), 'omni_site_frame', False):
            return None
        from .tasks.universal.mdp import command_term

        command = command_term(self.env.unwrapped)
        cfg = getattr(command, "cfg", None)
        if getattr(cfg, "training", False) and hasattr(command, "motion_speed_stages"):
            return cast("DeploymentCommand", command)
        return None

    def _run_validation(self):
        """基础能力守门，四方向分别使用带滞回的闭环速度课程。"""
        assert self._evaluator is not None
        algorithm = cast(MultiTeacherPPO, self.alg)
        result = dict(self._evaluator(algorithm.actor))
        if getattr(getattr(self.env.unwrapped, 'cfg', None), 'omni_site_frame', False):
            return result
        command = self._forward_speed_command()
        if algorithm.is_warmup:
            return result
        if self._speed_course_last_iteration == algorithm.distillation_iteration:
            return result | self._speed_course_metrics
        self._refresh_evaluation_anchor(self._evaluator)
        if command is None or self._speed_evaluator is None:
            return result
        # 同轮重复读取验收结果不能累计连续通过次数。
        direction_index = (algorithm.distillation_iteration // algorithm.validation_interval_iterations) % len(self._motion_directions)
        direction = self._motion_directions[direction_index]
        levels = command.course_levels(direction)
        stage = command.motion_speed_stages[direction]
        speed = levels[stage]
        self._speed_evaluator.forward_speed = speed
        self._speed_evaluator.motion_direction = direction
        self._speed_evaluator.rl_iteration = algorithm.rl_iteration
        checked = self._speed_evaluator(algorithm.actor)
        self._refresh_evaluation_anchor(self._speed_evaluator, group_zero_only=True)
        success = float(checked.get("forward_success", 0.))
        no_fall = float(checked.get("forward_no_fall_fraction", 1.))
        robust_ascent = algorithm.robust_validation_metrics.get("robust_ascent_success")
        recovery_bad = float(result.get("ascent_success", 1.)) < .9 or (robust_ascent is not None and robust_ascent < .8)
        self._motion_recovery_failures = self._motion_recovery_failures + 1 if recovery_bad else 0
        passed = success >= .9 and no_fall >= .9 and bool(result.get("validation_passed", False)) and robust_ascent is not None and robust_ascent >= .8
        significant_failure = success < .6 or no_fall < .9
        rollback_reason = 0  # 0：无回退；1：该方向显著失败；2：共同恢复能力退化。
        if self._motion_recovery_failures >= 3:
            # 恢复能力是所有方向共同的底线，同次检查最多退一档。
            for name in self._motion_directions:
                command.set_motion_speed_stage(name, max(0, command.motion_speed_stages[name] - 1))
                command.motion_speed_passes[name] = command.motion_speed_failures[name] = 0
            self._motion_recovery_failures = 0
            rollback_reason = 2
        elif significant_failure:
            command.motion_speed_passes[direction] = 0
            command.motion_speed_failures[direction] += 1
            if command.motion_speed_failures[direction] >= 3:
                command.set_motion_speed_stage(direction, max(0, stage - 1))
                command.motion_speed_failures[direction] = 0
                rollback_reason = 1
        elif passed:
            command.motion_speed_failures[direction] = 0
            command.motion_speed_passes[direction] += 1
            if command.motion_speed_passes[direction] >= 3:
                command.set_motion_speed_stage(direction, min(stage + 1, len(levels) - 1))
                command.motion_speed_passes[direction] = 0
        else:
            command.motion_speed_passes[direction] = command.motion_speed_failures[direction] = 0
        self._speed_course_metrics.update({
            "motion_course_direction": float(direction_index),
            "motion_course_rollback_reason": float(rollback_reason),
            "motion_course_recovery_failures": float(self._motion_recovery_failures),
            f"motion_course_{direction}_stage": float(stage),
            f"motion_course_{direction}_speed": float(speed),
            f"motion_course_{direction}_success": success,
            f"motion_course_{direction}_no_fall_fraction": no_fall,
            f"motion_course_{direction}_passed": float(passed),
            f"motion_course_{direction}_passes": float(command.motion_speed_passes[direction]),
            f"motion_course_{direction}_failures": float(command.motion_speed_failures[direction]),
            f"speed_direction_{direction}_success": success,
            f"speed_direction_{direction}_speed": float(speed),
        })
        for name in self._motion_directions:
            current = command.motion_speed_stages[name]
            self._speed_course_metrics[f"motion_course_{name}_next_stage"] = float(current)
            self._speed_course_metrics[f"motion_course_{name}_next_speed"] = float(command.course_levels(name)[current])
            self._speed_course_metrics[f"motion_course_{name}_passes"] = float(command.motion_speed_passes[name])
            self._speed_course_metrics[f"motion_course_{name}_failures"] = float(command.motion_speed_failures[name])
        self._speed_course_last_iteration = algorithm.distillation_iteration
        return result | self._speed_course_metrics

    def _refresh_evaluation_anchor(self, evaluator, group_zero_only=False):
        """接纳部分成功的闭环轨迹；方向验收只接纳受测组，避免混入其他起步组。"""
        algorithm = cast(MultiTeacherPPO, self.alg)
        refresh = getattr(algorithm, "refresh_anchor", None)
        samples = getattr(evaluator, "foundation_samples", None)
        if algorithm.is_warmup or not callable(refresh) or samples is None:
            return
        if group_zero_only:
            selected = samples[3] == 0
            if not selected.any():
                return
            samples = tuple(value[selected] for value in samples)
        refresh(samples)

    def _publish_student_progress(self, iteration: int, pretraining: bool):
        physical = cast(Any, self.env.unwrapped)
        physical._student_rl_iteration = iteration
        physical._student_pretraining = pretraining

    def _run_robust_validation(self):
        assert self._robust_evaluator is not None
        algorithm = cast(MultiTeacherPPO, self.alg)
        self._robust_evaluator.rl_iteration = algorithm.rl_iteration
        result = self._robust_evaluator(algorithm.actor)
        self._refresh_evaluation_anchor(self._robust_evaluator)
        return result

    def _set_training_phase(self, warmup: bool) -> TensorDict:
        from .tasks.universal.mdp import command_term

        command_term(self.env.unwrapped).set_pretraining(warmup)
        algorithm = cast(MultiTeacherPPO, self.alg)
        self._publish_student_progress(algorithm.rl_iteration, warmup)
        # 传感器缓冲区可能来自 inference rollout；重置也需相同上下文。
        with torch.inference_mode():
            observations, _ = self.env.reset()
        return observations

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        algorithm = cast(MultiTeacherPPO, self.alg)
        try:
            # 教师演示的步态时钟从初态开始，不随机跳到回合中部。
            return super().learn(num_learning_iterations, init_at_random_ep_len and not algorithm.is_warmup)
        finally:
            if self._evaluator is not None:
                self._evaluator.close()
            if self._robust_evaluator is not None:
                self._robust_evaluator.close()
            if self._speed_evaluator is not None:
                self._speed_evaluator.close()

    def save(self, path, infos=None):
        if getattr(self, "_training", False):
            from .tasks.universal.mdp import command_term

            course = command_term(self.env.unwrapped).tracking_ema.detach().cpu().tolist()
            infos = {**(infos or {}), "universal_course": course}
            command = self._forward_speed_command()
            if command is not None:
                speed_cfg = cast("DeploymentCommandCfg", command.cfg)
                infos["forward_speed_course"] = {
                    "stage": command.forward_speed_stage,
                    "passes": command.forward_speed_passes,
                    "levels": list(speed_cfg.forward_speed_levels),
                }
                infos["motion_speed_course"] = {
                    "version": 2,
                    "levels": {name: list(command.course_levels(name)) for name in self._motion_directions},
                    "stages": dict(command.motion_speed_stages),
                    "passes": dict(command.motion_speed_passes),
                    "failures": dict(command.motion_speed_failures),
                    "recovery_failures": self._motion_recovery_failures,
                }
        return super().save(path, infos)

    def _decode_motion_course(self, infos, command):
        """完整恢复前验证全部方向；旧前向课程保守迁移，不继承旧通过计数。"""
        levels = {name: list(command.course_levels(name)) for name in self._motion_directions}
        state = infos.get("motion_speed_course")
        if state is not None:
            if not isinstance(state, dict) or type(state.get("version")) is not int or state["version"] != 2 or state.get("levels") != levels:
                raise ValueError("四向速度课程版本或档位与检查点不一致")
            for field in ("stages", "passes", "failures"):
                values = state.get(field)
                if not isinstance(values, dict) or set(values) != set(self._motion_directions):
                    raise ValueError("四向速度课程缺少完整进度")
                for name, value in values.items():
                    bound = len(levels[name]) if field == "stages" else 3
                    if type(value) is not int or not 0 <= value < bound:
                        raise ValueError("四向速度课程进度无效")
            failures = state.get("recovery_failures")
            if type(failures) is not int or not 0 <= failures < 3:
                raise ValueError("四向速度课程恢复失败计数无效")
            return state
        state = {
            "version": 2, "levels": levels,
            "stages": dict.fromkeys(self._motion_directions, 0),
            "passes": dict.fromkeys(self._motion_directions, 0),
            "failures": dict.fromkeys(self._motion_directions, 0),
            "recovery_failures": 0,
        }
        legacy = infos.get("forward_speed_course")
        if legacy is not None:
            if not isinstance(legacy, dict) or legacy.get("levels") != levels["forward"]:
                raise ValueError("旧前进速度课程档位与当前配置不一致")
            stage = legacy.get("stage")
            if type(stage) is not int or not 0 <= stage < len(levels["forward"]):
                raise ValueError("旧前进速度课程进度无效")
            state["stages"]["forward"] = min(stage, 1)
        return state

    def _teacher_provenance(self) -> dict:
        from .teachers import DEFAULT_CHECKPOINTS, SNAKE_CHECKPOINT

        algorithm = cast(MultiTeacherPPO, self.alg)
        paths = DEFAULT_CHECKPOINTS | {'snake': SNAKE_CHECKPOINT} | (algorithm.teacher_checkpoints or {})
        records = {}
        for name in algorithm.teacher_names:
            checkpoint = paths[name]
            path = Path(checkpoint).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            records[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return records

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        algorithm = cast(MultiTeacherPPO, self.alg)
        command = self._forward_speed_command()
        full_resume = _full_resume(load_cfg)
        saved = None
        motion_state = None
        if full_resume and (command is not None or (algorithm.distillation_enabled and algorithm.teachers is not None)):
            saved = torch.load(path, map_location="cpu", weights_only=False)
        if full_resume and command is not None:
            assert saved is not None
            motion_state = self._decode_motion_course(saved.get("infos") or {}, command)
        if _full_resume(load_cfg) and algorithm.distillation_enabled and algorithm.teachers is not None:
            # 在父类修改网络、优化器和环境前检查教师内容；搬迁路径不影响续训。
            assert saved is not None
            previous = saved.get("multi_teacher", {}).get("provenance", {})
            expected = {name: record.get("sha256") for name, record in previous.items()}
            actual = {name: record.get("sha256") for name, record in algorithm.teacher_provenance.items()}
            if expected != actual:
                raise ValueError("教师检查点内容与保存记录不一致，拒绝完整续训；请恢复原教师或显式热启动")
        # Mjlab 基类无条件恢复环境步数，热启动需要保留新环境课程。
        previous_steps = self.env.unwrapped.common_step_counter
        infos = super().load(path, load_cfg, strict, map_location)
        if not _full_resume(load_cfg):
            self.env.unwrapped.common_step_counter = previous_steps
        if getattr(self, "_training", False):
            from .tasks.universal.mdp import command_term

            term = command_term(self.env.unwrapped)
            if _full_resume(load_cfg) and infos and "universal_course" in infos:
                term.tracking_ema.copy_(term.tracking_ema.new_tensor(infos["universal_course"]))
            if full_resume and command is not None:
                assert motion_state is not None
                for name in self._motion_directions:
                    command.set_motion_speed_stage(name, motion_state["stages"][name])
                    command.motion_speed_passes[name] = motion_state["passes"][name]
                    command.motion_speed_failures[name] = motion_state["failures"][name]
                self._motion_recovery_failures = motion_state["recovery_failures"]
            self._speed_course_last_iteration = None
            self._speed_course_metrics = {}
            if self._training_distillation:
                self._set_training_phase(algorithm.is_warmup)
        return infos
