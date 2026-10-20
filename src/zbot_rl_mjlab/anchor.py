"""按运动方向和变形阶段保护成功学生轨迹，固定基础标签并补充近期成功。"""

from typing import cast

import torch
from tensordict import TensorDict

Anchor = tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def strata(observations: TensorDict, teacher_ids: torch.Tensor) -> torch.Tensor:
    """部署学生按运动能力分层；原任务或测试代理沿用教师分组。"""
    actor, critic = observations.get("actor"), observations.get("critic")
    if (
        actor is None
        or critic is None
        or actor.shape[-1] != 185
        or critic.shape[-1] != 53
    ):
        return teacher_ids.long().clone()
    mode, phase = critic[:, 38], critic[:, 39:43].argmax(-1)
    velocity = critic[:, 43:46]
    result = torch.full_like(teacher_ids, 7, dtype=torch.long)
    x_axis = velocity[:, 0].abs() >= velocity[:, 1].abs()
    direction = torch.where(
        x_axis, (velocity[:, 0] < 0).long(), 2 + (velocity[:, 1] < 0).long()
    )
    moving = (phase == 0) & (mode > 0) & (velocity[:, :2].abs().amax(-1) > 0)
    result[moving] = direction[moving]
    result[(phase == 1) & (mode < 0)] = 6
    result[(phase == 3) & (mode > 0)] = 5
    snake_moving = (phase == 2) & (mode < 0) & (velocity[:, :2].abs().amax(-1) > 0)
    result[snake_moving] = 8
    result[mode == 0] = 4
    return result


def select(
    observations: TensorDict,
    ids: torch.Tensor,
    held_out: torch.Tensor,
    limit: int,
    required_teacher_training: bool = False,
    expected_teacher_count: int = 3,
) -> torch.Tensor:
    """稀有层先取、轮流补足容量；不复制行，不改变训练与留出身份。"""
    if type(limit) is not int or limit < 0:
        raise ValueError("保护样本容量必须是非负整数")
    selected = []
    if required_teacher_training:
        if type(expected_teacher_count) is not int or expected_teacher_count < 1:
            raise ValueError("教师数量必须是正整数")
        if limit < expected_teacher_count:
            raise ValueError(f"教师训练保护至少需要 {expected_teacher_count} 个样本容量")
        for teacher in range(expected_teacher_count):
            candidates = torch.where((ids == teacher) & ~held_out)[0]
            if not len(candidates):
                raise ValueError(f"教师 {teacher} 缺少训练样本，不能建立基础保护")
            selected.append(
                candidates[torch.randint(len(candidates), (), device=ids.device)]
            )
    labels = strata(observations, ids)
    queues = []
    initial = torch.stack(selected) if selected else ids.new_empty(0)
    for label in labels.unique():
        group = torch.where(labels == label)[0]
        chosen = initial[labels[initial] == label]
        remaining = group[~torch.isin(group, initial)]
        train, validation = (
            remaining[~held_out[remaining]],
            remaining[held_out[remaining]],
        )
        train = train[torch.randperm(len(train), device=ids.device)]
        validation = validation[torch.randperm(len(validation), device=ids.device)]
        # 已预留训练行时先补留出行，否则先训练再留出。
        first, second = (
            (validation, train)
            if len(chosen) and (~held_out[chosen]).any()
            else (train, validation)
        )
        paired = min(len(first), len(second))
        queue = torch.cat(
            (
                torch.stack((first[:paired], second[:paired]), -1).flatten(),
                first[paired:],
                second[paired:],
            )
        )
        queues.append((len(group), queue))
    queues.sort(key=lambda item: item[0])
    capacity = min(limit, len(ids))
    position = 0
    while len(selected) < capacity:
        for _, queue in queues:
            if position < len(queue) and len(selected) < capacity:
                selected.append(queue[position])
        position += 1
    return (
        torch.stack(selected).long() if selected else ids.new_empty(0, dtype=torch.long)
    )


def _clone(sample: Anchor | None) -> Anchor | None:
    return (
        None
        if sample is None
        else cast(Anchor, tuple(value.detach().clone() for value in sample))
    )


def _take(sample: Anchor, indices: torch.Tensor) -> Anchor:
    return cast(Anchor, tuple(value[indices].detach().clone() for value in sample))


def _combine(first: Anchor, second: Anchor) -> Anchor:
    return cast(
        Anchor,
        tuple(
            torch.cat((left, right), 0).detach().clone()
            for left, right in zip(first, second)
        ),
    )


def refresh(
    existing: Anchor | None,
    foundation: Anchor | None,
    recent: Anchor | None,
    supplied: Anchor | None,
    max_samples: int,
) -> tuple[Anchor | None, Anchor | None, Anchor | None]:
    """永久冻结旧基础半集，其余容量只保留已验收成功学生的新旧标签。"""
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("保护样本容量必须是正整数")
    if existing is None or supplied is None or not len(supplied[1]):
        return _clone(existing), _clone(foundation), _clone(recent)
    if foundation is None:
        indices = select(existing[0], existing[3], existing[4], max_samples // 2)
        foundation = _take(existing, indices)
    elif len(foundation[1]) > max_samples:
        raise ValueError("新容量不能小于已冻结基础保护集")
    pool = supplied if recent is None else _combine(recent, supplied)
    indices = select(pool[0], pool[3], pool[4], max_samples - len(foundation[1]))
    updated_recent = _take(pool, indices)
    frozen = _clone(foundation)
    return _combine(foundation, updated_recent), frozen, updated_recent
