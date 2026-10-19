"""困难阶段保护采样不覆盖旧标签，也不复制稀缺样本。"""

import pytest
import torch
from tensordict import TensorDict

from zbot_rl_mjlab.anchor import refresh, select, strata


def observations(labels):
    labels = torch.as_tensor(labels)
    critic = torch.zeros(len(labels), 53)
    critic[:, 38] = 1
    for i, bucket in enumerate(labels.tolist()):
        phase = 0 if bucket < 4 else 3 if bucket in (4, 5) else 1 if bucket == 6 else 2
        critic[i, 39 + phase] = 1
        if bucket < 4:
            critic[i, 43 + (bucket // 2)] = 1 if bucket % 2 == 0 else -1
        if bucket == 4:
            critic[i, 38] = 0
        elif bucket in (6, 7):
            critic[i, 38] = -1
    return TensorDict(
        {"actor": torch.zeros(len(labels), 185), "critic": critic},
        batch_size=[len(labels)],
    )


def anchor(labels, offset=0):
    obs = observations(labels)
    n = len(labels)
    return (
        obs,
        torch.arange(n).float()[:, None].expand(n, 6).clone() + offset,
        torch.ones(n),
        torch.arange(n) % 3,
        torch.arange(n) % 2 == 0,
    )


def test_eight_buckets_and_small_schema_fallback():
    obs = observations(range(8))
    ids = torch.arange(8) % 3
    assert strata(obs, ids).tolist() == list(range(8))
    for width in (1, 50):
        old = TensorDict({"actor": torch.zeros(8, width)}, batch_size=[8])
        torch.testing.assert_close(strata(old, ids), ids)


def test_rare_nineteen_rows_are_kept_in_large_steady_pool():
    obs = observations([4] * 19 + [0] * 4076)
    ids = torch.zeros(4095, dtype=torch.long)
    held = torch.arange(4095) % 5 == 0
    chosen = select(obs, ids, held, 512)
    assert len(chosen) == len(chosen.unique()) == 512
    assert (chosen < 19).sum() == 19
    assert held[chosen[chosen < 19]].any()
    assert (~held[chosen[chosen < 19]]).any()


def test_balanced_selection_keeps_each_bucket_train_and_validation():
    obs = observations([i for i in range(8) for _ in range(4)])
    held = torch.arange(32) % 2 == 0
    selected = select(obs, torch.arange(32) % 3, held, 16)
    buckets = strata(obs, torch.arange(32) % 3)
    for i in range(8):
        subset = selected[buckets[selected] == i]
        assert len(subset) == 2
        assert held[subset].any() and (~held[subset]).any()


def test_required_teacher_training_and_short_inputs():
    obs = observations([0] * 6)
    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    held = torch.tensor([True, False] * 3)
    chosen = select(obs, ids, held, 3, required_teacher_training=True)
    assert not held[chosen].any()
    assert set(ids[chosen].tolist()) == {0, 1, 2}
    assert len(select(obs, ids, held, 100)) == 6
    with pytest.raises(ValueError, match="3"):
        select(obs, ids, held, 2, required_teacher_training=True)
    with pytest.raises(ValueError, match="教师"):
        select(
            obs, ids, torch.ones(6, dtype=torch.bool), 6, required_teacher_training=True
        )
    assert select(observations([]), ids[:0], held[:0], 5).numel() == 0


def test_refresh_freezes_original_foundation_and_preserves_new_labels():
    old = anchor(list(range(8)) * 8)
    accepted = anchor([0, 1, 2, 3, 6, 7] * 6, offset=1000)
    combined, foundation, recent = refresh(old, None, None, accepted, 32)
    assert len(combined[1]) == 32
    assert len(foundation[1]) == len(recent[1]) == 16
    assert (foundation[1] < 1000).all() and (recent[1] >= 1000).all()
    original = tuple(x.clone() for x in foundation)
    newer = anchor([4, 5] * 8, offset=2000)
    result, foundation, recent = refresh(combined, foundation, recent, newer, 32)
    for before, after in zip(original, foundation):
        if isinstance(before, TensorDict):
            for key, value in before.items():
                torch.testing.assert_close(value, after[key])
        else:
            torch.testing.assert_close(before, after)
    assert (recent[1] >= 2000).any()
    for actions, held in ((foundation[1], foundation[4]), (recent[1], recent[4])):
        original_row = actions[:, 0].long() % 1000
        torch.testing.assert_close(held, original_row % 2 == 0)
    result[1].zero_()
    assert old[1].sum() > 0 and accepted[1].sum() > 0 and newer[1].sum() > 0


@pytest.mark.parametrize("limit", [1, 2, 7, 32])
def test_capacity_and_empty_refresh(limit):
    old = anchor(list(range(8)))
    new = anchor(list(range(8)), offset=100)
    result, foundation, recent = refresh(old, None, None, new, limit)
    assert len(result[1]) <= limit
    unchanged, _, _ = refresh(result, foundation, recent, anchor([]), limit)
    torch.testing.assert_close(unchanged[1], result[1])
    unchanged[1].zero_()
    assert result[1].sum() > 0
    assert refresh(None, None, None, new, limit) == (None, None, None)


def test_refresh_results_are_detached_independent_copies():
    old = anchor(list(range(8)) * 2)
    supplied = anchor(list(range(8)), offset=100)
    supplied[1].requires_grad_(True)
    combined, foundation, recent = refresh(old, None, None, supplied, 16)
    assert not combined[1].requires_grad and not recent[1].requires_grad
    old_actor = old[0]["actor"].clone()
    combined[0]["actor"].fill_(7)
    recent[1].zero_()
    torch.testing.assert_close(old[0]["actor"], old_actor)
    assert supplied[1].sum() > 0 and combined[1].sum() > 0
    assert foundation[0]["actor"].eq(0).all()


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_selection_capacity(limit):
    sample = anchor([0, 1, 2])
    with pytest.raises(ValueError):
        select(sample[0], sample[3], sample[4], limit)
