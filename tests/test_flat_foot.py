import math
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_apply

from zbot_rl_mjlab.experiments import flat_foot_velocity_energy as flat
from zbot_rl_mjlab.tasks.walking import mdp


def axis_quat(axis, deg):
    half = math.radians(deg) / 2
    v = torch.tensor(axis, dtype=torch.float32)
    v = v / v.norm()
    return torch.cat((torch.tensor([math.cos(half)]), math.sin(half) * v))


def test_tilt_matches_a_real_rotation_and_ignores_normal_direction():
    for axis in ([1, 0, 0], [0, 1, 0], [1, 1, 0]):
        for deg in (0.0, 3.0, 10.0, 45.0):
            q = axis_quat(axis, deg)
            normal = quat_apply(q[None], torch.tensor([[0.0, 0.0, 1.0]]))
            expected = 1.0 - normal[0, 2] ** 2
            assert torch.allclose(mdp.tilt_sq_from_quat(q), expected, atol=1e-6)
            assert torch.isclose(
                mdp.tilt_sq_from_quat(q),
                torch.tensor(math.sin(math.radians(deg)) ** 2),
                atol=1e-6,
            )
    # foot_1 的脚底法向指向脚外：翻转 180 度后仍视为平坦。
    assert mdp.tilt_sq_from_quat(axis_quat([1, 0, 0], 180.0)) < 1e-6
    assert mdp.tilt_sq_from_quat(axis_quat([1, 0, 0], 183.0)) > 1e-3


class FakeEnv:
    """双脚状态：脚 0 承重并倾斜 deg0，脚 1 离地且高度为 h1。"""

    def __init__(self, deg0, deg1, contact, heights, first):
        quats = torch.stack([axis_quat([0, 1, 0], deg0), axis_quat([0, 1, 0], deg1)])
        self.step_dt = 0.02
        robot = SimpleNamespace(
            find_sites=lambda names, preserve_order=True: ([0, 1], names),
            data=SimpleNamespace(
                site_quat_w=quats[None],
                site_pos_w=torch.tensor(
                    [[[0.0, 0.0, heights[0]], [0.0, 0.0, heights[1]]]]
                ),
            ),
        )
        sensor = SimpleNamespace(
            data=SimpleNamespace(found=torch.tensor([contact], dtype=torch.float32)),
            compute_first_contact=lambda dt: torch.tensor([first]),
        )
        self.scene = {"robot": robot, "feet_ground_contact": sensor}
        self.scene = _Scene(self.scene)


class _Scene(dict):
    env_origins = torch.zeros(1, 3)


def test_penalties_apply_only_in_their_own_phase():
    sigma = flat.FLAT_SIGMA
    tilted = 3.0
    full = 1.0 - math.exp(-(math.sin(math.radians(tilted)) ** 2) / sigma**2)

    # 脚 0 承重且倾斜，脚 1 保持平坦并高高离地。
    env = FakeEnv(tilted, 0.0, [1, 0], [0.0, 0.10], [False, False])
    assert torch.isclose(
        mdp.stance_flat_penalty(env, sigma), torch.tensor([full]), atol=1e-5
    )
    assert mdp.touchdown_flat_penalty(env, sigma).item() == 0.0
    assert mdp.pre_landing_flat_penalty(env, sigma, 0.03).item() == 0.0

    # 同一只脚处于首次接触时间步：落地时应将其计入。
    env = FakeEnv(tilted, 0.0, [1, 0], [0.0, 0.10], [True, False])
    assert torch.isclose(
        mdp.touchdown_flat_penalty(env, sigma), torch.tensor([full]), atol=1e-5
    )

    # 离地且倾斜的脚：高度较高时忽略，低于阈值后开始惩罚。
    env = FakeEnv(0.0, tilted, [1, 0], [0.0, 0.10], [False, False])
    assert mdp.pre_landing_flat_penalty(env, sigma, 0.03).item() == 0.0
    env = FakeEnv(0.0, tilted, [1, 0], [0.0, 0.02], [False, False])
    assert torch.isclose(
        mdp.pre_landing_flat_penalty(env, sigma, 0.03), torch.tensor([full]), atol=1e-5
    )


def test_recipe_replaces_old_foot_term():
    from mjlab.tasks.registry import load_env_cfg

    import zbot_rl_mjlab  # noqa: F401

    for play in (False, True):
        cfg = load_env_cfg(flat.FLAT_FOOT_VELOCITY_ENERGY_TASK_ID, play=play)
        assert "foot_nonparallel_ground" not in cfg.rewards
        for name in (
            "foot_flat_stance",
            "foot_flat_touchdown",
            "foot_flat_pre_landing",
        ):
            assert cfg.rewards[name].weight < 0
