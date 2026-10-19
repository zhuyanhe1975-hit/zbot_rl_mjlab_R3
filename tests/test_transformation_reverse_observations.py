"""反向任务的可选观测提供当前支撑状态，且不修改该状态。"""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.observation_manager import ObservationManager

from zbot_rl_mjlab.tasks.transformation import mdp
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as reverse
from zbot_rl_mjlab.tasks.transformation.env_cfg import transformation_env_cfg
from zbot_rl_mjlab.tasks.transformation.reverse_observations import (
    SUPPORT_OBSERVATION_COLUMNS,
    reserved_hold_observation,
    support_observation,
)


def env_fixture(enabled=True):
    class Scene(dict):
        env_origins: torch.Tensor

    data = SimpleNamespace(
        body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(2, 1, 4).clone(),
        body_link_ang_vel_w=torch.tensor([[[1.0, -2.0, 3.0]]]).expand(2, 1, 3).clone(),
        site_pos_w=torch.tensor([[[0.0, 0.0, 0.32], [0.0, 0.0, 0.27]]])
        .expand(2, 2, 3)
        .clone(),
        site_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        .expand(2, 2, 4)
        .clone(),
        joint_pos=torch.zeros(2, 6),
        joint_vel=torch.zeros(2, 6),
        default_joint_vel=torch.zeros(2, 6),
        body_link_pos_w=torch.zeros(2, 1, 3),
        body_com_pos_w=torch.zeros(2, 1, 3),
        body_com_lin_vel_w=torch.zeros(2, 1, 3),
    )
    robot = SimpleNamespace(
        data=data,
        find_bodies=lambda name: ([0], [name]),
        find_sites=lambda names, preserve_order=True: ([0, 1], names),
        indexing=SimpleNamespace(body_ids=torch.tensor([0])),
    )
    scene = Scene(
        robot=robot,
        feet_ground_contact=SimpleNamespace(
            data=SimpleNamespace(found=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        ),
    )
    scene.env_origins = torch.tensor([[0.0, 0.0, 0.3]]).expand(2, 3)
    cfg = reverse.snake_to_biped_env_cfg()
    cfg.support_observations = enabled
    cfg.standing_stage_ramp_steps = 100
    env = SimpleNamespace(
        cfg=cfg,
        scene=scene,
        num_envs=2,
        device="cpu",
        common_step_counter=50,
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(1))),
        action_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(
                commanded_target=torch.zeros(2, 6), raw_action=torch.zeros(2, 6)
            )
        ),
    )
    return env


def test_column_order_and_scaled_current_state_are_explicit():
    env = env_fixture()
    env._standing_curriculum_stage = 2
    env._standing_stage_start_step = 25
    state = reverse.support_protection_state(env)
    state.support_armed[:] = torch.tensor([True, False])
    state.support_violation_s[:] = torch.tensor([0.025, 0.0])
    result = support_observation(env)
    assert len(SUPPORT_OBSERVATION_COLUMNS) == 13
    assert result.shape == (2, 13)
    torch.testing.assert_close(
        result[0],
        torch.tensor(
            [0.1, -0.2, 0.3, 1.0, 0.0, 0.2, -0.3, 1.0, -1.0, 1.0, 0.25, 1.0, 0.25]
        ),
    )
    assert result[1, 3:5].tolist() == [0.0, 1.0]
    assert result[1, 9:11].tolist() == [0.0, 0.0]


def test_angular_velocity_is_in_base_frame():
    env = env_fixture()
    root_half = 2**-0.5
    env.scene["robot"].data.body_link_quat_w[:] = torch.tensor(
        [root_half, 0.0, 0.0, root_half]
    )
    env.scene["robot"].data.body_link_ang_vel_w[:] = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(
        support_observation(env)[:, :3],
        torch.tensor([[0.0, -0.1, 0.0]]).expand(2, 3),
        atol=1e-7,
        rtol=1e-6,
    )


def test_reading_does_not_create_or_advance_support_state():
    env = env_fixture()
    before_keys = set(vars(env))
    assert support_observation(env)[:, 9:13].eq(0).all()
    assert set(vars(env)) == before_keys
    assert not hasattr(env, "_support_protection_state")
    state = reverse.support_protection_state(env)
    state.support_armed[:] = True
    state.support_violation_s[:] = 0.05
    state.support_hold_s[:] = 0.4
    snapshots = {name: value.clone() for name, value in vars(state).items()}
    for _ in range(3):
        support_observation(env)
    for name, value in snapshots.items():
        assert torch.equal(getattr(state, name), value)
    assert env.common_step_counter == 50
    assert not hasattr(env, "_standing_curriculum_stage")


@pytest.mark.parametrize("enabled,total", [(False, 30), (True, 43)])
def test_default_shape_stays_checkpoint_compatible_for_both_groups(enabled, total):
    env = env_fixture(enabled)
    for group in env.cfg.observations.values():
        assert list(group.terms)[-1] == "support_state"
        result = torch.cat(
            [term.func(env, **term.params) for term in group.terms.values()], dim=-1
        )
        assert result.shape == (2, total)
    manager = ObservationManager(env.cfg.observations, env)
    for result in manager.compute().values():
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, total)


def test_disabled_observation_needs_no_robot_and_does_not_mutate_env():
    env = SimpleNamespace(
        cfg=SimpleNamespace(support_observations=False), num_envs=3, device="cpu"
    )
    assert support_observation(env).shape == (3, 0)
    assert not hasattr(env, "_support_protection_state")


def test_finite_bounded_outputs_for_invalid_or_extreme_inputs():
    env = env_fixture()
    data = env.scene["robot"].data
    data.body_link_ang_vel_w[:] = float("inf")
    data.site_pos_w[0, 0, 2] = float("nan")
    data.site_pos_w[1, 1, 2] = 1000
    data.site_quat_w[0] = float("nan")
    env.scene["feet_ground_contact"].data.found[:] = float("nan")
    state = reverse.support_protection_state(env)
    state.support_violation_s[:] = float("inf")
    result = support_observation(env)
    assert torch.isfinite(result).all()
    assert result.abs().le(1).all()
    assert result[:, 3:5].eq(0).all()


def test_stage_two_without_ramp_state_matches_explicit_full_objective():
    env = env_fixture()
    env.cfg.curriculum_start_stage = 2
    assert support_observation(env)[:, 11:13].eq(1).all()
    env._standing_stage_start_step = 100
    assert support_observation(env)[:, 12].eq(0).all()


def test_configuration_isolation_preserves_forward_task_and_fresh_defaults():
    first = reverse.snake_to_biped_env_cfg()
    first.support_observations = True
    second = reverse.snake_to_biped_env_cfg()
    forward = transformation_env_cfg()
    assert not second.support_observations
    assert not hasattr(forward, "support_observations")
    assert "support_state" not in forward.observations["actor"].terms
    assert (
        first.observations["actor"].terms["support_state"].func is support_observation
    )


def test_eval_flag_selects_support_observation_config_without_starting_simulator(
    monkeypatch,
):
    from scripts import transformation_eval

    class CapturedConfig(Exception):
        pass

    def capture(cfg, device):
        assert cfg.support_observations
        raise CapturedConfig

    monkeypatch.setattr(transformation_eval, "ManagerBasedRlEnv", capture)
    with pytest.raises(CapturedConfig):
        transformation_eval.evaluate(
            task_id=reverse.TASK_ID, support_observations=True, device="cpu"
        )


def test_eval_coarse_pose_flag_is_forwarded_without_starting_simulator(monkeypatch):
    from scripts import transformation_eval

    class CapturedConfig(Exception):
        pass

    def capture(cfg, device):
        assert cfg.curriculum_joint_error_threshold == 1.0
        assert cfg.curriculum_start_stage == 1
        raise CapturedConfig

    monkeypatch.setattr(transformation_eval, "ManagerBasedRlEnv", capture)
    with pytest.raises(CapturedConfig):
        transformation_eval.evaluate(
            task_id=reverse.TASK_ID,
            curriculum_stage=1,
            curriculum_joint_error_threshold=1.0,
            device="cpu",
        )


@pytest.mark.parametrize("enabled,width", [(False, 30), (True, 43)])
def test_reverse_hold_slot_retains_order_and_width_but_never_exposes_real_timer(
    enabled, width
):
    env = env_fixture(enabled)
    task = mdp.state(env)
    for seconds in (0.0, 0.5, 2.0, 1000.0):
        task.hold_time[:] = seconds
        before = task.hold_time.clone()
        for name in ("actor", "critic"):
            terms = env.cfg.observations[name].terms
            assert list(terms).index("hold_time") == 6
            assert terms["hold_time"].func is reserved_hold_observation
            result = torch.cat(
                [term.func(env, **term.params) for term in terms.values()], dim=-1
            )
            assert result.shape == (2, width)
            assert result[:, 29].eq(0).all()
        assert torch.equal(task.hold_time, before)


def test_reserved_hold_observation_does_not_even_create_timer_state():
    env = SimpleNamespace(num_envs=2, device="cpu")
    assert reserved_hold_observation(env).shape == (2, 1)
    assert not hasattr(env, "_transformation_state")


def test_forward_observation_keeps_real_legacy_hold_timer():
    cfg = transformation_env_cfg()
    env = SimpleNamespace(cfg=cfg, num_envs=2, device="cpu")
    mdp.state(env).hold_time[:] = cfg.success_hold_s
    for name in ("actor", "critic"):
        term = cfg.observations[name].terms["hold_time"]
        assert term.func is mdp.hold_observation
        assert term.func(env).eq(1).all()
