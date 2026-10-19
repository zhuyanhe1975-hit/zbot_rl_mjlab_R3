"""热启动隔离与物理力／力矩调度检查。"""

import json
import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.experiments.transformation import (
    SnakeToBipedBeta,
    SnakeToBipedRobust,
)
from zbot_rl_mjlab.rl import (
    HotRewardController,
    RobustSnakeToBipedRunner,
    SnakeToBipedHotRewardRunner,
)
from zbot_rl_mjlab.tasks.transformation import robust
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as reverse
from zbot_rl_mjlab.tasks.transformation.actions import (
    MirroredJointTargetRateActionCfg,
    expand_mirror,
    mirrored_last_action,
)
from zbot_rl_mjlab.tasks.transformation.robust import (
    ROBUST_TASK_ID,
    RampedBaseImpulse,
    RobustSnakeToBipedEnvCfg,
    disturbance_scale,
)


@pytest.mark.parametrize("play", [False, True])
def test_robust_task_preserves_the_checkpoint_contract_and_full_objective(play):
    cfg = load_env_cfg(ROBUST_TASK_ID, play=play)
    original = load_env_cfg(SnakeToBipedBeta.task_id, play=play)
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    assert cfg.pretrained_checkpoint is None
    assert cfg.scene.entities["robot"].init_state == original.scene.entities["robot"].init_state
    action_values = asdict(cfg.actions["joint_pos"])
    original_action_values = asdict(original.actions["joint_pos"])
    assert action_values.pop("max_target_speed") == 2 * math.pi
    original_action_values.pop("max_target_speed")
    assert action_values == original_action_values
    assert isinstance(cfg.actions["joint_pos"], MirroredJointTargetRateActionCfg)
    assert not isinstance(cfg.actions["joint_pos"], robust.RecoveryJointTargetRateActionCfg)
    for name, group in cfg.observations.items():
        assert group.terms["last_action"].func is mirrored_last_action
        for term in group.terms.keys() - {"last_action"}:
            assert group.terms[term] == original.observations[name].terms[term]
    assert cfg.sim == original.sim
    assert set(cfg.terminations) == {"update_state", "time_out", "fallen_after_standing", "early_base_ground_contact", "abandoned_exploration"}
    assert cfg.terminations["time_out"].time_out
    assert cfg.terminations["update_state"].func is robust.update_fixed_standing_state
    assert cfg.terminations["time_out"] == original.terminations["time_out"]
    assert cfg.terminations["fallen_after_standing"].func is robust.fallen_after_standing
    assert cfg.terminations["early_base_ground_contact"].func is robust.early_base_ground_contact
    assert cfg.terminations["abandoned_exploration"].func is robust.abandoned_exploration
    assert any(sensor.name == "base_ground_contact" for sensor in cfg.scene.sensors)
    assert "recovery_active" not in cfg.metrics
    assert cfg.events["reset_transition"].func is robust.reset_fall_retry
    assert {"base_z_roll", "unsafe", "foot_support_lost"} <= original.terminations.keys()
    assert cfg.curriculum_start_stage == 2
    assert not cfg.curriculum and not cfg.curriculum_prepare_pose
    assert cfg.standing_stage_ramp_steps == 0
    assert not cfg.require_sole_support_for_positive_rewards
    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        "joint_standing": 12.0, "base_height_progress": 1.0,
        "base_ascent_velocity": 8.0, "sole_ground_alignment": 1.0, "reset_failure": -10.0,
    }
    assert cfg.rewards["base_ascent_velocity"].params["target_speed"] == 0.15
    assert "base_disturbance" not in original.events
    term = cfg.events["base_disturbance"]
    assert term.func is RampedBaseImpulse
    assert term.params["force_range"] == (0, 0)
    assert term.params["torque_range"] == (0, 0)
    assert term.params["disturbed_fraction"] == 0
    assert load_runner_cls(ROBUST_TASK_ID) is RobustSnakeToBipedRunner
    agent = load_rl_cfg(ROBUST_TASK_ID)
    original_agent = load_rl_cfg(SnakeToBipedBeta.task_id)
    assert asdict(agent.actor) == asdict(original_agent.actor)
    assert asdict(agent.critic) == asdict(original_agent.critic)
    assert agent.algorithm.learning_rate == 1e-4
    assert agent.algorithm.entropy_coef == 0.001
    assert agent.max_iterations == 2000
    assert not agent.resume
    assert disturbance_scale(SimpleNamespace(cfg=cfg, common_step_counter=0)) == 1.0


@pytest.mark.parametrize("step,expected", [(0, 0.1), (1200, 0.1), (13200, 0.55), (25200, 1), (99999, 1)])
def test_disturbance_ramp_is_monotone_and_capped(step, expected):
    env = SimpleNamespace(cfg=SnakeToBipedRobust().build_env(), common_step_counter=step)
    env.cfg.disturbance_initial_scale = 0.1
    env.cfg.disturbance_warmup_steps = 1200
    env.cfg.disturbance_ramp_steps = 24000
    assert disturbance_scale(env) == pytest.approx(expected)


@pytest.mark.parametrize("field,value", [("disturbance_initial_scale", float("nan")), ("disturbance_initial_scale", 1.1), ("disturbance_ramp_steps", -1), ("disturbance_warmup_steps", -1)])
def test_invalid_disturbance_schedule_is_rejected(field, value):
    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        disturbance_scale(SimpleNamespace(cfg=cfg, common_step_counter=0))


def test_impulses_apply_both_wrench_parts_preserve_quiet_worlds_and_reset():
    torch.manual_seed(3)
    wrench = torch.zeros(16, 1, 6)

    def write(forces, torques, env_ids, body_ids):
        wrench[env_ids, :, :3] = forces
        wrench[env_ids, :, 3:] = torques

    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    cfg.disturbance_initial_scale = 1.0
    params = dict(cfg.events["base_disturbance"].params)
    # 显式测试保留的引擎；当前实验未施加推力。
    params.update(force_range=(-3.0, 3.0), torque_range=(-0.15, 0.15), disturbed_fraction=0.75)
    params["asset_cfg"] = SimpleNamespace(name="robot", body_ids=[0])
    body_quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(16, 1, 4)
    env = SimpleNamespace(cfg=cfg, num_envs=16, device="cpu", step_dt=0.02, common_step_counter=0, scene={"robot": SimpleNamespace(write_external_wrench_to_sim=write, data=SimpleNamespace(body_com_quat_w=body_quats))})
    off_params = dict(cfg.events["base_disturbance"].params)
    off_params["asset_cfg"] = params["asset_cfg"]
    off = RampedBaseImpulse(SimpleNamespace(params=off_params), env)
    off._interval_time_left[:] = 0
    off(env, None, **off_params)
    assert not off.disturbed_mask.any() and wrench.count_nonzero() == 0
    torch.manual_seed(3)
    impulse = RampedBaseImpulse(SimpleNamespace(params=params), env)
    impulse._interval_time_left[:] = 0
    impulse(env, None, **params)
    mask = impulse._disturbed_mask
    assert mask.any() and (~mask).any()
    assert wrench[mask, :, :3].abs().max() > 0
    assert wrench[mask, :, 3:].abs().max() > 0
    assert wrench[..., :3].abs().max() <= 3
    assert wrench[..., 3:].abs().max() <= 0.15
    assert wrench[~mask].count_nonzero() == 0
    active_before = impulse._active.clone()
    impulse.reset(torch.tensor([0, 1]))
    assert not impulse._active[:2].any()
    assert torch.equal(impulse._active[2:], active_before[2:])
    assert wrench[:2].count_nonzero() == 0
    impulse.reset()
    assert not impulse._active.any() and wrench.count_nonzero() == 0
    recovering = int(mask.nonzero(as_tuple=False)[0, 0])
    robust.fall_retry_state(env).recovering[recovering] = True
    impulse._interval_time_left[:] = 0
    impulse(env, None, **params)
    assert wrench[recovering].count_nonzero() == 0
    assert not impulse._active[recovering]


@pytest.mark.parametrize("resume,log_enabled,source_enabled,loads", [(False, True, True, 1), (True, True, True, 0), (False, False, True, 0), (False, True, False, 0)])
def test_pretrained_load_is_a_fresh_weight_transfer_and_resume_never_reinitializes(
    tmp_path, monkeypatch, resume, log_enabled, source_enabled, loads
):
    calls = []
    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    source = tmp_path / "source.pt"
    source.write_bytes(b"test checkpoint")
    cfg.pretrained_checkpoint = str(source) if source_enabled else None
    env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg, common_step_counter=0))

    def initialize(self, *args):
        self.current_learning_iteration = 0
        self.alg = SimpleNamespace(learning_rate=1e-4)

    def load(self, path, **kwargs):
        calls.append((path, kwargs))

    monkeypatch.setattr(SnakeToBipedHotRewardRunner, "__init__", initialize)
    monkeypatch.setattr(RobustSnakeToBipedRunner, "load", load)
    log_dir = tmp_path / "run" if log_enabled else None
    RobustSnakeToBipedRunner(env, {"resume": resume}, log_dir, "cpu")
    assert len(calls) == loads
    if loads:
        assert calls[0][1]["load_cfg"] == {"actor": True, "critic": True, "optimizer": False, "iteration": False}
        record = json.loads((log_dir / "pretrained_initialization.json").read_text())
        assert record["initial_iteration"] == record["initial_common_step_counter"] == 0
        assert record["learning_rate"] == 1e-4


def test_default_training_never_loads_a_prior_model(tmp_path, monkeypatch):
    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    assert cfg.pretrained_checkpoint is None
    assert not SnakeToBipedRobust().build_ppo().resume
    monkeypatch.setattr(SnakeToBipedHotRewardRunner, "__init__", lambda *args: None)

    def forbid_load(*args, **kwargs):
        raise AssertionError("From-scratch training must not read a prior checkpoint")

    monkeypatch.setattr(RobustSnakeToBipedRunner, "load", forbid_load)
    RobustSnakeToBipedRunner(SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg)), {}, str(tmp_path))
    assert not (tmp_path / "pretrained_initialization.json").exists()


@pytest.fixture
def fall_env(monkeypatch):
    env = SimpleNamespace(
        cfg=SnakeToBipedRobust().build_env(), num_envs=2, device="cpu",
        step_dt=0.02, common_step_counter=0,
        heights=torch.tensor([0.0525, 0.0525]), support=torch.tensor([False, False]),
        snake_ready=torch.tensor([False, False]), episode_length_buf=torch.full((2,), 100),
    )
    monkeypatch.setattr(robust, "base_height", lambda env: env.heights)
    monkeypatch.setattr(robust, "snake_recovery_ready", lambda env: env.snake_ready)
    monkeypatch.setattr(robust, "randomize_initial_snake_roll", lambda env, env_ids: None)
    env.reward_manager = SimpleNamespace(get_term_cfg=env.cfg.rewards.__getitem__)
    return env


def test_initial_snake_does_not_arm_but_rising_needs_no_sole_gate(fall_env):
    env = fall_env
    for step in range(100):
        env.common_step_counter = step
        assert not robust.fallen_after_standing(env).any()
    env.heights[:] = 0.22
    env.common_step_counter += 1
    assert not robust.fallen_after_standing(env).any()
    assert robust.fall_retry_state(env).armed.all()


def test_confirmed_fall_retries_only_after_standing_and_clears_on_reset(fall_env):
    env = fall_env
    env.heights[:] = 0.22
    env.support[:] = True
    assert not robust.fallen_after_standing(env).any()
    assert robust.fall_retry_state(env).armed.all()
    env.heights[:] = 0.05
    env.support[:] = False
    for step in range(1, 5):
        env.common_step_counter = step
        assert not robust.fallen_after_standing(env).any()
        assert not robust.fallen_after_standing(env).any()
    env.common_step_counter = 5
    assert robust.fallen_after_standing(env).all()
    robust.reset_fall_retry(env, torch.tensor([0]))
    state = robust.fall_retry_state(env)
    assert state.armed.tolist() == [False, True]
    assert state.low_time[0] == 0
    assert robust.fallen_after_standing(env).tolist() == [False, True]
    robust.reset_fall_retry(env, None)
    env.common_step_counter += 1
    assert not robust.fallen_after_standing(env).any()


def test_brief_height_drop_is_cleared_if_robot_recovers(fall_env):
    env = fall_env
    env.heights[:] = 0.22
    env.support[:] = True
    robust.fallen_after_standing(env)
    env.heights[:] = 0.05
    for step in range(1, 4):
        env.common_step_counter = step
        assert not robust.fallen_after_standing(env).any()
    env.heights[:] = 0.15
    env.common_step_counter += 1
    assert not robust.fallen_after_standing(env).any()
    assert robust.fall_retry_state(env).low_time.eq(0).all()


def test_fall_enters_motion_recovery_without_reset_and_returns_to_policy(fall_env, monkeypatch):
    env = fall_env
    def forbid_roll_reset(*args):
        raise AssertionError("Continuous recovery must not randomize/overwrite root state")
    monkeypatch.setattr(robust, "randomize_initial_snake_roll", forbid_roll_reset)
    env.heights[:] = 0.22
    env.support[:] = True
    robust.update_recovery_phase(env)
    env.heights[:] = 0.05
    env.support[:] = False
    for step in range(1, 6):
        env.common_step_counter = step
        robust.update_recovery_phase(env)
    state = robust.fall_retry_state(env)
    assert state.recovering.all()
    assert env.episode_length_buf.eq(100).all()
    env.snake_ready[0] = True
    for step in range(6, 31):
        env.common_step_counter = step
        robust.update_recovery_phase(env)
        robust.update_recovery_phase(env)
    assert state.recovering.tolist() == [False, True]
    assert env.episode_length_buf.tolist() == [0, 100]
    assert not state.armed.any()


def test_timeout_is_deferred_during_motion_recovery(fall_env, monkeypatch):
    env = fall_env
    monkeypatch.setattr(robust.common_mdp, "time_out", lambda env: env.episode_length_buf >= 100)
    robust.fall_retry_state(env).recovering[0] = True
    assert robust.recovery_time_out(env).tolist() == [False, True]


def test_recovery_action_moves_targets_slowly_without_writing_joint_state(monkeypatch):
    def initialize(self, cfg, env):
        self._env = env
        self._target_names = [f"joint{i}" for i in range(1, 7)]
        self._target_ids = list(range(6))
        self._raw_actions = torch.zeros(2, 6)
        self._processed_actions = torch.zeros(2, 6)
        self._entity = SimpleNamespace(data=env.data)

    monkeypatch.setattr(JointPositionAction, "__init__", initialize)
    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    data = SimpleNamespace(default_joint_pos=expand_mirror(torch.full((2, 3), 0.6)), joint_pos=torch.full((2, 6), 0.4), joint_pos_limits=torch.tensor([-6.28, 6.28]).expand(2, 6, 2).clone())
    env = SimpleNamespace(cfg=cfg, data=data, num_envs=2, device="cpu", step_dt=0.02)
    action = robust.RecoveryJointTargetRateActionCfg(**asdict(cfg.actions["joint_pos"])).build(env)
    robust.fall_retry_state(env).recovering[0] = True
    previous = action.commanded_target.clone()
    action.process_actions(torch.ones(2, 3))
    torch.testing.assert_close(action.commanded_target[0], expand_mirror(previous[0, :3] - 0.01))
    torch.testing.assert_close(action.commanded_target[1], expand_mirror(previous[1, :3] + 2 * math.pi * 0.02))
    for _ in range(100):
        previous = action.commanded_target.clone()
        action.process_actions(torch.ones(2, 3))
        assert (action.commanded_target[0] - previous[0]).abs().amax() <= 0.010001
    assert action.commanded_target[0].eq(0).all()
    assert data.joint_pos.eq(0.4).all()


def test_base_upward_speed_and_height_reward_do_not_require_any_sole_data():
    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    positions = torch.zeros(3, 1, 3)
    positions[:, 0, 2] = torch.tensor([0.0525, 0.15, 0.264])
    velocities = torch.zeros(3, 1, 3)
    velocities[:, 0, 2] = torch.tensor([0, 0.075, 0.15])

    class Scene(dict):
        pass

    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=positions, body_link_lin_vel_w=velocities, joint_pos=torch.tensor(cfg.standing_joint_pos).expand(3, 6) + 1), find_bodies=lambda name: ([0], [name])))
    scene.env_origins = torch.zeros(3, 3)
    env = SimpleNamespace(cfg=cfg, scene=scene, device="cpu", num_envs=3, common_step_counter=0)
    term = cfg.rewards["base_ascent_velocity"]
    torch.testing.assert_close(term.weight * term.func(env, **term.params), torch.tensor([0.0, 6.0, 0.0]))
    height = reverse.base_height_progress_reward(env)
    assert height[1] > 0
    torch.testing.assert_close(height[[0, 2]], torch.tensor([0.0, 1.0]))


def fixed_reward_env():
    cfg = SnakeToBipedRobust().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    return SimpleNamespace(cfg=cfg, common_step_counter=0, reward_manager=SimpleNamespace(active_terms=list(cfg.rewards), get_term_cfg=cfg.rewards.__getitem__))


def test_fixed_weights_never_ramp_and_hot_edit_uses_compact_rewards():
    env = fixed_reward_env()
    for step in (0, 1, 2000, 100000):
        env.common_step_counter = step
        robust.apply_fixed_standing_weights(env)
        assert env.cfg.rewards["joint_standing"].weight == 12
        assert env.cfg.rewards["base_ascent_velocity"].weight == 8
        assert env.cfg.rewards["base_height_progress"].weight == 1
        assert disturbance_scale(env) == 1
    HotRewardController(env)._transaction({"config": {"standing_stage_velocity_weight": 12.0}})
    assert env.cfg.rewards["base_ascent_velocity"].weight == 12
    assert env.cfg.rewards["sole_ground_alignment"].weight == 1
    HotRewardController(env)._transaction({"rewards": {"sole_ground_alignment": 3.0}})
    assert env.cfg.rewards["sole_ground_alignment"].weight == 3
    robust.apply_fixed_standing_weights(env)
    assert env.cfg.rewards["sole_ground_alignment"].weight == 3
    with pytest.raises(ValueError, match="Invalid"):
        HotRewardController(env)._transaction({"rewards": {"sole_ground_alignment": -1.0}})
    with pytest.raises(ValueError, match="Unknown"):
        HotRewardController(env)._transaction({"config": {"alignment_stage_sole_weight": 2.0}})


def test_resuming_old_checkpoint_keeps_current_fixed_reward_recipe(monkeypatch):
    env = fixed_reward_env()

    def legacy_load(self, *args):
        env.cfg.standing_stage_joint_weight = 1
        env.cfg.standing_stage_height_weight = 0
        env.cfg.standing_stage_velocity_weight = 0
        env.cfg.rewards["reset_failure"].weight = -5
        env.cfg.rewards["sole_ground_alignment"].weight = 0
        return {"loaded": True}

    monkeypatch.setattr(SnakeToBipedHotRewardRunner, "load", legacy_load)
    runner = object.__new__(RobustSnakeToBipedRunner)
    runner.env = SimpleNamespace(unwrapped=env)
    assert runner.load("old.pt") == {"loaded": True}
    assert env.cfg.rewards["joint_standing"].weight == 12
    assert env.cfg.rewards["base_ascent_velocity"].weight == 8
    assert env.cfg.rewards["base_height_progress"].weight == 1
    assert env.cfg.rewards["reset_failure"].weight == -10
    assert env.cfg.rewards["sole_ground_alignment"].weight == 1
    assert len(env.cfg.rewards) == 5


def test_six_action_resume_becomes_weight_transfer_without_incompatible_optimizer(tmp_path, monkeypatch):
    source = tmp_path / "six_action.pt"
    torch.save({"actor_state_dict": {"mlp.4.weight": torch.ones(12, 128), "mlp.4.bias": torch.arange(12.0)}}, source)
    calls = []

    def load_converted(self, path, load_cfg, strict, map_location):
        converted = torch.load(path, weights_only=False)
        assert converted["actor_state_dict"]["mlp.4.weight"].shape == (6, 128)
        calls.append(load_cfg)
        return {}

    monkeypatch.setattr(SnakeToBipedHotRewardRunner, "load", load_converted)
    runner = object.__new__(RobustSnakeToBipedRunner)
    runner.env = SimpleNamespace(unwrapped=fixed_reward_env())
    runner.device = "cpu"
    runner.load(str(source))
    assert calls == [{"actor": True, "critic": True, "optimizer": False, "iteration": False}]
    assert runner._mirror_migration["new_action_dim"] == 3
    assert torch.load(source, weights_only=False)["actor_state_dict"]["mlp.4.weight"].shape[0] == 12


def test_compact_goal_counter_matches_original_without_support_collection():
    from test_snake_to_biped import _standing_env

    from zbot_rl_mjlab.tasks.transformation import mdp

    old, compact = _standing_env(), _standing_env()
    compact.cfg = SnakeToBipedRobust().build_env()
    assert isinstance(compact.cfg, RobustSnakeToBipedEnvCfg)
    for step in range(104):
        old.common_step_counter = compact.common_step_counter = step
        if step == 102:
            old.scene["robot"].data.joint_pos[0, 0] += 1
            compact.scene["robot"].data.joint_pos[0, 0] += 1
        reverse.update_state(old)
        robust.update_fixed_standing_state(compact)
        robust.update_fixed_standing_state(compact)
        torch.testing.assert_close(mdp.state(old).hold_time, mdp.state(compact).hold_time)
        torch.testing.assert_close(robust.standing_success_metric(compact), reverse.success(old))
    assert not hasattr(compact, "_support_protection_state")


def test_robust_sole_reward_is_steepest_near_flat_and_independent_of_height():
    from test_snake_to_biped import _standing_env

    angles = torch.tensor([0., 1., 5., 10., 20., 45., 90., 180.], dtype=torch.float64)
    env = _standing_env(num_envs=len(angles))
    env.cfg = SnakeToBipedRobust().build_env()
    assert isinstance(env.cfg, RobustSnakeToBipedEnvCfg)
    halves = torch.deg2rad(angles[:, None] + torch.tensor([0., 180.])) / 2
    quaternions = torch.zeros(len(angles), 2, 4, dtype=torch.float64)
    quaternions[..., 0] = halves.cos()
    quaternions[..., 1] = halves.sin()
    env.scene["robot"].data.site_quat_w = quaternions
    term = env.cfg.rewards["sole_ground_alignment"]
    score = term.func(env, **term.params)
    expected = 0.5 * (torch.deg2rad(angles).cos() - 1) + torch.exp(-angles / 10)
    torch.testing.assert_close(score, expected)
    assert (score[:-1] > score[1:]).all()
    assert score[0].item() == pytest.approx(1.)
    assert score[-1].item() == pytest.approx(-1., abs=1e-7)
    assert ((score >= -1) & (score <= 1)).all()
    assert ((term.weight * score >= -1) & (term.weight * score <= 1)).all()
    assert score[0] - score[2] > score[2] - score[3]
    legacy = reverse.sole_ground_alignment_reward(env)
    assert score[0] - score[1] > 4 * (legacy[0] - legacy[1])
    env.scene["robot"].data.site_pos_w[..., 2] = 0.5
    torch.testing.assert_close(term.func(env, **term.params), score)
    # 单只脚底倾斜的影响仍应大于仅考虑双脚平均值时的影响。
    env.scene["robot"].data.site_quat_w[:, 0] = quaternions[0, 0].clone()
    mixed = term.func(env, **term.params)
    torch.testing.assert_close(mixed, 0.25 + 0.75 * expected)


def test_signed_sole_reward_assigns_zero_to_invalid_orientations():
    from test_snake_to_biped import _standing_env

    env = _standing_env()
    env.cfg = SnakeToBipedRobust().build_env()
    assert isinstance(env.cfg, RobustSnakeToBipedEnvCfg)
    env.scene["robot"].data.site_quat_w[0] = float("nan")
    env.scene["robot"].data.site_quat_w[1] = float("inf")
    term = env.cfg.rewards["sole_ground_alignment"]
    torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2))


@pytest.mark.parametrize("velocity", [0.075, -0.01])
def test_velocity_reward_fades_for_every_joint_near_pose_goal(velocity):
    from test_snake_to_biped import _standing_env

    env = _standing_env(num_envs=5)
    env.cfg = SnakeToBipedRobust().build_env()
    assert isinstance(env.cfg, RobustSnakeToBipedEnvCfg)
    data = env.scene["robot"].data
    data.body_link_pos_w[:, 6, 2] = 0.15
    # 只有一个关节不同：最大误差必须防止其余五个关节掩盖该误差。
    data.joint_pos[:, 0] += torch.tensor([0., 0.2, 0.35, 0.5, 0.8])
    data.body_link_lin_vel_w[:, 6, 2] = velocity
    term = env.cfg.rewards["base_ascent_velocity"]
    score = term.func(env, **term.params)
    ungated = reverse.base_ascent_velocity_reward(env, target_speed=0.15)
    torch.testing.assert_close(score, ungated * torch.tensor([0., 0., 0.15625, 0.5, 1.]))
    assert torch.isfinite(score).all()
    # 此项独立于脚底支撑，不会削弱姿态反馈。
    assert env.cfg.rewards["joint_standing"].weight == 12
    pose = reverse.joint_standing_reward(env)
    assert (pose[1:] < pose[:-1]).all()


@pytest.mark.parametrize("limits", [(-0.1, 0.8), (0.8, 0.2), (0.2, 0.2), (0.2, float("inf"))])
def test_velocity_pose_gate_rejects_invalid_ranges(limits):
    from test_snake_to_biped import _standing_env

    with pytest.raises(ValueError, match="pose_error_range"):
        reverse.base_ascent_velocity_reward(_standing_env(), pose_error_range=limits)


def test_ascent_reward_stays_off_after_first_rise_until_partial_reset(monkeypatch):
    from test_snake_to_biped import _standing_env

    env = _standing_env()
    env.cfg = SnakeToBipedRobust().build_env()
    assert isinstance(env.cfg, RobustSnakeToBipedEnvCfg)
    env.reward_manager = SimpleNamespace(get_term_cfg=env.cfg.rewards.__getitem__)
    data = env.scene["robot"].data
    data.joint_pos += 1.0
    data.body_link_pos_w[:, 6, 2] = 0.15
    data.body_link_lin_vel_w[:, 6, 2] = 0.075
    term = env.cfg.rewards["base_ascent_velocity"]
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0.75, 0.75]))
    data.body_link_pos_w[0, 6, 2] = env.cfg.ascent_reward_stop_height
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0., 0.75]))
    data.body_link_pos_w[:, 6, 2] = 0.15
    data.joint_pos += 1.0  # 下降或破坏姿态均不得重新启用该项。
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0., 0.75]))
    data.body_link_pos_w[1, 6, 2] = 0.24
    term.func(env, **term.params)
    data.body_link_pos_w[:, 6, 2] = 0.15
    data.body_link_lin_vel_w[:, 6, 2] = -0.01
    torch.testing.assert_close(term.func(env, **term.params), torch.zeros(2))
    monkeypatch.setattr(robust, "randomize_initial_snake_roll", lambda env, ids: None)
    robust.reset_fall_retry(env, torch.tensor([0]))
    data.body_link_lin_vel_w[:, 6, 2] = 0.075
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0.75, 0.]))
