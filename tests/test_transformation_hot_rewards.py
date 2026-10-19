"""仅使用 CPU 检查奖励热更新事务、更新边界与检查点。"""

import json
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.modules import GaussianDistribution

from zbot_rl_mjlab.rl import (
    CHECKPOINT_KEY,
    CONFIG_SIGNS,
    REWARD_SIGNS,
    HotRewardController,
    SnakeToBipedHotRewardRunner,
)


def fake_env(path):
    config: dict[str, object] = {
        name: sign * 0.1 for name, sign in CONFIG_SIGNS.items()
    }
    config.update(
        reward_hot_reload_path=str(path),
        sole_alignment_stage_threshold=1.0,
        standing_stage_ramp_steps=100,
        curriculum_prepare_pose=False,
    )
    names = set(REWARD_SIGNS) | {
        "base_height_progress",
        "base_ascent_velocity",
        "joint_standing",
        "base_support_distance",
        "base_gravity_alignment",
        "survival",
        "initial_joint_pose",
        "end_joint_flip",
        "axial_roll_rate",
        "both_feet_contact",
    }
    terms = {name: SimpleNamespace(weight=-1.0) for name in names}
    terms["sole_ground_alignment"].weight = 2.0
    env = SimpleNamespace(
        cfg=SimpleNamespace(**config),
        common_step_counter=50,
        _standing_curriculum_stage=1,
        reward_manager=SimpleNamespace(
            active_terms=list(terms), get_term_cfg=terms.__getitem__
        ),
    )
    return env


def write(path, **sections):
    path.write_text(json.dumps(sections))


def test_applies_phase_and_static_weights_and_logs_once(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env, tmp_path)
    write(
        path,
        config={"first_stage_joint_weight": 0.8, "feet_contact_weight": 1.2},
        rewards={"sole_ground_alignment": 3.0, "sole_height": -2.0},
    )
    assert controller.poll(7)
    snapshot = controller.snapshot()
    assert snapshot["effective_weights"]["joint_standing"] == 0.8
    assert snapshot["effective_weights"]["both_feet_contact"] == 1.2
    assert snapshot["effective_weights"]["sole_height"] == -2.0
    assert not controller.poll(8)
    assert controller.log_path is not None
    records = [
        json.loads(line) for line in controller.log_path.read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["iteration"] == 7
    assert records[0]["status"] == "applied"


def test_stage_two_respects_current_ramp_fraction(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    env._standing_curriculum_stage = 2
    env._standing_stage_start_step = 0
    controller = HotRewardController(env)
    write(
        path,
        config={
            "first_stage_joint_weight": 0.2,
            "standing_stage_joint_weight": 2.2,
            "standing_stage_height_weight": 4.0,
        },
    )
    assert controller.poll(1)
    assert controller.snapshot()["effective_weights"][
        "joint_standing"
    ] == pytest.approx(1.2)
    assert controller.snapshot()["effective_weights"]["base_height_progress"] == 2.0


@pytest.mark.parametrize(
    "payload",
    [
        {"config": {"standing_joint_pos": 0.5}},
        {"rewards": {"unknown": 2}},
        {"rewards": {"sole_ground_alignment": 1.0}},
        {"config": {"end_joint_flip_weight": 1.0}},
        {"config": {"standing_stage_joint_weight": -1.0}},
        {"config": {"feet_contact_weight": True}},
        {"config": {"feet_contact_weight": float("nan")}},
        {"config": {"feet_contact_weight": float("inf")}},
        {"config": {"feet_contact_weight": 10**1000}},
        {"config": []},
        {"unexpected": {}},
        [],
    ],
)
def test_rejects_whole_invalid_file_atomically_and_logs_once(tmp_path, payload):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env, tmp_path)
    previous = controller.snapshot()
    path.write_text(json.dumps(payload))
    assert not controller.poll(1)
    assert not controller.poll(2)
    assert controller.snapshot() == previous
    assert controller.log_path is not None
    assert len(controller.log_path.read_text().splitlines()) == 1


def test_partial_write_can_be_completed_and_missing_disabled_are_safe(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env, tmp_path)
    assert not controller.poll(0)
    previous = controller.snapshot()
    path.write_text('{"config": {')
    assert not controller.poll(1)
    assert controller.snapshot() == previous
    write(path, config={"standing_stage_joint_weight": 2.0})
    assert controller.poll(2)
    env.cfg.reward_hot_reload_path = None
    write(path, config={"standing_stage_joint_weight": 3.0})
    assert not controller.poll(3)
    assert env.cfg.standing_stage_joint_weight == 2.0


def test_failed_apply_rolls_back_config_and_all_live_weights(tmp_path, monkeypatch):
    from zbot_rl_mjlab.tasks.transformation import snake_to_biped

    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env)
    previous = controller.snapshot()

    def fail(env):
        env.reward_manager.get_term_cfg("joint_standing").weight = 90
        raise ValueError("test interrupted apply")

    monkeypatch.setattr(snake_to_biped, "apply_standing_stage_weights", fail)
    write(path, config={"standing_stage_joint_weight": 2.0})
    assert not controller.poll(1)
    assert controller.snapshot() == previous


def test_restore_stage_progress_then_latest_file_wins(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    env._standing_curriculum_stage = 2
    env._standing_stage_start_step = 20
    controller = HotRewardController(env)
    saved = controller.snapshot()
    env.common_step_counter = 200
    env._standing_curriculum_stage = 1
    write(path, config={"standing_stage_joint_weight": 3.0})
    controller.restore(saved, 9)
    assert env._standing_curriculum_stage == 2
    assert env._standing_stage_start_step == 170
    assert env.cfg.standing_stage_joint_weight == 3.0
    assert controller.snapshot()["effective_weights"][
        "joint_standing"
    ] == pytest.approx(0.97)


@pytest.fixture
def make_runner(tmp_path, monkeypatch):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    updates = []
    result = {"loss": 1.0}

    def update():
        if getattr(env, "fail_update", False):
            raise RuntimeError("failed optimizer update")
        updates.append(env.cfg.standing_stage_joint_weight)
        return result

    def initialize(self, wrapped_env, train_cfg, log_dir, device):
        self.env = wrapped_env
        self.current_learning_iteration = 0
        distribution = GaussianDistribution(6, init_std=1.1, std_range=(0.1, 1.0))
        policy = SimpleNamespace(distribution=distribution)
        self.alg = SimpleNamespace(
            update=update,
            get_policy=lambda: policy,
            optimizer=torch.optim.Adam(distribution.parameters(), lr=0.1),
        )

    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", initialize)
    write(path, config={"standing_stage_joint_weight": 2.0})
    runner = SnakeToBipedHotRewardRunner(
        cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=env)), {}, str(tmp_path)
    )
    return runner, env, path, updates, result


def test_runner_only_polls_after_successful_update_and_preserves_result(make_runner):
    runner, env, path, updates, result = make_runner
    assert env.cfg.standing_stage_joint_weight == 2.0
    write(path, config={"standing_stage_joint_weight": 4.0})
    assert runner.alg.update() is result
    assert updates == [2.0]
    assert env.cfg.standing_stage_joint_weight == 4.0


def test_runner_save_and_full_load_restore_but_actor_only_does_not(
    make_runner, monkeypatch
):
    runner, env, path, _, _ = make_runner
    captured = {}

    def save(self, path, infos):
        captured.update(infos)

    def load(self, path, load_cfg, strict, map_location):
        self.env.unwrapped.common_step_counter = 500
        self.current_learning_iteration = 20
        return captured

    monkeypatch.setattr(MjlabOnPolicyRunner, "save", save)
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    env._standing_curriculum_stage = 2
    env._standing_stage_start_step = 20
    runner.save("ignored", {"extra": True})
    assert CHECKPOINT_KEY in captured and captured["extra"]
    path.unlink()
    env._standing_curriculum_stage = 1
    env.cfg.standing_stage_joint_weight = 9.0
    runner.load("ignored", {"actor": True})
    assert env.common_step_counter == 50
    assert env._standing_curriculum_stage == 1
    assert env.cfg.standing_stage_joint_weight == 9.0
    runner.load("ignored")
    assert env.common_step_counter == 500
    assert env._standing_curriculum_stage == 2
    assert env._standing_stage_start_step == 470
    assert env.cfg.standing_stage_joint_weight == 2.0


def test_failed_optimizer_update_does_not_poll(make_runner):
    runner, env, path, _, _ = make_runner
    write(path, config={"standing_stage_joint_weight": 4.0})
    env.fail_update = True
    with pytest.raises(RuntimeError, match="failed optimizer"):
        runner.alg.update()
    assert env.cfg.standing_stage_joint_weight == 2.0
    assert runner._hot_update_iteration == 0


def test_runner_projects_std_at_initialization_and_after_adam_steps(make_runner):
    runner, _, _, _, _ = make_runner
    distribution = runner.alg.get_policy().distribution
    value = distribution.std_param
    assert value.lt(1.0).all() and value.gt(0.1).all()
    assert value.eq(torch.nextafter(torch.tensor(1.0), torch.tensor(0.1))).all()
    optimizer = runner.alg.optimizer
    for iteration in range(3):
        optimizer.zero_grad()
        (-value.sum()).backward()
        optimizer.step()
        assert value.lt(1.0).all() and value.gt(0.1).all()
        assert optimizer.state[value]["step"] == iteration + 1


@pytest.mark.parametrize(
    "load_cfg",
    [None, {"actor": True, "critic": True, "optimizer": True, "iteration": True}],
)
def test_full_load_projects_dead_std_bounds(make_runner, monkeypatch, load_cfg):
    runner, _, _, _, _ = make_runner
    distribution = runner.alg.get_policy().distribution
    optimizer = runner.alg.optimizer
    value = distribution.std_param

    def load(self, path, cfg, strict, map_location):
        with torch.no_grad():
            value[:] = torch.tensor([0.0, 0.1, 0.5, 1.0, 1.1, 2.0])
        return {}

    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    runner.load("ignored", load_cfg=load_cfg)
    assert value.gt(0.1).all() and value.lt(1.0).all()
    assert value[2] == 0.5
    assert runner.alg.optimizer is optimizer
    distribution.update(torch.zeros(1, 6))
    gradient = torch.autograd.grad(distribution.entropy.sum(), value)[0]
    assert gradient.ne(0).all()


def test_actor_only_load_preserves_checkpoint_std_exactly(make_runner, monkeypatch):
    runner, _, _, _, _ = make_runner
    value = runner.alg.get_policy().distribution.std_param
    loaded = torch.tensor([0.0, 0.1, 0.5, 1.0, 1.1, 2.0])

    def load(self, path, cfg, strict, map_location):
        with torch.no_grad():
            value.copy_(loaded)
        return {}

    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    runner.load("ignored", load_cfg={"actor": True})
    assert torch.equal(value, loaded)


def test_only_actual_alignment_changes_restart_promotion_wait(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env)
    write(path, config={"standing_stage_joint_weight": 2.0})
    assert controller.poll(1)
    assert not hasattr(env, "_standing_alignment_hot_changed_step")
    write(path, rewards={"sole_ground_alignment": 2.0})
    assert controller.poll(2)
    assert not hasattr(env, "_standing_alignment_hot_changed_step")
    write(path, rewards={"sole_ground_alignment": 3.0})
    assert controller.poll(3)
    assert env._standing_alignment_hot_changed_step == 50
    env.common_step_counter = 90
    write(
        path,
        rewards={"sole_ground_alignment": 3.0},
        config={"feet_contact_weight": 2.0},
    )
    assert controller.poll(4)
    assert env._standing_alignment_hot_changed_step == 50
    write(path, rewards={"sole_ground_alignment": 2.5})
    assert controller.poll(5)
    assert env._standing_alignment_hot_changed_step == 90


@pytest.mark.parametrize("previous_mark", [None, 20])
def test_failed_hot_transaction_rolls_back_alignment_wait_marker(
    tmp_path, monkeypatch, previous_mark
):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    if previous_mark is not None:
        env._standing_alignment_hot_changed_step = previous_mark
    controller = HotRewardController(env)
    previous = controller.snapshot()

    def fail(payload):
        env._standing_alignment_hot_changed_step = 1000
        env.reward_manager.get_term_cfg("sole_ground_alignment").weight = 3.0
        raise ValueError("interrupted apply")

    monkeypatch.setattr(controller, "_apply", fail)
    write(path, rewards={"sole_ground_alignment": 3.0})
    assert not controller.poll(1)
    assert controller.snapshot() == previous
    assert getattr(env, "_standing_alignment_hot_changed_step", None) == previous_mark


def test_full_gravity_weight_is_hot_editable_and_rolls_back_with_invalid_transaction(
    tmp_path,
):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env)
    write(path, config={"base_gravity_alignment_weight": -0.75})
    assert controller.poll(1)
    assert env.cfg.base_gravity_alignment_weight == -0.75
    assert controller.snapshot()["effective_weights"]["base_gravity_alignment"] == -0.75
    before = controller.snapshot()
    write(
        path,
        config={"base_gravity_alignment_weight": -1.0},
        rewards={"sole_height": 1.0},
    )
    assert not controller.poll(2)
    assert controller.snapshot() == before
    write(path, config={"base_gravity_alignment_weight": 1.0})
    assert not controller.poll(3)
    assert controller.snapshot() == before


def test_old_snapshot_without_full_gravity_field_keeps_default_zero(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    env.cfg.reward_hot_reload_path = None
    env.cfg.base_gravity_alignment_weight = 0.0
    env.reward_manager.get_term_cfg("base_gravity_alignment").weight = 0.0
    controller = HotRewardController(env)
    old = controller.snapshot()
    old["config"].pop("base_gravity_alignment_weight")
    old["effective_weights"].pop("base_gravity_alignment")
    controller.restore(old, 10)
    assert env.cfg.base_gravity_alignment_weight == 0.0
    assert env.reward_manager.get_term_cfg("base_gravity_alignment").weight == 0.0


def test_survival_weight_hot_update_and_invalid_sign_transaction(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    controller = HotRewardController(env)
    write(path, config={"survival_weight": 4.0})
    assert controller.poll(1)
    assert env.cfg.survival_weight == 4.0
    assert controller.snapshot()["effective_weights"]["survival"] == 4.0
    previous = controller.snapshot()
    write(path, config={"survival_weight": -1.0, "standing_stage_joint_weight": 5.0})
    assert not controller.poll(2)
    assert controller.snapshot() == previous
    write(path, config={"survival_weight": 0.0})
    assert controller.poll(3)
    assert env.reward_manager.get_term_cfg("survival").weight == 0.0


def test_old_snapshot_without_survival_keeps_startup_default_zero(tmp_path):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    env.cfg.reward_hot_reload_path = None
    env.cfg.survival_weight = 0.0
    env.reward_manager.get_term_cfg("survival").weight = 0.0
    controller = HotRewardController(env)
    old = controller.snapshot()
    old["config"].pop("survival_weight")
    old["effective_weights"].pop("survival")
    controller.restore(old, 10)
    assert env.cfg.survival_weight == 0.0
    assert env.reward_manager.get_term_cfg("survival").weight == 0.0


def test_prepared_pose_weight_hot_updates_without_allowing_boolean_mode_changes(
    tmp_path,
):
    path = tmp_path / "weights.json"
    env = fake_env(path)
    env.cfg.curriculum_prepare_pose = True
    env._standing_pose_prepared = True
    controller = HotRewardController(env)
    write(path, config={"first_stage_prepared_joint_weight": 3.0})
    assert controller.poll(1)
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 3.0
    before = controller.snapshot()
    write(
        path,
        config={
            "first_stage_prepared_joint_weight": 4.0,
            "curriculum_prepare_pose": False,
        },
    )
    assert not controller.poll(2)
    assert controller.snapshot() == before


def test_new_metadata_restores_preparation_latch_and_weight_before_next_rollout(
    tmp_path,
):
    env = fake_env(tmp_path / "weights.json")
    env.cfg.reward_hot_reload_path = None
    env.cfg.curriculum_prepare_pose = True
    env.cfg.first_stage_joint_weight = 0.6
    env.cfg.first_stage_prepared_joint_weight = 3.0
    env._standing_pose_prepared = True
    controller = HotRewardController(env)
    saved = controller.snapshot()
    env._standing_pose_prepared = False
    env.cfg.first_stage_prepared_joint_weight = 8.0
    controller.restore(saved, 10)
    assert env._standing_pose_prepared
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 3.0
    assert controller.snapshot()["pose_prepared"]


@pytest.mark.parametrize("stage,expected", [(1, 0.6), (2, 4.5)])
def test_old_metadata_without_preparation_fields_retains_startup_recipe(
    tmp_path, stage, expected
):
    env = fake_env(tmp_path / "weights.json")
    env.cfg.reward_hot_reload_path = None
    env.cfg.curriculum_prepare_pose = True
    env.cfg.first_stage_joint_weight = 0.6
    env.cfg.first_stage_prepared_joint_weight = 3.0
    env.cfg.standing_stage_joint_weight = 6.0
    env._standing_curriculum_stage = stage
    env._standing_stage_start_step = 0
    controller = HotRewardController(env)
    saved = controller.snapshot()
    saved.pop("pose_prepared")
    saved["config"].pop("first_stage_prepared_joint_weight")
    controller.restore(saved, 10)
    assert env.cfg.first_stage_prepared_joint_weight == 3.0
    assert env.reward_manager.get_term_cfg("joint_standing").weight == expected
    assert controller.snapshot()["pose_prepared"] == (stage == 2)
