"""真实 CPU 策略检查有界 Beta 探索、检查点语义与新版教师回放。"""

import math
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import BetaDistribution
from tensordict import TensorDict

from zbot_rl_mjlab.experiments.snake import (
    SnakeOmniTeacher,
    SnakeTeacherRunner,
)


def runner_cfg():
    cfg = asdict(SnakeOmniTeacher().build_ppo())
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [16]
    cfg["upload_model"] = False
    return cfg


def make_runner(cfg=None):
    refreshes = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(motion_penalty_min_scale=0., motion_penalty_speed_gain=5., forward_speed_cap=.2, head_yaw_limit=math.pi / 3, head_yaw_integral_time_scale=1., body_yaw_limit=math.pi / 3, body_yaw_integral_time_scale=1., tail_site="foot_1_sole", actions={"joint_pos": SimpleNamespace(target_limit=math.pi / 3, max_target_speed=2 * math.pi, use_default_offset=False)}, motion_frame="body_gravity", head_gyro_frame="head", head_body="foot_0", motion_site="foot_0_sole", observation_reference="world_x"),
        common_step_counter=0,
        observation_manager=SimpleNamespace(compute=lambda **kwargs: refreshes.append(kwargs)),
    )
    obs = TensorDict({"actor": torch.randn(8, 51), "critic": torch.randn(8, 66)}, [8])
    wrapped = SimpleNamespace(
        unwrapped=env, cfg={}, num_envs=8, num_actions=6, get_observations=lambda: obs,
    )
    return SnakeTeacherRunner(cast(VecEnv, wrapped), runner_cfg() if cfg is None else cfg), refreshes


def distribution(runner) -> Any:
    return runner.alg.get_policy().distribution


def state_copy(runner):
    return {key: value.clone() for key, value in runner.alg.get_policy().state_dict().items()}


def update_once(runner):
    actor = runner.alg.get_policy()
    obs = runner.env.get_observations()
    samples = actor(obs, stochastic_output=True)
    old_params = tuple(parameter.detach().clone() for parameter in distribution(runner).params)
    before = state_copy(runner)
    old_log_prob = distribution(runner).log_prob(samples.detach()).detach()
    advantages = torch.linspace(-1., 1., samples.shape[0])
    ratio = (distribution(runner).log_prob(samples.detach()) - old_log_prob).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages)
    loss = -surrogate.mean() - .002 * distribution(runner).entropy.mean()
    runner.alg.optimizer.zero_grad()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters())
    runner.alg.optimizer.step()
    assert any(not torch.equal(before[key], value) for key, value in actor.state_dict().items())
    actor(obs, stochastic_output=True)
    assert torch.isfinite(distribution(runner).kl_divergence(old_params, distribution(runner).params)).all()
    return samples


def test_beta_recipe_uses_native_distribution_and_unchanged_entropy():
    cfg = SnakeOmniTeacher().build_ppo()
    assert cfg.actor.distribution_cfg == {"class_name": "BetaDistribution", "action_range": (-1., 1.)}
    assert cfg.algorithm.entropy_coef == .002


def test_native_beta_samples_means_and_probability_math_remain_finite_and_bounded():
    runner, _ = make_runner()
    actor = runner.alg.get_policy()
    assert isinstance(distribution(runner), BetaDistribution)
    obs = runner.env.get_observations()
    for _ in range(20):
        for stochastic in (False, True):
            actions = actor(obs, stochastic_output=stochastic)
            assert actions.shape == (8, 6)
            assert torch.isfinite(actions).all() and actions.abs().le(1).all()
        assert torch.isfinite(distribution(runner).log_prob(actor(obs, stochastic_output=True))).all()
        assert torch.isfinite(distribution(runner).entropy).all()
    update_once(runner)
    assert runner.alg.optimizer.state
    for stochastic in (False, True):
        assert actor(obs, stochastic_output=stochastic).abs().le(1).all()


def test_save_and_full_resume_preserve_beta_optimizer_weights_and_counters(tmp_path):
    source, _ = make_runner()
    update_once(source)
    source.current_learning_iteration = 17
    source.env.unwrapped.common_step_counter = 765
    path = tmp_path / "new.pt"
    source.save(str(path), {"user": "保留"})
    assert source.env.get_observations()["actor"].shape == (8, 51)
    assert source.env.get_observations()["critic"].shape == (8, 66)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["infos"]["snake_recipe"] == {
        "version": 22, "motion_frame": "body_gravity", "head_gyro_frame": "head", "motion_body": "foot_0",
        "motion_site": "foot_0_sole", "tail_site": "foot_1_sole",
        "body_yaw_limit": math.pi / 3, "body_yaw_reference": "episode_initial_heading",
        "body_yaw_integral": "signed_episode_dt_tanh_squared",
        "body_yaw_integral_time_scale": 1.,
        "objective": "snake_world_x_capped_speed",
        "forward_speed_cap": .2,
        "motion_penalty_speed_gain": 5.,
        "motion_penalty_min_scale": 0.,
        "motion_penalty_scaling": "clipped_positive_world_x_ramp",
        "yaw_reference": "episode_initial_heading", "head_yaw_limit": math.pi / 3,
        "head_yaw_integral": "signed_episode_dt_tanh_squared",
        "head_yaw_integral_time_scale": 1.,
        "head_orientation_observation": "quaternion_wxyz_and_relative_yaw_sincos",
        "actor_observation_dim": 51, "critic_observation_dim": 66,
        "observation_reference": "world_x",
        "distribution": "BetaDistribution", "action_range": (-1., 1.),
        "action_control": "integrated_joint_velocity", "joint_target_limit": math.pi / 3,
        "max_target_speed": 2 * math.pi,
    }
    assert checkpoint["infos"]["user"] == "保留"
    target, _ = make_runner()
    target.load(str(path))
    assert target.current_learning_iteration == 17
    assert target.env.unwrapped.common_step_counter == 765
    assert target.alg.optimizer.state
    for key, value in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(value, source.alg.get_policy().state_dict()[key])
    assert any(state["step"] == 1 for state in target.alg.optimizer.state.values())
    assert target.alg.optimizer.param_groups[0]["lr"] == source.alg.optimizer.param_groups[0]["lr"]


def last_head_key(checkpoint):
    return [key for key, value in checkpoint["actor_state_dict"].items()
            if key.startswith("mlp.") and key.endswith("weight") and value.ndim == 2][-1]


@pytest.mark.parametrize("fault", ["no_recipe", "version_2", "version_3", "version_4", "version_5", "version_6", "version_7", "version_10", "version_11", "missing_body", "base_body", "missing_site", "other_site", "frame", "gyro", "distribution", "range", "head", "objective", "observation_reference"])
def test_full_resume_preflight_rejects_incompatible_inputs_without_side_effects(tmp_path, fault):
    source, _ = make_runner()
    path = tmp_path / "incompatible.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint["infos"]["snake_recipe"]
    if fault == "no_recipe":
        checkpoint["infos"].pop("snake_recipe")
    elif fault == "version_2":
        recipe["version"] = 2
    elif fault == "version_3":
        recipe["version"] = 3
        recipe.pop("motion_body")
        recipe.pop("motion_site")
    elif fault == "version_4":
        recipe["version"] = 4
        recipe.pop("motion_site")
    elif fault == "version_5":
        recipe["version"] = 5
        recipe.pop("objective")
        recipe.pop("observation_reference")
    elif fault in ("version_6", "version_7"):
        recipe["version"] = int(fault[-1])
        if fault == "version_6":
            recipe["objective"] = "world_x_signed_speed"
    elif fault in ("version_10", "version_11"):
        recipe["version"] = int(fault.split("_")[1])
    elif fault == "missing_body":
        recipe.pop("motion_body")
    elif fault == "base_body":
        recipe["motion_body"] = "base"
    elif fault == "missing_site":
        recipe.pop("motion_site")
    elif fault == "other_site":
        recipe["motion_site"] = "foot_1_sole"
    elif fault == "frame":
        recipe["motion_frame"] = "legacy_head"
    elif fault == "gyro":
        recipe["head_gyro_frame"] = "world"
    elif fault == "distribution":
        recipe["distribution"] = "GaussianDistribution"
    elif fault == "range":
        recipe["action_range"] = (0., 1.)
    elif fault == "objective":
        recipe["objective"] = "velocity_tracking"
    elif fault == "observation_reference":
        recipe["observation_reference"] = "command"
    else:
        key = last_head_key(checkpoint)
        checkpoint["actor_state_dict"][key] = checkpoint["actor_state_dict"][key][:6].clone()
    torch.save(checkpoint, path)
    target, refreshes = make_runner()
    before = state_copy(target)
    target.env.unwrapped.common_step_counter = 123
    target.current_learning_iteration = 9
    with pytest.raises(ValueError):
        target.load(str(path))
    for key, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[key])
    assert target.env.unwrapped.common_step_counter == 123
    assert target.current_learning_iteration == 9
    assert not target.alg.optimizer.state and not refreshes


def test_prepare_new_beta_actor_only_keeps_training_state_untouched(tmp_path):
    source, _ = make_runner()
    update_once(source)
    source.current_learning_iteration = 77
    source.env.unwrapped.common_step_counter = 876
    path = tmp_path / "beta.pt"
    source.save(str(path))
    cfg = runner_cfg()
    SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"] == {"class_name": "BetaDistribution", "action_range": (-1., 1.)}
    runner, refreshes = make_runner(cfg)
    runner.env.unwrapped.common_step_counter = 123
    runner.current_learning_iteration = 9
    runner.load(str(path), load_cfg={"actor": True})
    for key, value in runner.alg.get_policy().state_dict().items():
        torch.testing.assert_close(value, source.alg.get_policy().state_dict()[key])
    assert runner.env.unwrapped.common_step_counter == 123
    assert runner.current_learning_iteration == 9
    assert not runner.alg.optimizer.state and len(refreshes) == 1


def attach_curriculum(runner, play=False):
    from mjlab.envs import ManagerBasedRlEnv

    from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg

    env = runner.env.unwrapped
    env.num_envs, env.device, env.step_dt = 8, 'cpu', .02
    cfg = SnakeVelocityCommandCfg(
        resampling_time_range=(5., 9.), training=not play,
        translation_min_speed=.02 if not play else 0.,
        translation_curriculum=(.06, .10, .15, .20, .25) if not play else (),
    )
    term = cfg.build(cast(ManagerBasedRlEnv, env))
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    return term


def test_full_resume_restores_curriculum_but_actor_only_playback_does_not(tmp_path):
    source, _ = make_runner()
    term = attach_curriculum(source)
    term.translation_stage = 2
    path = tmp_path / 'curriculum.pt'
    source.save(str(path))
    target, _ = make_runner()
    restored = attach_curriculum(target)
    target.load(str(path))
    assert restored.translation_stage == 2
    assert restored.current_translation_cap == .15
    playback, _ = make_runner()
    play_term = attach_curriculum(playback, play=True)
    playback.load(str(path), load_cfg={'actor': True})
    assert play_term.translation_stage == 0
    assert play_term.current_translation_cap == .25


def test_incompatible_curriculum_rejected_before_weights_change(tmp_path):
    source, _ = make_runner()
    attach_curriculum(source)
    path = tmp_path / 'wrong_curriculum.pt'
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint['infos']['snake_curriculum']['caps'] = (.1, .25)
    torch.save(checkpoint, path)
    target, _ = make_runner()
    term = attach_curriculum(target)
    before = state_copy(target)
    with pytest.raises(ValueError, match='课程'):
        target.load(str(path))
    assert term.translation_stage == 0
    for key, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[key])


@pytest.mark.parametrize("version", [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
def test_legacy_checkpoints_rejected_before_playback_network_or_weights_change(tmp_path, version):
    source, _ = make_runner()
    path = tmp_path / 'legacy.pt'
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    if version is None:
        checkpoint['infos'].pop('snake_recipe')
    else:
        checkpoint['infos']['snake_recipe']['version'] = version
    torch.save(checkpoint, path)
    cfg = runner_cfg()
    cfg['actor']['distribution_cfg'] = {'class_name': 'UnchangedDistribution'}
    with pytest.raises(ValueError, match='从零训练'):
        SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg['actor']['distribution_cfg'] == {'class_name': 'UnchangedDistribution'}
    target, refreshes = make_runner()
    before = state_copy(target)
    for load_cfg in (None, {'actor': True}):
        with pytest.raises(ValueError, match='从零训练'):
            target.load(str(path), load_cfg=load_cfg)
    assert not refreshes
    assert not target.alg.optimizer.state
    for key, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[key])


@pytest.mark.parametrize("key,value", [
    ("action_control", None),
    ("action_control", "default_pose_joint_position"),
    ("action_control", "direct_joint_position"),
    ("max_target_speed", None),
    ("max_target_speed", 0.),
    ("max_target_speed", -1.),
    ("max_target_speed", float("nan")),
    ("max_target_speed", float("inf")),
    ("max_target_speed", True),
    ("max_target_speed", {".*": 1 / 3}),
    ("joint_target_limit", None),
    ("joint_target_limit", 0.),
    ("joint_target_limit", -1.),
    ("joint_target_limit", True),
    ("joint_target_limit", float("nan")),
    ("joint_target_limit", float("inf")),
])
def test_invalid_action_recipe_rejected_before_playback_or_weights_change(tmp_path, key, value):
    source, _ = make_runner()
    path = tmp_path / "wrong_action.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint["infos"]["snake_recipe"]
    if value is None:
        recipe.pop(key)
    else:
        recipe[key] = value
    torch.save(checkpoint, path)
    cfg = runner_cfg()
    cfg["actor"]["distribution_cfg"] = {"class_name": "UnchangedDistribution"}
    with pytest.raises(ValueError, match="动作元数据"):
        SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"] == {"class_name": "UnchangedDistribution"}
    target, refreshes = make_runner()
    before = state_copy(target)
    for load_cfg in (None, {"actor": True}):
        with pytest.raises(ValueError, match="动作元数据"):
            target.load(str(path), load_cfg=load_cfg)
    assert not refreshes and not target.alg.optimizer.state
    for parameter_name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[parameter_name])


@pytest.mark.parametrize("attribute,value", [("target_limit", math.pi / 2),
                                            ("max_target_speed", 1.)])
def test_full_resume_joint_mapping_mismatch_rejected_before_weights_change(tmp_path, attribute, value):
    source, _ = make_runner()
    path = tmp_path / "other_limit.pt"
    source.save(str(path))
    target, refreshes = make_runner()
    setattr(target.env.unwrapped.cfg.actions["joint_pos"], attribute, value)
    before = state_copy(target)
    with pytest.raises(ValueError):
        target.load(str(path))
    assert not refreshes and not target.alg.optimizer.state
    for key, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[key])


@pytest.mark.parametrize("key,value", [
    ("yaw_reference", None), ("yaw_reference", "world_x"),
    ("head_orientation_observation", None), ("head_orientation_observation", "quaternion_xyzw"),
    ("head_yaw_limit", None), ("head_yaw_limit", 0.), ("head_yaw_limit", -1.),
    ("head_yaw_limit", math.pi + .01), ("head_yaw_limit", float("nan")),
    ("head_yaw_limit", float("inf")), ("head_yaw_limit", True),
    ("actor_observation_dim", 43), ("critic_observation_dim", 58),
    ("actor_observation_dim", 49), ("critic_observation_dim", 64),
    ("actor_observation_dim", 50), ("critic_observation_dim", 65),
])
def test_invalid_heading_metadata_rejected_before_playback_or_weights_change(tmp_path, key, value):
    source, _ = make_runner()
    path = tmp_path / "wrong_heading.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint["infos"]["snake_recipe"]
    if value is None:
        recipe.pop(key)
    else:
        recipe[key] = value
    torch.save(checkpoint, path)
    cfg = runner_cfg()
    cfg["actor"]["distribution_cfg"] = {"class_name": "UnchangedDistribution"}
    with pytest.raises(ValueError, match="初始航向元数据"):
        SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"] == {"class_name": "UnchangedDistribution"}
    target, refreshes = make_runner()
    before = state_copy(target)
    for load_cfg in (None, {"actor": True}):
        with pytest.raises(ValueError, match="初始航向元数据"):
            target.load(str(path), load_cfg=load_cfg)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("actor_only", [False, True])
def test_heading_limit_mismatch_rejected_before_weights_change(tmp_path, actor_only):
    source, _ = make_runner()
    path = tmp_path / "heading_limit.pt"
    source.save(str(path))
    target, refreshes = make_runner()
    target.env.unwrapped.cfg.head_yaw_limit = math.pi / 2
    before = state_copy(target)
    with pytest.raises(ValueError):
        target.load(str(path), load_cfg={"actor": True} if actor_only else None)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("network,width", [("actor", 43), ("critic", 58), ("actor", 49), ("critic", 64), ("actor", 50), ("critic", 65)])
def test_old_observation_weights_rejected_even_with_current_recipe(tmp_path, network, width):
    source, _ = make_runner()
    path = tmp_path / "old_observations.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    weights = checkpoint[f"{network}_state_dict"]
    first = min((int(key.split('.')[1]), key) for key in weights
                if key.startswith("mlp.") and key.endswith(".weight"))[1]
    weights[first] = weights[first][:, :width].clone()
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="观测维度"):
        SnakeTeacherRunner.prepare_playback_cfg(runner_cfg(), str(path))






def test_full_resume_preserves_learned_action_mean_and_std(tmp_path):
    source, _ = make_runner()
    actor = source.alg.get_policy()
    head = [layer for layer in actor.mlp.modules() if isinstance(layer, torch.nn.Linear)][-1]
    with torch.no_grad():
        head.bias[:6].add_(20.)
        head.bias[6:].sub_(10.)
    obs = source.env.get_observations()
    actor(obs, stochastic_output=True)
    expected_mean = distribution(source).mean.clone()
    expected_std = distribution(source).std.clone()
    assert expected_mean.abs().min() > .1
    path = tmp_path / "learned.pt"
    source.save(str(path))
    target, _ = make_runner()
    target.load(str(path))
    target.alg.get_policy()(obs, stochastic_output=True)
    torch.testing.assert_close(distribution(target).mean, expected_mean)
    torch.testing.assert_close(distribution(target).std, expected_std)


@pytest.mark.parametrize("key,value", [
    ("body_yaw_reference", None), ("body_yaw_reference", "world_x"),
    ("tail_site", None), ("tail_site", ""), ("tail_site", 1),
    ("body_yaw_limit", None), ("body_yaw_limit", 0.), ("body_yaw_limit", -1.),
    ("body_yaw_limit", math.pi + .01), ("body_yaw_limit", float("nan")),
    ("body_yaw_limit", float("inf")), ("body_yaw_limit", True),
])
def test_invalid_body_heading_metadata_rejected_before_playback_or_weights_change(tmp_path, key, value):
    source, _ = make_runner()
    path = tmp_path / "wrong_body_heading.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint["infos"]["snake_recipe"]
    if value is None:
        recipe.pop(key)
    else:
        recipe[key] = value
    torch.save(checkpoint, path)
    cfg = runner_cfg()
    cfg["actor"]["distribution_cfg"] = {"class_name": "UnchangedDistribution"}
    with pytest.raises(ValueError, match="体轴初始航向元数据"):
        SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"] == {"class_name": "UnchangedDistribution"}
    target, refreshes = make_runner()
    before = state_copy(target)
    for load_cfg in (None, {"actor": True}):
        with pytest.raises(ValueError, match="体轴初始航向元数据"):
            target.load(str(path), load_cfg=load_cfg)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize("attribute,value", [("tail_site", "other_tail"),
                                            ("body_yaw_limit", math.pi / 2)])
def test_body_heading_mismatch_rejected_before_weights_change(tmp_path, actor_only, attribute, value):
    source, _ = make_runner()
    path = tmp_path / "body_heading_limit.pt"
    source.save(str(path))
    target, refreshes = make_runner()
    setattr(target.env.unwrapped.cfg, attribute, value)
    before = state_copy(target)
    with pytest.raises(ValueError):
        target.load(str(path), load_cfg={"actor": True} if actor_only else None)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("integral_kind", ["body", "head"])
@pytest.mark.parametrize("key,value", [
    ("body_yaw_integral", None), ("body_yaw_integral", "absolute_episode_dt"),
    ("body_yaw_integral_time_scale", None), ("body_yaw_integral_time_scale", 0.),
    ("body_yaw_integral_time_scale", -1.), ("body_yaw_integral_time_scale", True),
    ("body_yaw_integral_time_scale", float("nan")),
    ("body_yaw_integral_time_scale", float("inf")),
])
def test_invalid_yaw_integral_metadata_rejected_before_playback_or_weights_change(tmp_path, key, value, integral_kind):
    key = key.replace("body_yaw", f"{integral_kind}_yaw")
    source, _ = make_runner()
    path = tmp_path / "wrong_body_integral.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint["infos"]["snake_recipe"]
    if value is None:
        recipe.pop(key)
    else:
        recipe[key] = value
    torch.save(checkpoint, path)
    cfg = runner_cfg()
    cfg["actor"]["distribution_cfg"] = {"class_name": "UnchangedDistribution"}
    with pytest.raises(ValueError, match="积分元数据"):
        SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"] == {"class_name": "UnchangedDistribution"}
    target, refreshes = make_runner()
    before = state_copy(target)
    for load_cfg in (None, {"actor": True}):
        with pytest.raises(ValueError, match="积分元数据"):
            target.load(str(path), load_cfg=load_cfg)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize("integral_kind", ["body", "head"])
def test_integral_time_scale_mismatch_rejected_before_weights_change(tmp_path, actor_only, integral_kind):
    source, _ = make_runner()
    path = tmp_path / "integral_time_scale.pt"
    source.save(str(path))
    target, refreshes = make_runner()
    setattr(target.env.unwrapped.cfg, f"{integral_kind}_yaw_integral_time_scale", 2.)
    before = state_copy(target)
    with pytest.raises(ValueError):
        target.load(str(path), load_cfg={"actor": True} if actor_only else None)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("key,value", [
    ("motion_penalty_scaling", None),
    ("motion_penalty_scaling", "positive_raw_forward_linear"),
    ("motion_penalty_min_scale", None), ("motion_penalty_min_scale", -.01),
    ("motion_penalty_min_scale", 1.01), ("motion_penalty_min_scale", True),
    ("motion_penalty_min_scale", "0.25"),
    ("motion_penalty_min_scale", float("nan")),
    ("motion_penalty_min_scale", float("inf")),
    ("motion_penalty_speed_gain", None), ("motion_penalty_speed_gain", -1.),
    ("motion_penalty_speed_gain", True), ("motion_penalty_speed_gain", "5"),
    ("motion_penalty_speed_gain", float("nan")),
    ("motion_penalty_speed_gain", float("inf")),
])
def test_invalid_motion_penalty_metadata_rejected_before_playback_or_weights_change(tmp_path, key, value):
    source, _ = make_runner()
    path = tmp_path / "wrong_motion_penalty.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint["infos"]["snake_recipe"]
    if value is None:
        recipe.pop(key)
    else:
        recipe[key] = value
    torch.save(checkpoint, path)
    cfg = runner_cfg()
    cfg["actor"]["distribution_cfg"] = {"class_name": "UnchangedDistribution"}
    with pytest.raises(ValueError, match="速度缩放元数据"):
        SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"] == {"class_name": "UnchangedDistribution"}
    target, refreshes = make_runner()
    before = state_copy(target)
    for load_cfg in (None, {"actor": True}):
        with pytest.raises(ValueError, match="速度缩放元数据"):
            target.load(str(path), load_cfg=load_cfg)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize("attribute,value", [("motion_penalty_speed_gain", 10.),
                                            ("motion_penalty_min_scale", .5)])
def test_motion_penalty_mapping_mismatch_rejected_before_weights_change(tmp_path, actor_only, attribute, value):
    source, _ = make_runner()
    path = tmp_path / "motion_penalty_speed_gain.pt"
    source.save(str(path))
    target, refreshes = make_runner()
    setattr(target.env.unwrapped.cfg, attribute, value)
    before = state_copy(target)
    with pytest.raises(ValueError):
        target.load(str(path), load_cfg={"actor": True} if actor_only else None)
    assert not refreshes and not target.alg.optimizer.state
    for name, parameter in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(parameter, before[name])


def test_motion_penalty_zero_gain_metadata_is_valid(tmp_path):
    runner, _ = make_runner()
    runner.env.unwrapped.cfg.motion_penalty_speed_gain = 0.
    path = tmp_path / "constant_motion_penalty.pt"
    runner.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    assert SnakeTeacherRunner._describe_checkpoint(checkpoint)["motion_penalty_speed_gain"] == 0.


@pytest.mark.parametrize("min_scale", [0., 1.])
def test_motion_penalty_min_scale_boundary_metadata_is_valid(tmp_path, min_scale):
    runner, _ = make_runner()
    runner.env.unwrapped.cfg.motion_penalty_min_scale = min_scale
    path = tmp_path / "motion_penalty_floor.pt"
    runner.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    assert SnakeTeacherRunner._describe_checkpoint(checkpoint)["motion_penalty_min_scale"] == min_scale


def test_previous_head_frame_objective_is_rejected():
    runner, _ = make_runner()
    checkpoint = {"infos": {"snake_recipe": runner._recipe()}}
    checkpoint['infos']['snake_recipe']['objective'] = 'snake_head_frame_capped_speed'
    with pytest.raises(ValueError, match='坐标'):
        SnakeTeacherRunner._describe_checkpoint(checkpoint)


def test_full_resume_reports_changed_joint_limit_and_exact_cli_override(tmp_path):
    source, _ = make_runner()
    path = tmp_path / 'original_range.pt'
    source.save(str(path))
    target, _ = make_runner()
    target.env.unwrapped.cfg.actions['joint_pos'].target_limit = math.pi
    before = state_copy(target)
    with pytest.raises(ValueError) as error:
        target.load(str(path))
    message = str(error.value)
    assert 'joint_target_limit' in message
    assert f'检查点={math.pi / 3!r}' in message
    assert f'当前={math.pi!r}' in message
    assert f'--env.actions.joint-pos.target-limit {math.pi / 3}' in message
    for key, value in before.items():
        torch.testing.assert_close(target.alg.get_policy().state_dict()[key], value)
    target.env.unwrapped.cfg.actions['joint_pos'].target_limit = math.pi / 3
    target.load(str(path))
    for key, value in state_copy(source).items():
        torch.testing.assert_close(target.alg.get_policy().state_dict()[key], value)


def test_explicit_joint_limit_transfer_restores_full_state_and_saves_new_recipe(tmp_path):
    source, _ = make_runner()
    update_once(source)
    source.current_learning_iteration = 1499
    source.env.unwrapped.common_step_counter = 36000
    original = tmp_path / 'original.pt'
    source.save(str(original))
    target, _ = make_runner()
    target.env.unwrapped.cfg.actions['joint_pos'].target_limit = math.pi
    target.env.unwrapped.cfg.allow_resume_joint_limit_change = True
    target.load(str(original))
    assert target.current_learning_iteration == 1499
    assert target.env.unwrapped.common_step_counter == 36000
    assert target.alg.optimizer.state
    for key, value in state_copy(source).items():
        torch.testing.assert_close(target.alg.get_policy().state_dict()[key], value)
    update_once(target)
    transferred = tmp_path / 'transferred.pt'
    target.save(str(transferred))
    saved = torch.load(transferred, weights_only=False)
    assert saved['infos']['snake_recipe']['joint_target_limit'] == math.pi
    assert saved['infos']['snake_resume_joint_limit_change']['old_limit'] == math.pi / 3
    assert saved['infos']['snake_resume_joint_limit_change']['new_limit'] == math.pi
    assert torch.load(original, weights_only=False)['infos']['snake_recipe']['joint_target_limit'] == math.pi / 3


def test_joint_limit_transfer_flag_does_not_bypass_other_compatibility_checks(tmp_path):
    source, _ = make_runner()
    original = tmp_path / 'original.pt'
    source.save(str(original))
    target, _ = make_runner()
    target.env.unwrapped.cfg.actions['joint_pos'].target_limit = math.pi
    target.env.unwrapped.cfg.allow_resume_joint_limit_change = True
    target.env.unwrapped.cfg.actions['joint_pos'].max_target_speed = math.pi
    before = state_copy(target)
    with pytest.raises(ValueError):
        target.load(str(original))
    for key, value in before.items():
        torch.testing.assert_close(target.alg.get_policy().state_dict()[key], value)


@pytest.mark.parametrize('attribute,value', [('target_limit', math.pi), ('max_target_speed', math.pi)])
def test_actor_only_load_accepts_joint_mapping_change_and_preserves_playback_configuration(tmp_path, attribute, value):
    source, _ = make_runner()
    update_once(source)
    source.env.unwrapped.common_step_counter = 123
    path = tmp_path / 'original_mapping.pt'
    source.save(str(path))
    target, refreshes = make_runner()
    action = target.env.unwrapped.cfg.actions['joint_pos']
    setattr(action, attribute, value)
    target.env.unwrapped.common_step_counter = 7
    target.load(str(path), load_cfg={'actor': True})
    assert getattr(action, attribute) == value
    assert target.env.unwrapped.common_step_counter == 7
    assert not target.alg.optimizer.state
    assert refreshes
    for key, parameter in state_copy(source).items():
        torch.testing.assert_close(target.alg.get_policy().state_dict()[key], parameter)
