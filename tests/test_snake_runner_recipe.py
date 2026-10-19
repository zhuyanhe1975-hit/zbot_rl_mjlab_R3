"""真实 CPU 策略检查有界 Beta 探索、检查点语义与旧 Gaussian 回放。"""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import BetaDistribution, GaussianDistribution
from tensordict import TensorDict

from zbot_rl_mjlab.experiments.snake import SnakeOmniTeacher, SnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake.env_cfg import SnakeEnvCfg


def runner_cfg():
    cfg = asdict(SnakeOmniTeacher().build_ppo())
    cfg["actor"]["hidden_dims"] = cfg["critic"]["hidden_dims"] = [16]
    cfg["upload_model"] = False
    return cfg


def make_runner(cfg=None):
    refreshes = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(motion_frame="body_gravity", head_gyro_frame="head", head_body="foot_0", motion_site="foot_0_sole", observation_reference="world_x"),
        common_step_counter=0,
        observation_manager=SimpleNamespace(compute=lambda **kwargs: refreshes.append(kwargs)),
    )
    obs = TensorDict({"actor": torch.randn(8, 40), "critic": torch.randn(8, 55)}, [8])
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
    loss = -distribution(runner).log_prob(samples.detach()).mean() - .002 * distribution(runner).entropy.mean()
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
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["infos"]["snake_recipe"] == {
        "version": 7, "motion_frame": "body_gravity", "head_gyro_frame": "head", "motion_body": "foot_0",
        "motion_site": "foot_0_sole",
        "objective": "snake_forward_signed_speed", "observation_reference": "world_x",
        "distribution": "BetaDistribution", "action_range": (-1., 1.),
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


@pytest.mark.parametrize("fault", ["no_recipe", "version_2", "version_3", "version_4", "version_5", "missing_body", "base_body", "missing_site", "other_site", "frame", "gyro", "distribution", "range", "head", "objective", "observation_reference"])
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
    for key, value in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert target.env.unwrapped.common_step_counter == 123
    assert target.current_learning_iteration == 9
    assert not target.alg.optimizer.state and not refreshes


def gaussian_checkpoint(path, versioned, std_type="scalar"):
    cfg = runner_cfg()
    cfg["actor"]["distribution_cfg"] = {
        "class_name": "GaussianDistribution", "init_std": .25,
        "std_range": (.08, .45) if versioned else (1e-6, 1e6), "std_type": std_type,
    }
    runner, _ = make_runner(cfg)
    with torch.no_grad():
        value = .3 if versioned else 1.7
        if std_type == "log":
            distribution(runner).log_std_param.fill_(torch.log(torch.tensor(value)).item())
        else:
            distribution(runner).std_param.fill_(value)
    payload: dict[str, Any] = runner.alg.save()
    payload["iter"] = 77
    infos: dict[str, Any] = {"env_state": {"common_step_counter": 876}}
    if versioned:
        infos["snake_recipe"] = {
            "version": 2, "motion_frame": "body_gravity", "head_gyro_frame": "head",
        }
    payload["infos"] = infos
    torch.save(payload, path)
    return payload


@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_prepare_gaussian_playback_preserves_original_distribution_and_frame(tmp_path, versioned, std_type):
    path = tmp_path / "gaussian.pt"
    gaussian_checkpoint(path, versioned, std_type)
    cfg = runner_cfg()
    SnakeTeacherRunner.prepare_playback_cfg(cfg, str(path))
    assert cfg["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution"
    assert cfg["actor"]["distribution_cfg"]["std_type"] == std_type
    expected_range = (.08, .45) if versioned else (1e-6, 1e6)
    assert tuple(cfg["actor"]["distribution_cfg"]["std_range"]) == expected_range
    runner, refreshes = make_runner(cfg)
    runner.env.unwrapped.common_step_counter = 123
    runner.current_learning_iteration = 9
    runner.load(str(path), load_cfg={"actor": True})
    assert isinstance(distribution(runner), GaussianDistribution)
    assert tuple(distribution(runner).std_range) == expected_range
    if std_type == "log":
        actual_std = distribution(runner).log_std_param.exp()
    else:
        actual_std = distribution(runner).std_param
    torch.testing.assert_close(actual_std, torch.full_like(actual_std, .3 if versioned else 1.7))
    env_cfg = cast(SnakeEnvCfg, runner.env.unwrapped.cfg)
    assert env_cfg.motion_frame == ("body_gravity" if versioned else "legacy_head")
    assert env_cfg.head_gyro_frame == ("head" if versioned else "world")
    assert env_cfg.observation_reference == 'command'
    assert runner.env.unwrapped.common_step_counter == 123
    assert runner.current_learning_iteration == 9
    assert not runner.alg.optimizer.state and len(refreshes) == 1


def test_direct_beta_runner_rejects_old_gaussian_actor_before_loading_any_weights(tmp_path):
    path = tmp_path / "gaussian.pt"
    gaussian_checkpoint(path, versioned=True)
    runner, _ = make_runner()
    before = state_copy(runner)
    with pytest.raises(ValueError):
        runner.load(str(path), load_cfg={"actor": True})
    for key, value in runner.alg.get_policy().state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert runner.env.unwrapped.common_step_counter == 0


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
    for key, value in target.alg.get_policy().state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_legacy_beta_actor_restores_command_reference_but_cannot_resume_world_task(tmp_path):
    source, _ = make_runner()
    path = tmp_path / 'legacy_beta.pt'
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    recipe = checkpoint['infos']['snake_recipe']
    recipe['version'] = 5
    recipe.pop('objective')
    recipe.pop('observation_reference')
    torch.save(checkpoint, path)
    playback, _ = make_runner()
    playback.load(str(path), load_cfg={'actor': True})
    assert playback.env.unwrapped.cfg.observation_reference == 'command'
    with pytest.raises(ValueError, match='蛇形前向'):
        playback.learn(1)
    fresh, _ = make_runner()
    with pytest.raises(ValueError, match='蛇形前向'):
        fresh.load(str(path))


def test_world_x_v6_checkpoint_can_play_but_cannot_resume_new_objective(tmp_path):
    source, _ = make_runner()
    path = tmp_path / 'world_x_v6.pt'
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint['infos']['snake_recipe'].update(version=6, objective='world_x_signed_speed')
    torch.save(checkpoint, path)
    target, _ = make_runner()
    target.load(str(path), load_cfg={'actor': True})
    assert target.env.unwrapped.cfg.observation_reference == 'world_x'
    with pytest.raises(ValueError, match='蛇形前向'):
        target.load(str(path))
