"""显式的原生 Beta 正式入口与高斯任务保持分离。"""

import io

import pytest
import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.models import MLPModel
from rsl_rl.modules import BetaDistribution, GaussianDistribution
from tensordict import TensorDict

from zbot_rl_mjlab.experiments.transformation import (
    BipedToSnake,
    SnakeToBiped,
    SnakeToBipedBeta,
)
from zbot_rl_mjlab.rl import SnakeToBipedHotRewardRunner
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as reverse
from zbot_rl_mjlab.tasks.transformation.reverse_observations import (
    reserved_hold_observation,
)


def load_agent(task_id):
    agent = load_rl_cfg(task_id)
    assert isinstance(agent, RslRlOnPolicyRunnerCfg)
    return agent


def make_actor(task_id):
    actor = load_agent(task_id).actor
    observation = TensorDict({"actor": torch.zeros(3, 30)}, batch_size=[3])
    return MLPModel(
        obs=observation,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=6,
        hidden_dims=actor.hidden_dims,
        activation=actor.activation,
        obs_normalization=actor.obs_normalization,
        distribution_cfg=actor.distribution_cfg,
    ), observation


def test_beta_registry_is_explicit_and_original_gaussian_tasks_are_unchanged():
    beta = load_agent(reverse.BETA_TASK_ID)
    assert beta.experiment_name == "zbot_transform_snake_to_biped_beta"
    assert beta.max_iterations == 4000
    assert beta.actor.distribution_cfg == {
        "class_name": "BetaDistribution",
        "action_range": (-1.0, 1.0),
    }
    assert load_runner_cls(reverse.BETA_TASK_ID) is SnakeToBipedHotRewardRunner
    for cls in (BipedToSnake, SnakeToBiped):
        agent = load_agent(cls.task_id)
        assert agent.actor.distribution_cfg is not None
        assert agent.actor.distribution_cfg["class_name"] == "GaussianDistribution"
        assert agent.experiment_name != beta.experiment_name
    assert beta.actor.distribution_cfg is not None
    beta.actor.distribution_cfg["action_range"] = (-0.5, 0.5)
    fresh = load_agent(reverse.BETA_TASK_ID).actor.distribution_cfg
    assert fresh is not None
    assert fresh["action_range"] == (-1.0, 1.0)


@pytest.mark.parametrize("play", [False, True])
def test_beta_recipe_preserves_physics_initial_pose_actions_and_observation_schema(
    play,
):
    beta = load_env_cfg(reverse.BETA_TASK_ID, play=play)
    original = load_env_cfg(reverse.TASK_ID, play=play)
    assert isinstance(beta, reverse.SnakeToBipedEnvCfg)
    assert isinstance(original, reverse.SnakeToBipedEnvCfg)
    assert beta.scene.num_envs == (1 if play else 8192)
    assert beta.curriculum_prepare_pose
    assert beta.first_stage_joint_weight == 0.6
    assert beta.first_stage_prepared_joint_weight == 3.0
    assert beta.standing_stage_joint_weight == 6.0
    assert beta.curriculum_joint_error_threshold == 1.0
    assert beta.rewards["joint_symmetry"].weight == -0.2
    assert beta.rewards["joint_standing"].weight == (6.0 if play else 0.6)
    assert not original.curriculum_prepare_pose
    assert original.first_stage_joint_weight == 0.1
    assert original.standing_stage_joint_weight == 1.0
    assert original.rewards["joint_symmetry"].weight == -1.0
    assert (
        beta.scene.entities["robot"].init_state
        == original.scene.entities["robot"].init_state
    )
    assert beta.actions == original.actions
    assert beta.terminations == original.terminations
    assert beta.decimation == original.decimation
    assert beta.sim == original.sim
    assert not beta.support_observations
    for group in ("actor", "critic"):
        assert beta.observations[group] == original.observations[group]
        assert (
            beta.observations[group].terms["hold_time"].func
            is reserved_hold_observation
        )
    experiment = SnakeToBipedBeta()
    experiment.train_envs = 17
    experiment.max_iterations = 19
    assert experiment.build_env().scene.num_envs == 17
    assert experiment.build_ppo().max_iterations == 19


def test_real_cpu_beta_actor_strict_checkpoint_load_returns_six_bounded_actions():
    actor, obs = make_actor(reverse.BETA_TASK_ID)
    assert isinstance(actor.distribution, BetaDistribution)
    state_file = io.BytesIO()
    torch.save(actor.state_dict(), state_file)
    state_file.seek(0)
    restored, _ = make_actor(reverse.BETA_TASK_ID)
    restored.load_state_dict(torch.load(state_file, weights_only=True), strict=True)
    result = restored(obs)
    assert result.shape == (3, 6)
    assert torch.isfinite(result).all() and result.abs().le(1).all()
    torch.testing.assert_close(result, actor(obs))
    # 原生导出保留 Beta 均值转换，不直接输出含 12 个值的原始输出头。
    torch.testing.assert_close(restored.as_onnx(verbose=False)(obs["actor"]), result)
    gaussian, _ = make_actor(reverse.TASK_ID)
    assert isinstance(gaussian.distribution, GaussianDistribution)
    with pytest.raises(RuntimeError):
        gaussian.load_state_dict(actor.state_dict(), strict=True)


def test_beta_eval_uses_native_task_configuration_and_reverse_options(monkeypatch):
    from scripts import transformation_eval

    class CapturedConfig(Exception):
        pass

    def capture(cfg, device):
        assert isinstance(cfg, reverse.SnakeToBipedEnvCfg)
        assert cfg.curriculum_prepare_pose
        assert cfg.curriculum_joint_error_threshold == 1.0
        assert cfg.curriculum_start_stage == 1
        raise CapturedConfig

    monkeypatch.setattr(transformation_eval, "ManagerBasedRlEnv", capture)
    with pytest.raises(CapturedConfig):
        transformation_eval.evaluate(
            task_id=reverse.BETA_TASK_ID, curriculum_stage=1, device="cpu"
        )
