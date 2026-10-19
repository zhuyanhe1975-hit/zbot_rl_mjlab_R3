from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.experiments import addons
from zbot_rl_mjlab.experiments import variants as deploy
from zbot_rl_mjlab.tasks.walking.actions import (
    DelayedJointPositionAction,
    DelayedJointPositionActionCfg,
)

TASKS = [deploy.DEPLOY_WALKING_TASK_ID, deploy.DEPLOY_FLAT_FOOT_TASK_ID]
ACTOR_TERMS = [
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "phase",
    "frequency",
    "target_speed",
]
FLAT_TERMS = ("foot_flat_stance", "foot_flat_touchdown", "foot_flat_pre_landing")


@pytest.mark.parametrize("task", TASKS)
def test_actor_is_blind_to_privileged_terms_but_the_critic_keeps_them(task):
    for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        assert list(cfg.observations["actor"].terms) == ACTOR_TERMS
        critic = cfg.observations["critic"].terms
        for name in addons.PRIVILEGED_TERMS:
            assert name in critic
        assert all(term.noise is None for term in critic.values())
        assert not cfg.observations["critic"].enable_corruption


@pytest.mark.parametrize("task", TASKS)
def test_actor_noise_is_training_only_and_biased_only_where_specified(task):
    train = load_env_cfg(task).observations["actor"]
    play = load_env_cfg(task, play=True).observations["actor"]
    assert train.enable_corruption and not play.enable_corruption
    for name in ACTOR_TERMS:
        noise = train.terms[name].noise
        if name in addons.NOISE_BIAS:
            assert isinstance(noise, NoiseModelWithAdditiveBiasCfg)
            assert noise.noise_cfg.std == addons.NOISE_STD[name]
        elif name in addons.NOISE_STD:
            assert isinstance(noise, GaussianNoiseCfg)
            assert noise.std == addons.NOISE_STD[name]
        else:
            assert noise is None


@pytest.mark.parametrize("task", TASKS)
def test_action_delay_is_on_in_training_and_off_for_play(task):
    train = load_env_cfg(task).actions["joint_pos"]
    play = load_env_cfg(task, play=True).actions["joint_pos"]
    assert isinstance(train, DelayedJointPositionActionCfg)
    assert train.max_lag == addons.MAX_ACTION_LAG
    assert play.max_lag == 0
    assert (train.scale, train.use_default_offset) == (0.25, True)


@pytest.mark.parametrize("task", TASKS)
def test_randomization_events_are_present_and_mass_comes_before_com(task):
    for play in (False, True):
        events = list(load_env_cfg(task, play=play).events)
        for name in (
            "randomize_mass_inertia",
            "randomize_pd_gains",
            "base_disturbance",
            "randomize_foot_friction",
            "randomize_encoder_bias",
            "randomize_base_com",
        ):
            assert name in events
        # pseudo_inertia 会根据默认值重写 body_ipos，因此必须先执行。
        assert events.index("randomize_mass_inertia") < events.index(
            "randomize_base_com"
        )


def test_variants_differ_only_in_the_foot_reward():
    plain = load_env_cfg(deploy.DEPLOY_WALKING_TASK_ID)
    flat = load_env_cfg(deploy.DEPLOY_FLAT_FOOT_TASK_ID)
    assert not any(name in plain.rewards for name in FLAT_TERMS)
    assert not plain.curriculum
    assert all(name in flat.rewards for name in FLAT_TERMS)
    assert "foot_nonparallel_ground" not in flat.rewards
    assert set(plain.rewards) - set(flat.rewards) == {"foot_nonparallel_ground"}
    assert set(flat.rewards) - set(plain.rewards) == set(FLAT_TERMS)
    assert list(plain.events) == list(flat.events)


def test_experiment_names_schedule_and_curriculum():
    for task, name in (
        (deploy.DEPLOY_WALKING_TASK_ID, "zbot_deploy_walking"),
        (deploy.DEPLOY_FLAT_FOOT_TASK_ID, "zbot_deploy_flat_foot"),
    ):
        agent = load_rl_cfg(task)
        assert agent.experiment_name == name
        assert agent.max_iterations == deploy.MAX_ITERATIONS
    agent = load_rl_cfg(deploy.DEPLOY_FLAT_FOOT_TASK_ID)
    stages = (
        load_env_cfg(deploy.DEPLOY_FLAT_FOOT_TASK_ID)
        .curriculum["foot_flat_stance_weight"]
        .params["stages"]
    )
    assert [s["step"] for s in stages] == [
        row[0] * agent.num_steps_per_env for row in deploy.DEPLOY_WEIGHT_STAGES
    ]
    assert stages[-1]["weight"] == -0.5


def _delayed_action(lags):
    """不依赖环境的 DelayedJointPositionAction，每个延迟值对应一个环境。"""
    action = object.__new__(DelayedJointPositionAction)
    action.cfg = SimpleNamespace(max_lag=2, clip=None)
    action._scale = 1.0
    action._offset = 0.0
    action._raw_actions = torch.zeros(len(lags), 1)
    action._history = torch.zeros(3, len(lags), 1)
    action._lag = torch.tensor(lags)
    action._rows = torch.arange(len(lags))
    return action


def test_delay_applies_the_action_from_lag_steps_ago():
    action = _delayed_action([0, 1, 2])
    seen = []
    for value in (1.0, 2.0, 3.0, 4.0):
        action.process_actions(torch.full((3, 1), value))
        seen.append(action._processed_actions[:, 0].tolist())
    # 延迟 0 跟随命令，延迟 1 和 2 滞后于命令（尚无历史记录时为零）。
    assert seen[0] == [1.0, 0.0, 0.0]
    assert seen[1] == [2.0, 1.0, 0.0]
    assert seen[2] == [3.0, 2.0, 1.0]
    assert seen[3] == [4.0, 3.0, 2.0]
