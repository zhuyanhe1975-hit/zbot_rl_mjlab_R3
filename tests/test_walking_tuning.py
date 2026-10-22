import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.experiments import walking_tuning as tuning
from zbot_rl_mjlab.experiments.cards import load_card
from zbot_rl_mjlab.tasks.walking.env_cfg import WALKING_TASK_ID

TASK = tuning.WALKING_TUNING_TASK_ID


def rewards(task, play=False):
    return load_env_cfg(task, play=play).rewards


@pytest.mark.parametrize("play", [False, True])
def test_has_the_same_structure_as_walking(play):
    """奖励（名称、顺序、函数）、观测和事件均与 Walking 相同。

本任务用于调节权重、奖励参数和频率范围，因此允许这些值不同。"""
    base, new = rewards(WALKING_TASK_ID, play), rewards(TASK, play)
    assert list(base) == list(new)
    for name in base:
        assert base[name].func is new[name].func, name
    base_cfg, new_cfg = load_env_cfg(WALKING_TASK_ID, play), load_env_cfg(TASK, play)
    assert list(base_cfg.observations["actor"].terms) == list(
        new_cfg.observations["actor"].terms
    )
    assert base_cfg.events.keys() == new_cfg.events.keys()
    assert (new_cfg.step_frequency_min, new_cfg.step_frequency_max) == (0.25, 1.0)


def test_table_covers_every_reward_and_card_weights_take_priority():
    assert set(tuning.REWARD_WEIGHTS) == set(rewards(TASK))
    overrides = load_card('walking_tuning')['rewards']
    for name, weight in tuning.REWARD_WEIGHTS.items():
        assert rewards(TASK)[name].weight == overrides.get(name, {}).get('weight', weight)


def test_has_its_own_experiment_name_and_schedule():
    agent, base = load_rl_cfg(TASK), load_rl_cfg(WALKING_TASK_ID)
    assert agent.experiment_name == "zbot_walking_tuning"
    assert agent.experiment_name != base.experiment_name
    assert agent.max_iterations == tuning.MAX_ITERATIONS == 600
    assert load_env_cfg(TASK).scene.num_envs == tuning.NUM_ENVS


def test_editing_the_tuning_task_does_not_touch_walking():
    # load_env_cfg 返回深拷贝，因此在内存中修改某个任务的配置时，不得
    # 影响其他任务的已注册配置，也不得影响该任务自身的已注册配置。
    first = load_env_cfg(TASK)
    first.rewards["forward_velocity"].weight = 123.0
    assert (
        rewards(TASK)["forward_velocity"].weight
        == tuning.REWARD_WEIGHTS["forward_velocity"]
    )
    assert rewards(WALKING_TASK_ID)["forward_velocity"].weight == 2.0


def test_reward_params_are_applied():
    assert rewards(TASK)["swing_clearance"].params["target"] == 0.05
    assert rewards(TASK)["foot_nonparallel_ground"].params["std"] == 0.1
