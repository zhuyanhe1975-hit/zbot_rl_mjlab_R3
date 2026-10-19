"""``bests/`` 中归档的每个检查点都必须仍能用于当前代码。

这些低开销检查仅使用 CPU，对每个归档模型及其回放任务执行
（任务名称见 ``bests/test.sh``）。检查可发现导致旧检查点无法回放的变更：
策略观测项的集合或顺序、观测或动作维度、控制频率、动作缩放，以及执行器。
这些检查不会运行仿真器；``python scripts/fall_check.py <TASK> <checkpoint>`` 会运行仿真器，
每次修改物理设置后都应执行。"""

import fnmatch
import re
from pathlib import Path

import pytest
import torch
import yaml
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import zbot_rl_mjlab  # noqa: F401  （注册任务）
from zbot_rl_mjlab.robot import ARMATURE, DAMPING, SIM_EFFORT_LIMIT, STIFFNESS
from zbot_rl_mjlab.tasks.transformation.actions import MirroredJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.walking.env_cfg import ZbotEnvCfg

BESTS = Path(__file__).resolve().parents[1] / "bests"
ENTRIES = sorted(p for p in BESTS.iterdir() if p.is_dir())

# 每个策略观测项的维度。
TERM_DIMS = {
    "snake": 40,
    "teacher_state": 36,
    "universal_state": 50,
    "command": 4,
    "gait": 3,
    "command_teacher_state": 37,
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "yaw_error": 1,
    "joint_pos": 6,
    "joint_vel": 6,
    "actions": 6,
    "phase": 2,
    "frequency": 1,
    "target_speed": 1,
    "velocity_command": 3,
    "last_action": 6,
    "com_state": 2,
    "commanded_target": 6,
    "hold_time": 1,
    "support_state": 0,  # 默认反向任务模式兼容 30 维检查点。
    "joint_goal_error": 6,
    "joint_velocity": 6,
    "independent_target": 3,
    "independent_last_action": 3,
    "base_angular_velocity": 3,
    "base_height": 1,
    "base_linear_velocity": 3,
    "base_world_vertical_velocity": 1,
    "ground_contacts": 3,
    "sole_heights": 2,
    "sole_down_cosine": 2,
}
ACTION_DIM = 6


class _Loose(yaml.SafeLoader):
    """读取已保存的 env.yaml，不导入其中命名的 Python 对象。"""


def _sequence(loader, node):
    return loader.construct_sequence(node, deep=True)


def _python_object(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return suffix
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)


_Loose.add_constructor("tag:yaml.org,2002:python/tuple", _sequence)
_Loose.add_multi_constructor("tag:yaml.org,2002:python/", _python_object)


def archived_env(entry):
    return yaml.load((entry / "env.yaml").read_text(), Loader=_Loose)


def task_of(entry):
    """bests/test.sh 回放此目录时使用的任务（采用第一个匹配分支）。"""
    cases = re.findall(
        r'^[ \t]*([^)\s][^)\n]*)\)[ \t]*echo[ \t]*"(Mjlab-[^"]+)"',
        (BESTS / "test.sh").read_text(),
        re.MULTILINE,
    )
    for patterns, task in cases:
        if any(fnmatch.fnmatchcase(entry.name, p.strip()) for p in patterns.split("|")):
            return task
    raise AssertionError(f"bests/test.sh has no task for {entry.name}")


def checkpoint(entry):
    models = sorted(entry.glob("model_*.pt"))
    assert len(models) == 1, f"{entry.name} should hold exactly one model_*.pt"
    return torch.load(models[0], map_location="cpu", weights_only=False)


def live_actor_terms(task):
    return list(load_env_cfg(task, play=True).observations["actor"].terms)


def test_every_actor_term_of_every_task_has_a_known_size():
    # 必须在此处为新观测项指定维度，否则下面的检查无法检测该项。
    for task in (t for t in list_tasks() if t.startswith("Mjlab-Zbot")):
        unknown = set(live_actor_terms(task)) - set(TERM_DIMS)
        assert not unknown, f"{task}: add {sorted(unknown)} to TERM_DIMS"


def test_there_is_something_to_check():
    assert ENTRIES, "bests/ is empty"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_archived_model_matches_the_current_task(entry):
    task = task_of(entry)
    live = load_env_cfg(task, play=True)
    saved = archived_env(entry)

    saved_terms = list(saved["observations"]["actor"]["terms"])
    assert saved_terms == live_actor_terms(task), (
        f"{entry.name}: the actor's observation terms changed since it was trained"
    )

    state = checkpoint(entry)["actor_state_dict"]
    obs_dim = sum(
        13
        if name == "support_state" and saved.get("support_observations", False)
        else TERM_DIMS[name]
        for name in saved_terms
    )
    if "obs_normalizer._mean" in state:
        assert state["obs_normalizer._mean"].shape[-1] == obs_dim
    assert state["mlp.0.weight"].shape[1] == obs_dim
    agent = load_rl_cfg(task)
    assert isinstance(agent, RslRlOnPolicyRunnerCfg)
    obs = TensorDict({"actor": torch.zeros(2, obs_dim)}, batch_size=[2])
    action_dim = 3 if isinstance(live.actions["joint_pos"], MirroredJointTargetRateActionCfg) else ACTION_DIM
    actor = MLPModel(
        obs=obs,
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=action_dim,
        hidden_dims=agent.actor.hidden_dims,
        activation=agent.actor.activation,
        obs_normalization=agent.actor.obs_normalization,
        distribution_cfg=agent.actor.distribution_cfg,
    )
    actor.load_state_dict(state, strict=True)
    with torch.no_grad():
        actions = actor(obs)
    assert actions.shape == (2, action_dim)
    assert torch.isfinite(actions).all()

    if "frequency" in saved_terms:
        assert isinstance(live, ZbotEnvCfg)
        # 频率观测按此范围归一化，因此更改范围后，即使所有维度仍然匹配，
        # 策略接收到的输入也会不同。
        assert (saved["step_frequency_min"], saved["step_frequency_max"]) == (
            live.step_frequency_min,
            live.step_frequency_max,
        ), f"{entry.name}: the commanded stepping-frequency range changed"
    assert saved["decimation"] == live.decimation
    assert saved["sim"]["mujoco"]["timestep"] == live.sim.mujoco.timestep
    action = saved["actions"]["joint_pos"]
    live_action = live.actions["joint_pos"]
    assert isinstance(live_action, JointPositionActionCfg)
    assert action["scale"] == live_action.scale
    assert action["use_default_offset"] == live_action.use_default_offset


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_archived_model_was_trained_with_the_current_actuator(entry):
    (actuator,) = archived_env(entry)["scene"]["entities"]["robot"]["articulation"][
        "actuators"
    ]
    assert actuator["stiffness"] == STIFFNESS
    assert actuator["damping"] == DAMPING
    assert actuator["effort_limit"] == SIM_EFFORT_LIMIT
    assert actuator["armature"] == ARMATURE
