"""``bests/`` 中归档的每个检查点都必须仍能用于当前代码。

这些低开销检查仅使用 CPU，对每个归档模型及其回放任务执行
（任务名称见 ``bests/test.sh``）。检查可发现导致旧检查点无法回放的变更：
策略观测项的集合或顺序、观测或动作维度、控制频率、动作缩放，以及执行器。
这些检查不会运行仿真器；``python scripts/fall_check.py <TASK> <checkpoint>`` 会运行仿真器，
每次修改物理设置后都应执行。"""

import fnmatch
import hashlib
import json
import re
from dataclasses import make_dataclass
from pathlib import Path

import mjlab
import pytest
import torch
import tyro
import yaml
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import zbot_rl_mjlab  # noqa: F401  （注册任务）
from zbot_rl_mjlab.archived_site_snake import ArchivedSiteSnakeTeacherRunner
from zbot_rl_mjlab.gaussian_action_models import (
    GaussianActionRoutedTaskNormalizedMLP,
    GaussianTaskDistribution,
    expand_mirror,
)
from zbot_rl_mjlab.guided_teacher import GuidedTeacherRunner
from zbot_rl_mjlab.robot import ARMATURE, DAMPING, SIM_EFFORT_LIMIT, STIFFNESS
from zbot_rl_mjlab.student import TEACHER_SHA256
from zbot_rl_mjlab.student_models import HistoryGaussianStudentMLP
from zbot_rl_mjlab.tasks.snake.mdp import ACTOR_DIM
from zbot_rl_mjlab.tasks.transformation.actions import MirroredJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal.student_observations import GuidedStudentEnvCfg
from zbot_rl_mjlab.tasks.walking.env_cfg import ZbotEnvCfg

BESTS = Path(__file__).resolve().parents[1] / "bests"
ENTRIES = sorted(p for p in BESTS.iterdir() if p.is_dir())

# 每个策略观测项的维度。
TERM_DIMS = {
    "sensors": 56,
    "privileged": 59,
    "snake": ACTOR_DIM,
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


def actor_term_dim(name, saved):
    if name == "support_state" and saved.get("support_observations", False):
        return 13
    if name == "privileged":
        encoding = saved["commands"]["transform"].get(
            "task_observation_encoding", "one_hot"
        )
        assert encoding in ("scalar", "one_hot")
        return 56 if encoding == "scalar" else 59
    return TERM_DIMS[name]


@pytest.mark.parametrize("encoding, expected", [("scalar", 56), ("one_hot", 59), (None, 59)])
def test_archived_privileged_observation_encoding(encoding, expected):
    command = {} if encoding is None else {"task_observation_encoding": encoding}
    assert actor_term_dim("privileged", {"commands": {"transform": command}}) == expected


def test_every_actor_term_of_every_task_has_a_known_size():
    # 必须在此处为新观测项指定维度，否则下面的检查无法检测该项。
    for task in (t for t in list_tasks() if t.startswith("Mjlab-Zbot")):
        unknown = set(live_actor_terms(task)) - set(TERM_DIMS)
        assert not unknown, f"{task}: add {sorted(unknown)} to TERM_DIMS"


def test_there_is_something_to_check():
    assert ENTRIES, "bests/ is empty"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_archived_files_match_sha256_manifest(entry):
    manifest = entry / "SHA256SUMS"
    if not manifest.exists():
        assert not entry.name.startswith("snake_site_omni_"), "site snake archive needs SHA256SUMS"
        return
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = entry / name.removeprefix("*")
        assert path.resolve().is_relative_to(entry.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda p: p.name)
def test_archived_model_matches_the_current_task(entry):
    task = task_of(entry)
    live = load_env_cfg(task, play=True)
    playback_args = entry / "playback_args.json"
    if playback_args.exists():
        # Match bests/test.sh: archived CLI overrides are part of playback.
        playback_config = make_dataclass('ArchivePlayback', [('env', type(live))])
        live = tyro.cli(
            playback_config,
            default=playback_config(env=live),
            args=json.loads(playback_args.read_text()),
            config=mjlab.TYRO_FLAGS,
        ).env
    saved = archived_env(entry)

    saved_terms = list(saved["observations"]["actor"]["terms"])
    assert saved_terms == live_actor_terms(task), (
        f"{entry.name}: the actor's observation terms changed since it was trained"
    )

    saved_checkpoint = checkpoint(entry)
    if task == "Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher":
        ArchivedSiteSnakeTeacherRunner._validate_checkpoint(saved_checkpoint)
    state = saved_checkpoint["actor_state_dict"]
    saved_group = saved['observations']['actor']
    saved_history = saved_group.get('history_length') or 1
    live_history = live.observations['actor'].history_length or 1
    assert saved_history == live_history, f'{entry.name}: actor history length changed'
    obs_dim = sum(actor_term_dim(name, saved) for name in saved_terms) * saved_history
    if "obs_normalizer._mean" in state:
        assert state["obs_normalizer._mean"].shape[-1] == obs_dim
    agent = load_rl_cfg(task)
    assert isinstance(agent, RslRlOnPolicyRunnerCfg)
    guided = task == "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided"
    student = agent.actor.class_name == 'zbot_rl_mjlab.student_models:HistoryGaussianStudentMLP'
    if guided:
        GuidedTeacherRunner._validate_metadata(saved_checkpoint)
        metadata = saved_checkpoint["infos"]["privileged_omni_teacher"]
        assert metadata["task_input_encoding"] == "one_hot_local_gaussian_mirror_v1"
        assert metadata["network_input_dim"] == 61
        assert metadata["task_action_dims"] == (6, 3, 6, 3)
        assert obs_dim == 56
        assert state["obs_normalizer._mean"].shape == (4, 56)
        assert agent.actor.class_name == (
            "zbot_rl_mjlab.gaussian_action_models:GaussianActionRoutedTaskNormalizedMLP"
        )
    elif student:
        assert isinstance(live, GuidedStudentEnvCfg)
        assert saved_terms == ['sensors']
        assert saved_history == saved['student_history_length'] == live.student_history_length == 5
        assert saved_group['flatten_history_dim'] is live.observations['actor'].flatten_history_dim is True
        assert obs_dim == 280
        assert state['obs_normalizer.frame_normalizer._mean'].shape == (4, 56)
        expected_schema = {
            'role': 'universal_sensor_history_student', 'schema_version': 1,
            'frame_dim': 56, 'history_length': saved_history, 'imu_reference': 'base',
            'task_routing': 'software_fsm_v1', 'task_encoding': 'one_hot_recovery_v1',
            'action_contract': 'gaussian_position6_rate6_mirror3_v1',
            'task_action_dims': (6, 3, 6, 3), 'teacher_sha256': TEACHER_SHA256,
        }
        assert saved_checkpoint['infos']['universal_student'] == expected_schema
        assert saved_checkpoint['infos']['teacher_source']['sha256'] == TEACHER_SHA256
        dagger = saved_checkpoint['student_dagger']
        assert dagger['contract']['algorithm'] == 'universal_teacher_dagger_v2'
        assert dagger['contract']['teacher_sha256'] == TEACHER_SHA256
        assert dagger['completed_updates'] == saved_checkpoint['iter'] + 1
        saved_agent = yaml.load((entry / 'agent.yaml').read_text(), Loader=_Loose)
        assert saved_agent['actor']['class_name'] == agent.actor.class_name
        for name, value in dagger['contract'].items():
            if name != 'algorithm':
                assert saved_agent['algorithm'][name] == value
        metadata = json.loads((entry / 'metadata.json').read_text())
        model_path, = entry.glob('model_*.pt')
        assert metadata['task_id'] == task
        assert metadata['checkpoint_iter_field'] == saved_checkpoint['iter']
        assert metadata['checkpoint_iteration_filename'] == int(model_path.stem.removeprefix('model_'))
        assert metadata['completed_dagger_iterations'] == dagger['completed_updates']
        assert metadata['recipe'] == json.loads(json.dumps(expected_schema))
        assert metadata['teacher_source'] == saved_checkpoint['infos']['teacher_source']
        assert metadata['source_sha256'] == hashlib.sha256(model_path.read_bytes()).hexdigest()
        source = Path(metadata['source_checkpoint'])
        assert not source.is_absolute() and '..' not in source.parts
        assert source.parent == Path(metadata['source_run'])
        assert source.name == model_path.name
        manifest = (entry / 'SHA256SUMS').read_text()
        listed = {line.split(maxsplit=1)[1].removeprefix('*') for line in manifest.splitlines()}
        assert {model_path.name, 'env.yaml', 'agent.yaml', 'metadata.json'} <= listed
        assert agent.actor.distribution_cfg is not None
        assert agent.actor.distribution_cfg['class_name'].endswith(':GaussianTaskDistribution')
        assert agent.actor.distribution_cfg['learn_std'] is False
    else:
        assert state["mlp.0.weight"].shape[1] == obs_dim
    batch_size = 4 if guided or student else 2
    raw_obs = torch.zeros(batch_size, obs_dim)
    if guided:
        raw_obs[:, 39] = torch.arange(4)
    if student:
        raw_obs.reshape(4, saved_history, 56)[:, :, 39] = torch.arange(4)[:, None]
    obs = TensorDict({"actor": raw_obs}, batch_size=[batch_size])
    action_dim = 3 if isinstance(live.actions["joint_pos"], MirroredJointTargetRateActionCfg) else ACTION_DIM
    actor_class = HistoryGaussianStudentMLP if student else GaussianActionRoutedTaskNormalizedMLP if guided else MLPModel
    actor = actor_class(
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
    actor.eval()
    with torch.no_grad():
        actions = actor(obs)
    assert actions.shape == (batch_size, action_dim)
    assert torch.isfinite(actions).all()
    if guided or student:
        assert isinstance(actor, (GaussianActionRoutedTaskNormalizedMLP, HistoryGaussianStudentMLP))
        assert isinstance(actor.distribution, GaussianTaskDistribution)
        for head, width in zip(actor.mlp.heads, (6, 3, 6, 3), strict=True):
            linears = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
            assert linears[0].in_features == 61 * saved_history
            assert linears[-1].out_features == width
        with torch.no_grad():
            sampled = actor(obs, stochastic_output=True)
        assert sampled.shape == (4, 6)
        assert torch.isfinite(sampled).all()
        for output in (actions, sampled):
            assert torch.equal(output[[1, 3]], expand_mirror(output[[1, 3], :3]))

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
