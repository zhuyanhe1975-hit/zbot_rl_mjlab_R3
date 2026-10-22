"""重构前十九个任务的训练、回放与 PPO 契约，保留字典项顺序。"""

import dataclasses
import enum
import functools
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import zbot_rl_mjlab
from zbot_rl_mjlab.experiments.base import WalkingExperiment
from zbot_rl_mjlab.experiments.snake import SnakeOmniTeacher
from zbot_rl_mjlab.tasks.snake import mdp as snake_mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import SnakeEnvCfg

BASELINE = Path(__file__).parent / "data" / "task_contracts.json.gz"
with gzip.open(BASELINE, "rt", encoding="utf-8") as stream:
    PRE_REFACTOR_CONTRACTS = json.load(stream)


def canonical(value):
    """仅忽略模块搬迁；字段值、字典顺序、回调名称和类型仍参与比较。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__qualname__, "name": value.name}
    if isinstance(value, functools.partial):
        return {"partial": canonical(value.func), "args": canonical(value.args),
                "keywords": canonical(value.keywords)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # 下降任务从通用配置分离出具名子类，不改变其参数契约。
        name = type(value).__qualname__
        if name == "BipedToSnakeEnvCfg":
            name = "TransformationEnvCfg"
        return {"type": name, "fields": {
            f.name: canonical(getattr(value, f.name))
            for f in sorted(dataclasses.fields(value), key=lambda f: f.name)
        }}
    if isinstance(value, dict):
        return {"items": [[canonical(k), canonical(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "values": [canonical(v) for v in value]}
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, Path):
        return {"path": str(value)}
    if callable(value):
        return {"callable": value.__qualname__}
    return {"type": type(value).__qualname__,
            "value": re.sub(r"0x[0-9a-fA-F]+", "<地址>", repr(value))}


def task_contract(experiment):
    task = experiment.task_id
    return {
        "train": canonical(load_env_cfg(task)),
        "play": canonical(load_env_cfg(task, play=True)),
        "ppo": canonical(load_rl_cfg(task)),
        "runner": canonical(load_runner_cls(task)),
    }


@pytest.mark.parametrize("experiment", [cls for cls in zbot_rl_mjlab.EXPERIMENTS
                         if cls.task_id in PRE_REFACTOR_CONTRACTS],
                         ids=lambda cls: cls.task_id)
def test_registered_task_matches_pre_refactor_contract(experiment):
    assert task_contract(experiment) == PRE_REFACTOR_CONTRACTS[experiment.task_id]


def test_registered_task_ids_are_preserved():
    assert [cls.task_id for cls in zbot_rl_mjlab.EXPERIMENTS] == [
        *PRE_REFACTOR_CONTRACTS, "Mjlab-Zbot-6dof-Universal-Locomotion",
        "Mjlab-Zbot-6dof-Universal-Locomotion-Deploy",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Wide-Biped",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Multi-Critic",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Task-Critic",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Task-Routed",
        "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided",
        "Mjlab-Zbot-6dof-Snake-Omni-Teacher",
        "Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher",
        "Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher",
        "Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher",
        "Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher",
        "Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher",
    ]


@pytest.mark.parametrize("play", [False, True])
def test_new_snake_teacher_has_an_independent_nonwalking_registration(play):
    """新教师另立合约；十九个原任务仍逐项使用未修改的重构前快照。"""
    task = "Mjlab-Zbot-6dof-Snake-Omni-Teacher"
    assert SnakeOmniTeacher.task_id == task
    assert task not in PRE_REFACTOR_CONTRACTS
    assert not issubclass(SnakeOmniTeacher, WalkingExperiment)
    cfg = load_env_cfg(task, play=play)
    assert isinstance(cfg, SnakeEnvCfg)
    assert list(cfg.commands) == ["motion"]
    assert cfg.observations["actor"].terms["snake"].func is snake_mdp.actor_state
    assert cfg.observations["critic"].terms["snake"].func is snake_mdp.critic_state
    assert (snake_mdp.ACTOR_DIM, snake_mdp.CRITIC_DIM) == (51, 66)
    assert load_rl_cfg(task).experiment_name == "zbot_snake_omni_teacher"
    from zbot_rl_mjlab.experiments.snake import SnakeTeacherRunner

    assert load_runner_cls(task) is SnakeTeacherRunner


@pytest.mark.parametrize("config_file", sorted(
    (Path(__file__).resolve().parents[1] / "bests").glob("*/*.yaml")
), ids=lambda path: f"{path.parent.name}/{path.name}")
def test_archived_project_callback_paths_remain_resolvable(config_file):
    """只解析本项目的 Python 标签，不执行归档 YAML 的对象构造器。"""
    import importlib

    names = re.findall(
        r"!!python/(?:name|object|object/new|object/apply):(zbot_rl_mjlab[^\s]+)",
        config_file.read_text(),
    )
    for name in names:
        module, _, symbol = name.rpartition(".")
        assert hasattr(importlib.import_module(module), symbol), name
