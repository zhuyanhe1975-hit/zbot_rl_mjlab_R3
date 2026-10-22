"""Lock all experiment recipes while environment assembly moves out of task mechanics."""
import dataclasses
import functools
import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from mjlab.tasks.registry import load_env_cfg, load_runner_cls
from test_task_contracts import task_contract

import zbot_rl_mjlab

with gzip.open(Path(__file__).parent / 'data/experiment_configs.json.gz', 'rt') as stream:
    CONFIGURATIONS = json.load(stream)


def callback_paths(value: Any, path: str = '') -> dict[str, str]:
    result: dict[str, str] = {}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            result.update(callback_paths(getattr(value, field.name), path + '.' + field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(callback_paths(item, path + '.' + str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(callback_paths(item, path + '.' + str(index)))
    elif isinstance(value, functools.partial):
        for name in ('func', 'args', 'keywords'):
            result.update(callback_paths(getattr(value, name), path + '.' + name))
    elif callable(value):
        result[path] = value.__module__ + '.' + value.__qualname__
    return result


@pytest.mark.parametrize('experiment', zbot_rl_mjlab.EXPERIMENTS, ids=lambda cls: cls.task_id)
def test_complete_recipe_and_runtime_callbacks_are_preserved(experiment):
    task = experiment.task_id
    actual = task_contract(experiment) | {'callbacks': {
        'train': callback_paths(load_env_cfg(task)),
        'play': callback_paths(load_env_cfg(task, play=True)),
        'runner': callback_paths(load_runner_cls(task)),
    }}
    assert actual == CONFIGURATIONS[task]
