"""统一实验模板的钩子顺序、注册与继承边界。"""

from types import SimpleNamespace
from typing import Any

import zbot_rl_mjlab
from zbot_rl_mjlab.experiments import base
from zbot_rl_mjlab.experiments.base import Experiment


def test_hooks_run_once_from_parent_to_child_before_addons():
    calls = []

    class Addon:
        order = 0

        def apply(self, cfg, play):
            calls.append("addon")

    class Parent(Experiment):
        addons = (Addon(),)
        train_envs = 7

        def create_env(self, play=False) -> Any:
            calls.append("factory")
            return SimpleNamespace(scene=SimpleNamespace(num_envs=1), rewards={})

        def edit(self, cfg, play):
            calls.append("parent")
            assert cfg.scene.num_envs == (1 if play else 7)

    class Child(Parent):
        def edit(self, cfg, play):
            calls.append("child")

    for play in (False, True):
        calls.clear()
        Child().build_env(play=play)
        assert calls == ["factory", "parent", "child", "addon"]


def test_registration_uses_selected_runner_and_both_environment_modes(monkeypatch):
    captured = {}

    class Runner:
        pass

    class ConfiguredExperiment(Experiment):
        task_id = "Mjlab-Zbot-Architecture-Test"
        runner_cls = Runner

        def create_env(self, play=False) -> Any:
            return SimpleNamespace(play=play)

        def build_ppo(self) -> Any:
            return "ppo"

    monkeypatch.setattr(base, "register_mjlab_task", lambda **kw: captured.update(kw))
    ConfiguredExperiment().register()
    assert captured["runner_cls"] is Runner
    assert not captured["env_cfg"].play
    assert captured["play_env_cfg"].play
    assert captured["rl_cfg"] == "ppo"


def test_every_registered_experiment_uses_one_build_and_register_template():
    for experiment in zbot_rl_mjlab.EXPERIMENTS:
        assert experiment.build_env is Experiment.build_env
        assert experiment.build_ppo is Experiment.build_ppo
        assert experiment.register is Experiment.register


def test_transformation_directions_and_bidirectional_are_not_concrete_parent_tasks():
    from zbot_rl_mjlab.experiments.bidirectional import SnakeBipedBidirectional
    from zbot_rl_mjlab.experiments.transformation import (
        BipedToSnake,
        SnakeToBiped,
        SnakeToBipedTeacher,
    )

    assert not issubclass(SnakeToBiped, BipedToSnake)
    assert not issubclass(SnakeBipedBidirectional, SnakeToBipedTeacher)


def test_task_mechanics_do_not_depend_on_experiment_recipes():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'src/zbot_rl_mjlab'
    for path in (root / 'tasks').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert 'experiments' not in (node.module or ''), f'{path}:{node.lineno}'
            elif isinstance(node, ast.Import):
                assert all('experiments' not in alias.name for alias in node.names), f'{path}:{node.lineno}'


def test_runtime_runners_do_not_define_training_recipes():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / 'src/zbot_rl_mjlab'
    for path in root.glob('*.py'):
        functions = {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        assert not any(name.endswith('ppo_cfg') for name in functions), path


def test_registered_experiments_have_independent_matching_cards():
    from zbot_rl_mjlab.experiments.cards import load_card

    names = [experiment.card_name for experiment in zbot_rl_mjlab.EXPERIMENTS]
    assert len(set(names)) == len(names)
    for experiment in zbot_rl_mjlab.EXPERIMENTS:
        assert experiment.card_name is not None
        assert load_card(experiment.card_name)['task_id'] == experiment.task_id
