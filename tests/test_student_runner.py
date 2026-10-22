"""Full resume must align fresh FSM timing with the restored simulation clock."""

from types import SimpleNamespace
from typing import Any

import pytest

from zbot_rl_mjlab.student import MjlabOnPolicyRunner, UniversalStudentRunner


class ResumeEnvironment:
    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(student_history_length=5)
        self.common_step_counter = 0
        self.command = SimpleNamespace(start_step=0)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.command.start_step = self.common_step_counter


def runner(monkeypatch):
    instance: Any = UniversalStudentRunner.__new__(UniversalStudentRunner)
    instance.env = ResumeEnvironment()
    instance.alg = SimpleNamespace(teacher_sha256='frozen-source', completed_updates=125)
    instance.current_learning_iteration = 0
    saved = {'infos': {'universal_student': instance._schema()}}
    monkeypatch.setattr('zbot_rl_mjlab.student.torch.load', lambda *args, **kwargs: saved)
    monkeypatch.setattr('zbot_rl_mjlab.student.load_universal_teacher',
                        lambda *args, **kwargs: pytest.fail('checkpoint loading must not initialize a teacher'))
    return instance


@pytest.mark.parametrize('load_cfg', [None, {'actor': True, 'critic': True, 'optimizer': True, 'iteration': True}])
def test_full_resume_rebases_fsm_after_upstream_restores_clock(monkeypatch, load_cfg):
    instance = runner(monkeypatch)
    result = {'restored': True}

    def upstream_load(self, path, cfg, strict, map_location):
        assert self.env.reset_calls == 0
        self.env.common_step_counter = 4000
        self.current_learning_iteration = 124
        return result

    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', upstream_load)
    assert instance.load('student.pt', load_cfg) is result
    assert instance.env.reset_calls == 1
    assert instance.env.common_step_counter == 4000
    assert instance.env.command.start_step == 4000
    assert instance.current_learning_iteration == 125
    assert instance.alg.completed_updates == 125
    # A fresh transformation must have zero elapsed time, rather than inheriting
    # thousands of restored steps and immediately failing the 16-second limit.
    assert (instance.env.common_step_counter - instance.env.command.start_step) * .02 == 0


def test_actor_only_playback_preserves_environment_without_loading_teacher(monkeypatch):
    instance = runner(monkeypatch)
    instance.env.common_step_counter = 37
    instance.env.command.start_step = 31
    instance.current_learning_iteration = 7
    result = {'playback': True}

    def upstream_load(self, path, cfg, strict, map_location):
        assert cfg == {'actor': True}
        return result

    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', upstream_load)
    assert instance.load('student.pt', {'actor': True}) is result
    assert instance.env.reset_calls == 0
    assert instance.env.common_step_counter == 37
    assert instance.env.command.start_step == 31
    assert instance.current_learning_iteration == 7
