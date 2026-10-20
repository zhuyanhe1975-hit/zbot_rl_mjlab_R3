"""四向速度课程的独立升降档、恢复保护与检查点兼容契约。"""

from types import SimpleNamespace
from typing import cast

import pytest
import test_universal_distillation as distillation_tests
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from test_universal_distillation import rollout

from zbot_rl_mjlab import universal_evaluation
from zbot_rl_mjlab.distillation import MultiTeacherRunner

DIRECTIONS = ('forward', 'backward', 'left', 'right')
algorithm = distillation_tests.algorithm


class Command:
    def __init__(self):
        self.cfg = SimpleNamespace(training=True, forward_speed_levels=(.08, .10, .12, .15, .20))
        self.motion_speed_stages = dict.fromkeys(DIRECTIONS, 0)
        self.motion_speed_passes = dict.fromkeys(DIRECTIONS, 0)
        self.motion_speed_failures = dict.fromkeys(DIRECTIONS, 0)
        self.tracking_ema = torch.zeros(2)

    @property
    def forward_speed_stage(self):
        return self.motion_speed_stages['forward']

    @property
    def forward_speed_passes(self):
        return self.motion_speed_passes['forward']

    def course_levels(self, direction):
        return self.cfg.forward_speed_levels if direction == 'forward' else (.05, .075, .10)

    def set_motion_speed_stage(self, direction, stage):
        if type(stage) is not int or not 0 <= stage < len(self.course_levels(direction)):
            raise ValueError('无效档位')
        self.motion_speed_stages[direction] = stage

    def set_pretraining(self, enabled):
        self.pretraining = enabled


@pytest.fixture
def runner_setup(algorithm, monkeypatch, tmp_path, request):
    alg, obs, _ = algorithm
    alg.distillation_enabled = getattr(request, 'param', True)
    if not alg.distillation_enabled:
        alg.teachers = None
    alg.robust_validation_metrics = {'robust_ascent_success': 1.}
    command = Command()
    physical = SimpleNamespace(common_step_counter=123, cfg=SimpleNamespace(), command_manager=SimpleNamespace(get_term=lambda name: command))
    env = SimpleNamespace(unwrapped=physical, reset=lambda: (obs, {}))

    def base_init(self, env, cfg, log_dir, device):
        self.alg, self.env, self.cfg = alg, env, {'upload_model': False}
        self.current_learning_iteration = 10

    class Evaluator:
        foundation_samples = None

        def __init__(self, **kwargs):
            self.kwargs, self.calls, self.closed = kwargs, [], False
            self.forward_speed = kwargs.get('forward_speed', .08)
            self.motion_direction = kwargs.get('motion_direction', 'forward')
            self.result = {'validation_passed': True, 'ascent_success': 1., 'forward_success': 1., 'forward_no_fall_fraction': 1.}

        def __call__(self, actor):
            self.calls.append((self.motion_direction, self.forward_speed))
            return dict(self.result)

        def close(self):
            self.closed = True

    created = []

    def load_teachers(*args, **kwargs):
        assert alg.distillation_enabled, '纯 PPO 不得加载教师'
        return alg.teachers

    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', base_init)
    monkeypatch.setattr('zbot_rl_mjlab.teachers.TeacherEnsemble', load_teachers)
    monkeypatch.setattr(MultiTeacherRunner, '_teacher_provenance', lambda self: {})
    monkeypatch.setattr(universal_evaluation, 'UniversalEvaluator', lambda **kwargs: created.append(Evaluator(**kwargs)) or created[-1])
    runner = MultiTeacherRunner(cast(RslRlVecEnvWrapper, env), {'console_mode': 'full'}, log_dir=str(tmp_path))
    return runner, command, created, obs


def check_direction(runner, direction, attempt):
    runner.alg.ppo_start_iteration = 0
    runner.alg.distillation_iteration = (attempt * 4 + DIRECTIONS.index(direction)) * runner.alg.validation_interval_iterations
    return runner.alg.validation_callback()


def test_four_directions_advance_independently_after_three_fresh_checks(runner_setup):
    runner, command, evaluators, _ = runner_setup
    speed = evaluators[1]
    assert speed.kwargs['keyboard_protocol'] is True
    for attempt in range(3):
        result = check_direction(runner, 'left', attempt)
        assert len(speed.calls) == attempt + 1
        runner.alg.validation_callback()
        assert len(speed.calls) == attempt + 1
    assert command.motion_speed_stages == {'forward': 0, 'backward': 0, 'left': 1, 'right': 0}
    assert command.motion_speed_passes['left'] == 0
    assert result['motion_course_left_speed'] == .05
    assert result['motion_course_left_next_speed'] == .075
    for direction in DIRECTIONS:
        check_direction(runner, direction, 10)
        assert speed.calls[-1] == (direction, command.course_levels(direction)[command.motion_speed_stages[direction]])


@pytest.mark.parametrize('source', ['missing_robust', 'bad_robust', 'primary'])
def test_upgrade_requires_primary_and_known_good_robust_recovery(runner_setup, source):
    runner, command, evaluators, _ = runner_setup
    if source == 'missing_robust':
        runner.alg.robust_validation_metrics.clear()
    elif source == 'bad_robust':
        runner.alg.robust_validation_metrics['robust_ascent_success'] = .79
    else:
        evaluators[0].result['validation_passed'] = False
    for attempt in range(3):
        check_direction(runner, 'forward', attempt)
    assert command.motion_speed_stages['forward'] == 0
    assert command.motion_speed_passes['forward'] == 0


@pytest.mark.parametrize('bad_key,bad_value', [('forward_success', .59), ('forward_no_fall_fraction', .89)])
def test_three_significant_direction_failures_retreat_one_grade(runner_setup, bad_key, bad_value):
    runner, command, evaluators, _ = runner_setup
    command.set_motion_speed_stage('backward', 2)
    evaluators[1].result[bad_key] = bad_value
    for attempt in range(2):
        check_direction(runner, 'backward', attempt)
        assert command.motion_speed_stages['backward'] == 2
    result = check_direction(runner, 'backward', 2)
    assert command.motion_speed_stages['backward'] == 1
    assert command.motion_speed_failures['backward'] == 0
    assert result['motion_course_rollback_reason'] == 1


def test_middle_band_resets_direction_hysteresis_and_terminal_grade_is_bounded(runner_setup):
    runner, command, evaluators, _ = runner_setup
    command.set_motion_speed_stage('right', 2)
    command.motion_speed_passes['right'] = command.motion_speed_failures['right'] = 2
    evaluators[1].result['forward_success'] = .7
    check_direction(runner, 'right', 0)
    assert command.motion_speed_passes['right'] == command.motion_speed_failures['right'] == 0
    evaluators[1].result['forward_success'] = 1.
    for attempt in range(1, 4):
        check_direction(runner, 'right', attempt)
    assert command.motion_speed_stages['right'] == 2


@pytest.mark.parametrize('source', ['primary', 'robust'])
def test_three_recovery_failures_retreat_all_advanced_grades_once(runner_setup, source):
    runner, command, evaluators, _ = runner_setup
    for direction in DIRECTIONS:
        command.set_motion_speed_stage(direction, 2)
    if source == 'primary':
        evaluators[0].result['ascent_success'] = .89
    else:
        runner.alg.robust_validation_metrics['robust_ascent_success'] = .79
    for index in range(3):
        result = check_direction(runner, DIRECTIONS[index], 0)
    assert set(command.motion_speed_stages.values()) == {1}
    assert set(command.motion_speed_passes.values()) == set(command.motion_speed_failures.values()) == {0}
    assert result['motion_course_rollback_reason'] == 2


def test_absent_robust_metrics_do_not_trigger_rollback(runner_setup):
    runner, command, _, _ = runner_setup
    command.set_motion_speed_stage('forward', 2)
    runner.alg.robust_validation_metrics.clear()
    for attempt in range(3):
        check_direction(runner, 'forward', attempt)
    assert command.motion_speed_stages['forward'] == 2


def test_warmup_skips_motion_evaluation_and_refresh_and_closes_resources(runner_setup, monkeypatch):
    runner, command, evaluators, _ = runner_setup
    refreshed = []
    runner.alg.refresh_anchor = refreshed.append
    runner.alg.validation_callback()
    assert len(evaluators[0].calls) == 1 and not evaluators[1].calls
    assert not refreshed and set(command.motion_speed_passes.values()) == {0}
    monkeypatch.setattr(MjlabOnPolicyRunner, 'learn', lambda *args: None)
    runner.learn(1)
    assert all(evaluator.closed for evaluator in evaluators)


@pytest.mark.parametrize('runner_setup', [False], indirect=True)
def test_pure_ppo_course_advances_without_teachers(runner_setup):
    runner, command, _, obs = runner_setup
    alg = runner.alg
    assert alg.teachers is None and alg.phase_callback is None
    alg.validation_interval_iterations = 1
    for _ in range(12):
        rollout(alg, obs)
        alg.update()
    assert set(command.motion_speed_stages.values()) == {1}


def test_v2_checkpoint_roundtrip_and_actor_only_preserve_course(runner_setup, tmp_path):
    runner, command, _, _ = runner_setup
    path = tmp_path / 'course.pt'
    command.set_motion_speed_stage('forward', 3)
    command.set_motion_speed_stage('left', 1)
    command.motion_speed_passes['right'] = 2
    command.motion_speed_failures['backward'] = 1
    runner._motion_recovery_failures = 2
    runner.save(str(path), {'custom': 17})
    saved = torch.load(path, weights_only=False)
    state = saved['infos']['motion_speed_course']
    assert state['version'] == 2 and state['stages'] == command.motion_speed_stages
    assert state['levels']['left'] == [.05, .075, .10]
    assert saved['infos']['forward_speed_course']['stage'] == 3
    command.set_motion_speed_stage('forward', 0)
    runner.load(str(path))
    assert command.forward_speed_stage == 3
    assert command.motion_speed_passes['right'] == 2
    assert command.motion_speed_failures['backward'] == 1
    assert runner._motion_recovery_failures == 2
    command.set_motion_speed_stage('forward', 1)
    runner.load(str(path), {'actor': True})
    assert command.forward_speed_stage == 1


@pytest.mark.parametrize('legacy', [True, False])
def test_legacy_course_migration_caps_forward_and_clears_counts(runner_setup, tmp_path, legacy):
    runner, command, _, _ = runner_setup
    path = tmp_path / 'old.pt'
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    saved['infos'].pop('motion_speed_course')
    if legacy:
        saved['infos']['forward_speed_course']['stage'] = 4
    else:
        saved['infos'].pop('forward_speed_course')
    torch.save(saved, path)
    runner.load(str(path))
    assert command.motion_speed_stages == dict.fromkeys(DIRECTIONS, 0) | {'forward': int(legacy)}
    assert set(command.motion_speed_passes.values()) == set(command.motion_speed_failures.values()) == {0}


@pytest.mark.parametrize('field,value', [('version', 3), ('levels', {}), ('stages', {'forward': 9}), ('passes', dict.fromkeys(DIRECTIONS, 3)), ('failures', dict.fromkeys(DIRECTIONS, -1)), ('recovery_failures', 3)])
def test_invalid_v2_metadata_is_rejected_before_training_state_mutation(runner_setup, tmp_path, field, value):
    runner, _, _, _ = runner_setup
    path = tmp_path / 'invalid.pt'
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    saved['infos']['motion_speed_course'][field] = value
    torch.save(saved, path)
    runner.env.unwrapped.common_step_counter = 7
    runner.current_learning_iteration = 4
    with pytest.raises(ValueError):
        runner.load(str(path))
    assert runner.env.unwrapped.common_step_counter == 7 and runner.current_learning_iteration == 4


def test_partial_success_anchor_refresh_primary_direction_and_robust_are_isolated(runner_setup):
    runner, _, evaluators, obs = runner_setup
    samples = (obs[:4].clone(), torch.zeros(4, 6), torch.ones(4), torch.tensor([0, 1, 2, 0]), torch.zeros(4, dtype=torch.bool))
    primary, speed = evaluators
    primary.foundation_samples = speed.foundation_samples = samples
    primary.result['validation_passed'] = False
    refreshed = []
    runner.alg.refresh_anchor = refreshed.append
    check_direction(runner, 'forward', 0)
    assert len(refreshed) == 2
    assert len(refreshed[0][1]) == 4
    assert len(refreshed[1][1]) == 2 and refreshed[1][3].eq(0).all()
    runner._robust_evaluator = primary
    runner._run_robust_validation()
    assert len(refreshed) == 3 and len(refreshed[2][1]) == 4
