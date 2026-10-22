"""Independent metrics and periodic evaluation must expose, not hide, regressions."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.student_ppo import (
    MjlabOnPolicyRunner,
    StudentPPORunnerCfg,
    UniversalStudentPPORunner,
)
from zbot_rl_mjlab.student_ppo_evaluation import SCENARIOS, summarize_student_traces
from zbot_rl_mjlab.student_ppo_selection import (
    REQUIRED_METRICS,
    StudentPPOCheckpointSelector,
)
from zbot_rl_mjlab.tasks.universal.student_ppo_env import StudentPPOEnvCfg


def traces():
    targets = torch.tensor([target for _, _, target in SCENARIOS])
    actual = targets.repeat_interleave(4, dim=0).unsqueeze(0).repeat(5, 1, 1)
    return actual, torch.ones(5, 80, dtype=torch.bool), torch.ones(80, dtype=torch.bool)


def test_exact_signed_tracking_passes_all_cases_and_has_zero_error():
    velocity, ready, finite = traces()
    metrics = summarize_student_traces(velocity, ready, finite)
    assert metrics['validation_passed'] == 1.
    for name, _, _ in SCENARIOS:
        assert metrics[name + '_success'] == 1.
        for axis in ('vx', 'vy', 'wz'):
            assert metrics[name + '_rmse_' + axis] == 0.
    assert metrics['biped_moving_success'] == metrics['snake_moving_success'] == 1.


def test_standing_still_cannot_pass_nonzero_or_opposite_combined_commands():
    velocity, ready, finite = traces()
    metrics = summarize_student_traces(torch.zeros_like(velocity), ready, finite)
    for mode in ('biped', 'snake'):
        assert metrics[mode + '_stop_success'] == 1.
        assert metrics[mode + '_moving_success'] == 0.
        assert metrics[mode + '_worst_mixed_success'] == 0.
        for sign in ('positive', 'negative'):
            name = mode + '_mixed_' + sign
            assert metrics[name + '_rmse_vx'] == pytest.approx(.08)
            assert metrics[name + '_rmse_vy'] == pytest.approx(.05)
            assert metrics[name + '_rmse_wz'] == pytest.approx(.3)
            assert metrics[name + '_moving_fraction'] == 0.


def test_readiness_and_fall_failures_are_not_hidden_by_exact_velocity():
    velocity, ready, finite = traces()
    ready[:, :4] = False
    no_fall = finite.clone()
    no_fall[4:8] = False
    metrics = summarize_student_traces(velocity, ready, finite, no_fall=no_fall)
    assert metrics['biped_stop_success'] == metrics['biped_forward_success'] == 0.
    assert metrics['biped_backward_success'] == 1.


def test_periodic_validation_counts_policy_updates_after_warmup(monkeypatch):
    def upstream(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.device = env, cfg, device
        self.logger = SimpleNamespace(writer=None)
        self.alg = SimpleNamespace(completed_updates=0, critic_warmup_iterations=2,
                                   process_env_step=lambda *args: None)
        def update():
            self.alg.completed_updates += 1
            return {}
        self.alg.update = update
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', upstream)
    calls = []
    monkeypatch.setattr(UniversalStudentPPORunner, '_run_validation',
                        lambda self, **kwargs: calls.append(self.alg.completed_updates) or {})
    env: Any = SimpleNamespace(num_envs=4, step_dt=.02,
                               cfg=SimpleNamespace(terminations={}))
    env.unwrapped = env
    runner = UniversalStudentPPORunner(env, {'resume': True, 'console_mode': 'full',
                                            'validation_interval_iterations': 2}, log_dir='logs')
    for _ in range(6):
        runner.alg.update()
    assert calls == [4, 6]
    assert env._student_rl_iteration == 4


def test_validation_report_names_the_exact_saved_policy_and_restores_counter(monkeypatch, tmp_path):
    values = {'biped_moving_success': .25, 'snake_moving_success': .5,
              'descent_handoff_fraction': 1., 'ascent_handoff_fraction': .75}
    class Evaluator:
        def __init__(self, **kwargs):
            self.rl_iteration = -1
        def __call__(self, actor):
            assert self.rl_iteration == 5
            return values
    monkeypatch.setattr('zbot_rl_mjlab.student_ppo_evaluation.StudentPPOEvaluator', Evaluator)
    runner: Any = UniversalStudentPPORunner.__new__(UniversalStudentPPORunner)
    runner.alg = SimpleNamespace(completed_updates=7, critic_warmup_iterations=2, actor=object())
    runner._student_validator = None
    runner._student_log_dir = str(tmp_path)
    runner.current_learning_iteration = 5
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=object()))
    runner.device, runner.cfg = 'cpu', {}
    saved = []
    runner.save = lambda path, infos: saved.append((path, infos, runner.current_learning_iteration))
    result = runner._run_validation()
    assert result['validation_biped_moving_success'] == .25
    assert runner.current_learning_iteration == 5
    assert saved[0][2] == 6
    assert saved[0][0].endswith('model_6.pt')
    report = json.loads((tmp_path / 'validation_policy_000007.json').read_text())
    assert report['checkpoint'] == saved[0][0]
    assert report['policy_updates'] == 5
    runner._run_validation(baseline=True)
    assert len(saved) == 1


@pytest.mark.parametrize('settings', [
    {'validation_interval_iterations': -1}, {'validation_interval_iterations': True},
    {'validation_num_envs': 64}, {'validation_num_envs': 0},
    {'validation_seconds': 1.}, {'validation_seconds': float('nan')},
])
def test_invalid_acceptance_settings_fail_before_environment_access(settings):
    with pytest.raises(ValueError):
        UniversalStudentPPORunner(cast(Any, None), settings)


def test_v2_registration_enables_student_only_handoff_rewards_and_acceptance():
    task = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student'
    old = load_env_cfg(task)
    for play in (False, True):
        cfg = cast(StudentPPOEnvCfg, load_env_cfg(task + '-PPO', play=play))
        assert cfg.play_transition_sync and cfg.student_ppo_completion_gate
        assert cfg.completion_enter_tolerance == .15 and cfg.completion_exit_tolerance == .20
        assert 'transition_settle_action' in cfg.rewards
        assert 'transition_ground_impact' in cfg.rewards
        assert 'descent_ground_impact' not in cfg.rewards
    assert not getattr(old, 'student_ppo_completion_gate', False)
    assert 'transition_settle_action' not in old.rewards
    agent = cast(StudentPPORunnerCfg, load_rl_cfg(task + '-PPO'))
    assert agent.validation_interval_iterations == agent.save_interval == 25
    assert agent.run_name == 'student150_ppo_v2'


@pytest.fixture
def selection_runner(monkeypatch, tmp_path):
    """Exercise runner validation/save/load plumbing without a physics rollout."""
    values = dict.fromkeys(REQUIRED_METRICS, .5)
    events = []

    def upstream(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.device = env, cfg, device
        self.current_learning_iteration = 0
        self.logger = SimpleNamespace(writer=SimpleNamespace(
            flush=lambda: events.append('flush'), close=lambda: events.append('writer_close')))
        self.alg = SimpleNamespace(completed_updates=0, critic_warmup_iterations=2,
                                   actor=object(), process_env_step=lambda *args: None,
                                   validate_resume=lambda saved: events.append('validate_resume'))

        def update():
            self.alg.completed_updates += 1
            return {}

        self.alg.update = update

    def save(self, path, infos=None):
        torch.save({'iter': self.current_learning_iteration, 'infos': infos,
                    'completed_updates': self.alg.completed_updates}, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        events.append('upstream_load')
        saved = torch.load(path, weights_only=False)
        self.alg.completed_updates = saved['completed_updates']
        self.current_learning_iteration = saved['iter']
        return saved['infos']

    class Evaluator:
        def __init__(self, **kwargs):
            self.rl_iteration = -1

        def __call__(self, actor):
            return values.copy()

        def close(self):
            events.append('validator_close')

    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', upstream)
    monkeypatch.setattr(MjlabOnPolicyRunner, 'save', save)
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', load)
    monkeypatch.setattr('zbot_rl_mjlab.student_ppo_evaluation.StudentPPOEvaluator', Evaluator)
    env: Any = SimpleNamespace(num_envs=4, step_dt=.02,
                               cfg=SimpleNamespace(terminations={}, student_history_length=1),
                               reset=lambda: events.append('reset'))
    env.unwrapped = env
    runner: Any = UniversalStudentPPORunner(env, {
        'resume': True, 'console_mode': 'full', 'validation_interval_iterations': 1,
        'validation_selection_enabled': True,
    }, log_dir=str(tmp_path))
    return runner, values, events


def test_selection_saves_baseline_and_improved_best_with_exact_iterations(selection_runner, tmp_path):
    runner, values, _ = selection_runner
    runner._run_validation(baseline=True)
    baseline = torch.load(tmp_path / 'model_baseline.pt', weights_only=False)
    assert baseline['iter'] == -1 and baseline['completed_updates'] == 0
    assert baseline['infos']['student_ppo_selection']['baseline'] == values
    assert torch.load(tmp_path / 'model_best.pt', weights_only=False)['iter'] == -1
    assert runner.current_learning_iteration == 0

    values['snake_moving_success'] = .75
    runner.alg.completed_updates, runner.current_learning_iteration = 5, 3
    runner._run_validation()
    best = torch.load(tmp_path / 'model_best.pt', weights_only=False)
    candidate = torch.load(tmp_path / 'model_4.pt', weights_only=False)
    assert best['iter'] == candidate['iter'] == 4
    assert best['completed_updates'] == candidate['completed_updates'] == 5
    assert runner.current_learning_iteration == 3
    report = json.loads((tmp_path / 'validation_best.json').read_text())
    assert report['source_checkpoint'] == str(tmp_path / 'model_4.pt')
    assert report['selection']['improved'] and report['metrics'] == values


def test_three_regressions_stop_learning_after_saving_rejected_checkpoint(selection_runner, monkeypatch, tmp_path):
    runner, values, events = selection_runner

    def learn(self, iterations, init_at_random_ep_len=False):
        values['ascent_handoff_fraction'] = 0.
        for iteration in range(iterations):
            self.current_learning_iteration = iteration
            self.alg.update()

    monkeypatch.setattr(MjlabOnPolicyRunner, 'learn', learn)
    runner.learn(20)
    assert runner.alg.completed_updates == 5  # Two warmup, then three failed validations.
    assert runner.current_learning_iteration == 4
    assert (tmp_path / 'model_4.pt').is_file()
    report = json.loads((tmp_path / 'validation_policy_000005.json').read_text())
    assert report['selection']['should_stop']
    assert not report['selection']['eligible']
    assert report['checkpoint'] == str(tmp_path / 'model_4.pt')
    assert torch.load(tmp_path / 'model_best.pt', weights_only=False)['iter'] == -1
    assert events[-3:] == ['validator_close', 'flush', 'writer_close']
    assert runner._student_validator is None


def test_runner_resume_restores_selection_before_next_validation(selection_runner, tmp_path):
    runner, values, events = selection_runner
    runner._run_validation(baseline=True)
    baseline = values.copy()
    values['snake_moving_success'] = .75
    runner.alg.completed_updates = 3
    runner._run_validation()
    values['ascent_handoff_fraction'] = 0.
    for update in (4, 5):
        runner.alg.completed_updates = update
        runner._run_validation()
    expected = runner._validation_selector.state_dict()
    runner._validation_selector = None
    runner.alg.completed_updates = 0
    runner.load(tmp_path / 'model_4.pt')
    selector = cast(StudentPPOCheckpointSelector, runner._validation_selector)
    assert events == ['validate_resume', 'upstream_load', 'reset']
    assert selector.state_dict() == expected
    assert runner.current_learning_iteration == runner.alg.completed_updates == 5
    assert runner.env._student_rl_iteration == 3
    runner._run_validation(baseline=True)
    assert selector.baseline == baseline
    assert selector.consecutive_failures == 2
    runner.alg.completed_updates += 1
    runner._run_validation()
    assert runner._validation_stop_reason
    assert selector.consecutive_failures == 3


def test_resume_into_new_directory_preserves_historical_best_artifact(selection_runner, tmp_path):
    runner, values, _ = selection_runner
    runner._run_validation(baseline=True)
    values['snake_moving_success'] = .75
    runner.alg.completed_updates = 3
    runner._run_validation()
    source = tmp_path / 'model_2.pt'
    original_best = source.read_bytes()
    historical_metrics = values.copy()
    values['ascent_handoff_fraction'] = 0.
    runner.alg.completed_updates = 4
    runner._run_validation()
    resume_path = tmp_path / 'model_3.pt'
    original_resume = resume_path.read_bytes()
    state = runner._validation_selector.state_dict()
    saved = torch.load(resume_path, weights_only=False)
    assert saved['infos']['student_ppo_best_checkpoint'] == str(source.resolve())

    destination = tmp_path / 'resumed'
    destination.mkdir()
    runner._student_log_dir = str(destination)
    runner._validation_selector = None
    runner.load(resume_path)
    selector = cast(StudentPPOCheckpointSelector, runner._validation_selector)
    assert selector.state_dict() == state
    assert selector.consecutive_failures == 1
    assert (destination / 'model_best.pt').read_bytes() == original_best
    assert source.read_bytes() == original_best
    assert resume_path.read_bytes() == original_resume
    report = json.loads((destination / 'validation_best.json').read_text())
    assert report['resumed'] and report['available']
    assert report['source_checkpoint'] == str(source.resolve())
    assert report['checkpoint'] == str(destination / 'model_best.pt')
    assert report['metrics'] == historical_metrics


@pytest.mark.parametrize('invalid_state', ['missing', 'settings', 'best'])
def test_runner_rejects_invalid_selection_resume_before_upstream_mutation(
        selection_runner, tmp_path, invalid_state):
    runner, _, events = selection_runner
    runner._run_validation(baseline=True)
    path = tmp_path / 'model_baseline.pt'
    saved = torch.load(path, weights_only=False)
    if invalid_state == 'missing':
        del saved['infos']['student_ppo_selection']
    elif invalid_state == 'settings':
        saved['infos']['student_ppo_selection']['patience'] = 4
    else:
        saved['infos']['student_ppo_selection']['best_metrics']['ascent_handoff_fraction'] = 0.
    torch.save(saved, path)
    with pytest.raises(ValueError):
        runner.load(path)
    assert events == ['validate_resume']
    assert runner.alg.completed_updates == 0


def test_disabled_selection_preserves_v2_baseline_and_checkpoint_behavior(selection_runner, tmp_path):
    runner, values, _ = selection_runner
    runner.cfg['validation_selection_enabled'] = False
    runner._run_validation(baseline=True)
    assert not (tmp_path / 'model_baseline.pt').exists()
    assert not (tmp_path / 'model_best.pt').exists()
    values['ascent_handoff_fraction'] = 0.
    for update in (3, 4, 5):
        runner.alg.completed_updates = update
        runner._run_validation()
    assert runner._validation_selector is None
    assert not getattr(runner, '_validation_stop_reason', None)
    saved = torch.load(tmp_path / 'model_4.pt', weights_only=False)
    assert 'student_ppo_selection' not in saved['infos']
    assert not (tmp_path / 'model_best.pt').exists()


@pytest.mark.parametrize('invalid', ['missing', float('nan'), float('inf')])
def test_invalid_candidate_metrics_save_rejection_without_replacing_best(selection_runner, tmp_path, invalid):
    runner, values, _ = selection_runner
    runner._run_validation(baseline=True)
    key = 'ascent_handoff_fraction'
    if invalid == 'missing':
        del values[key]
    else:
        values[key] = invalid
    for update in (3, 4, 5):
        runner.alg.completed_updates = update
        result = runner._run_validation()
        assert 'validation_' + key not in result
    assert runner._validation_stop_reason
    saved = torch.load(tmp_path / 'model_4.pt', weights_only=False)
    assert saved['infos']['student_validation'].get(key) is None
    report = json.loads((tmp_path / 'validation_policy_000005.json').read_text())
    assert report['metrics'].get(key) is None
    assert report['selection']['should_stop']
    assert not report['selection']['eligible'] and not report['selection']['improved']
    assert report['selection']['score'] is None
    assert torch.load(tmp_path / 'model_best.pt', weights_only=False)['iter'] == -1
