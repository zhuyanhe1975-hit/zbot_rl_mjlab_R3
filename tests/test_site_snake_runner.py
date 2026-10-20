"""A world-X policy cannot enter the new qualification and snapshot contract."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from zbot_rl_mjlab import site_snake_evaluation
from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner, site_snake_ppo_cfg
from zbot_rl_mjlab.tasks.snake.site_omni import SiteSnakeCommandCfg

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'


@pytest.fixture
def setup(monkeypatch):
    obs = TensorDict({'actor': torch.zeros(2, 56)}, batch_size=[2])
    groups = {'actor': ['actor'], 'critic': ['actor']}
    actor = MLPModel(obs, groups, 'actor', 6, hidden_dims=(8,), obs_normalization=True,
                     distribution_cfg={'class_name': 'BetaDistribution', 'action_range': (-1., 1.)})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=(8,), obs_normalization=True)
    alg = SimpleNamespace(actor=actor, critic=critic, update=Mock(return_value={'value': 1.}))
    term = SimpleNamespace(tracking_ema=torch.zeros(2))
    physical = SimpleNamespace(cfg=load_env_cfg(TASK), common_step_counter=0,
                               command_manager=SimpleNamespace(get_term=lambda _: term))
    env = SimpleNamespace(unwrapped=physical, num_actions=6, get_observations=lambda: obs)
    logger = SimpleNamespace(writer=None, process_env_step=Mock(), log=Mock())

    def base_init(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.alg, self.logger = env, cfg, alg, logger
        self.current_learning_iteration = 0

    evaluator = Mock(return_value={'snake_omni_passed': 0., 'snake_omni_scenarios_passed': 0.})
    evaluator.last_report = {'passed': False, 'duration_eligible': True, 'protocol': 'site_snake_omni_v3'}
    factory = Mock(return_value=evaluator)
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', base_init)
    monkeypatch.setattr(site_snake_evaluation, 'SiteSnakeEvaluator', factory)
    cfg = asdict(site_snake_ppo_cfg())
    cfg['console_mode'] = 'full'
    cfg['validation_interval_iterations'] = 1
    return env, cfg, alg, evaluator, factory


def checkpoint(path, runner):
    saved = {'infos': {'privileged_omni_teacher': runner._schema() | {'completed_ppo_iterations': 17},
                       'universal_course': [.2, .6]},
             'actor_state_dict': runner.alg.actor.state_dict(), 'critic_state_dict': runner.alg.critic.state_dict()}
    torch.save(saved, path)
    return saved


def test_recipe_uses_successful_archived_snake_ppo_parameters():
    cfg = site_snake_ppo_cfg()
    assert cfg.algorithm.learning_rate == .001 and cfg.algorithm.entropy_coef == .002
    assert cfg.actor.hidden_dims == cfg.critic.hidden_dims == (256, 128)
    assert cfg.actor.obs_normalization and cfg.critic.obs_normalization
    assert cfg.algorithm.class_name == 'PPO'
    assert cfg.console_mode == 'full'
    assert cfg.validation_interval_iterations == 0
    assert cfg.robust_validation_interval == 0


@pytest.mark.parametrize('change', [
    {'role': 'privileged_omni_guided_teacher'}, {'actor_obs_dim': 51},
    {'snake_omni_version': 1}, {'snake_omni_version': 2}, {'tracking_point': 'whole_robot_com'},
    {'reward_recipe': 'archived_progress_v1'}, {'reward_recipe': 'zbot_snake_velocity_v3'},
    {'reward_recipe': 'zbot_snake_velocity_limits_v4'},
    {'objective': 'snake_world_x_capped_speed'}, {'command_axes': ('world_x', 'world_y', 'world_z')},
    {'joint_target_limit': 3.14}, {'max_target_speed': 1.},
])
def test_wrong_checkpoint_rejected_before_mutation(setup, monkeypatch, tmp_path, change):
    env, cfg, _, _, _ = setup
    runner = SiteSnakeTeacherRunner(env, cfg)
    path = tmp_path / 'wrong.pt'
    saved = checkpoint(path, runner)
    saved['infos']['privileged_omni_teacher'].update(change)
    torch.save(saved, path)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', load)
    with pytest.raises(ValueError):
        runner.load(path, {'actor': True})
    load.assert_not_called()


def test_playback_prepares_only_matching_site_snake_env(setup, tmp_path):
    env, cfg, *_ = setup
    runner = SiteSnakeTeacherRunner(env, cfg)
    path = tmp_path / 'model.pt'
    checkpoint(path, runner)
    play_cfg = load_env_cfg(TASK, play=True)
    SiteSnakeTeacherRunner.prepare_playback_env_cfg(play_cfg, path)
    command = play_cfg.commands['transform']
    assert isinstance(command, SiteSnakeCommandCfg) and command.mode == 'snake'
    with pytest.raises(ValueError):
        SiteSnakeTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(), path)


def test_architecture_failure_is_preflighted(setup, monkeypatch, tmp_path):
    env, cfg, *_ = setup
    runner = SiteSnakeTeacherRunner(env, cfg)
    path = tmp_path / 'model.pt'
    saved = checkpoint(path, runner)
    key = next(iter(saved['actor_state_dict']))
    saved['actor_state_dict'][key] = torch.zeros(123)
    torch.save(saved, path)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', load)
    with pytest.raises(ValueError, match='architecture'):
        runner.load(path, {'actor': True})
    load.assert_not_called()


@pytest.mark.parametrize('passed,eligible', [(False, True), (True, False), (True, True)])
def test_only_actual_eligible_pass_writes_qualified_snapshot(setup, tmp_path, monkeypatch, passed, eligible):
    env, cfg, alg, evaluator, factory = setup
    evaluator.return_value = {'snake_omni_passed': float(passed), 'snake_omni_scenarios_passed': 9. if passed else 0.}
    evaluator.last_report.update(passed=passed, duration_eligible=eligible)

    def save(self, path, infos=None):
        torch.save({'infos': infos, 'actor_state_dict': deepcopy(self.alg.actor.state_dict())}, path)

    monkeypatch.setattr(MjlabOnPolicyRunner, 'save', save)
    runner = SiteSnakeTeacherRunner(env, cfg, str(tmp_path))
    result = runner.alg.update()
    factory.assert_called_once()
    evaluator.assert_called_once_with(alg.actor)
    assert runner.completed_ppo_iterations == 1
    latest = tmp_path / 'qualification' / 'latest.json'
    assert latest.is_file()
    snapshots = list((tmp_path / 'qualification').glob('model_*.pt'))
    assert len(snapshots) == int(passed and eligible)
    assert result['snake_omni_passed'] == float(passed and eligible)
    if snapshots:
        saved = torch.load(snapshots[0], weights_only=False)
        assert saved['infos']['snake_site_qualification']['passed']
        for name, tensor in alg.actor.state_dict().items():
            assert torch.equal(tensor, saved['actor_state_dict'][name])


def test_play_never_builds_evaluator(setup):
    env, cfg, _, _, factory = setup
    SiteSnakeTeacherRunner(env, cfg)
    factory.assert_not_called()


def test_default_training_skips_100_and_500_iteration_reports(setup, tmp_path):
    env, cfg, _, evaluator, factory = setup
    defaults = site_snake_ppo_cfg()
    cfg.update(validation_interval_iterations=defaults.validation_interval_iterations,
               robust_validation_interval=defaults.robust_validation_interval)
    runner = SiteSnakeTeacherRunner(env, cfg, str(tmp_path))
    for _ in range(501):
        losses = runner.alg.update()
        assert 'snake_omni_passed' not in losses
    assert runner.completed_ppo_iterations == 501
    factory.assert_not_called()
    evaluator.assert_not_called()
    assert not (tmp_path / 'qualification').exists()


def test_default_console_installs_reward_summary_and_live_sampling(setup, tmp_path):
    env, cfg, _, _, _ = setup
    cfg['console_mode'] = site_snake_ppo_cfg().console_mode
    runner = SiteSnakeTeacherRunner(env, cfg, str(tmp_path))
    assert hasattr(runner.logger.log, '__wrapped__')
    assert hasattr(runner.logger.process_env_step, '__wrapped__')
