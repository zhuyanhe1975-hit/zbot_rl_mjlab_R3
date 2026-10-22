"""A world-X policy cannot enter the new qualification and snapshot contract."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.models import MLPModel
from rsl_rl.modules import BetaDistribution
from tensordict import TensorDict

from zbot_rl_mjlab import site_snake_evaluation
from zbot_rl_mjlab.experiments.site_snake import site_snake_env_cfg, site_snake_ppo_cfg
from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake.site_omni import SiteSnakeCommandCfg


@pytest.fixture
def setup(monkeypatch):
    obs = TensorDict({'actor': torch.zeros(2, 61)}, batch_size=[2])
    groups = {'actor': ['actor'], 'critic': ['actor']}
    actor = MLPModel(obs, groups, 'actor', 6, hidden_dims=(8,), obs_normalization=True,
                     distribution_cfg=site_snake_ppo_cfg().actor.distribution_cfg)
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=(8,), obs_normalization=True)
    alg = SimpleNamespace(actor=actor, critic=critic, update=Mock(return_value={'value': 1.}))
    term = SimpleNamespace(tracking_ema=torch.zeros(2))
    physical = SimpleNamespace(cfg=site_snake_env_cfg(), common_step_counter=0,
                               command_manager=SimpleNamespace(get_term=lambda _: term))
    env = SimpleNamespace(unwrapped=physical, num_actions=6, get_observations=lambda: obs)
    logger = SimpleNamespace(writer=None, process_env_step=Mock(), log=Mock())

    def base_init(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.alg, self.logger = env, cfg, alg, logger
        self.current_learning_iteration = 0

    evaluator = Mock(return_value={'snake_omni_passed': 0., 'snake_omni_scenarios_passed': 0.})
    evaluator.last_report = {'passed': False, 'duration_eligible': True, 'protocol': 'site_snake_omni_v4'}
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


def test_recipe_uses_g1_velocity_gaussian_exploration():
    cfg = site_snake_ppo_cfg()
    assert cfg.algorithm.learning_rate == .001 and cfg.algorithm.entropy_coef == .01
    assert cfg.actor.distribution_cfg == {'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}
    assert cfg.clip_actions == 1.
    assert cfg.actor.hidden_dims == cfg.critic.hidden_dims == (256, 128)
    assert cfg.actor.obs_normalization and cfg.critic.obs_normalization
    assert cfg.algorithm.class_name == 'PPO'
    assert cfg.console_mode == 'full'
    assert cfg.validation_interval_iterations == 0
    assert cfg.robust_validation_interval == 0


@pytest.mark.parametrize('change', [
    {'reward_recipe': 'zbot_snake_head_frame_signed_alive_v13'},
    {'reward_recipe': 'zbot_snake_head_frame_absolute_error_v14'},
    {'distribution': 'BetaDistribution'}, {'schema_version': 5}, {'snake_omni_version': 11},
    {'action_clipping': 'none'},
    {'role': 'privileged_omni_guided_teacher'}, {'actor_obs_dim': 51},
    {'schema_version': 3}, {'actor_obs_dim': 56}, {'critic_obs_dim': 56},
    {'schema_version': 4}, {'actor_obs_dim': 63}, {'critic_obs_dim': 63},
    {'snake_omni_version': 9}, {'reward_recipe': 'zbot_snake_site_amplitude_weighted_twist_v11'},
    {'snake_omni_version': 10}, {'reward_recipe': 'zbot_snake_head_frame_error_alive_v12'},
    {'snake_omni_version': 7}, {'observation_recipe': 'legacy_scalar'},
    {'snake_omni_version': 8}, {'reward_recipe': 'zbot_snake_site_signed_twist_reset_v10'},
    {'snake_omni_version': 1}, {'snake_omni_version': 2}, {'tracking_point': 'whole_robot_com'},
    {'reward_recipe': 'archived_progress_v1'}, {'reward_recipe': 'zbot_snake_velocity_v3'},
    {'reward_recipe': 'zbot_snake_velocity_limits_v4'},
    {'snake_omni_version': 3}, {'reward_recipe': 'zbot_snake_velocity_limits_tilt_v5'},
    {'snake_omni_version': 4}, {'reward_recipe': 'zbot_snake_commanded_heading_v6'},
    {'reward_recipe': 'zbot_snake_site_net_progress_v7'},
    {'snake_omni_version': 5}, {'reward_recipe': 'zbot_snake_site_net_heading_v8'},
    {'snake_omni_version': 6}, {'reward_recipe': 'zbot_snake_site_signed_twist_v9'},
    {'velocity_limits': (0., .2, .2)}, {'velocity_limits': (.2, float('nan'), .2)},
    {'objective': 'snake_world_x_capped_speed'}, {'command_axes': ('world_x', 'world_y', 'world_z')},
    {'joint_target_limit': 0.}, {'joint_target_limit': float('nan')}, {'max_target_speed': -1.},
    {'snake_target_limit': 0.}, {'max_target_speed': float('inf')},
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
    env.unwrapped.cfg.commands['transform'].velocity_limits = (.12, .15, .25)
    runner = SiteSnakeTeacherRunner(env, cfg)
    path = tmp_path / 'model.pt'
    checkpoint(path, runner)
    play_cfg = site_snake_env_cfg(play=True)
    SiteSnakeTeacherRunner.prepare_playback_env_cfg(play_cfg, path)
    command = play_cfg.commands['transform']
    assert isinstance(command, SiteSnakeCommandCfg) and command.mode == 'snake'
    assert command.velocity_limits == (.12, .15, .25)
    assert play_cfg.rewards['track_forward_velocity'].params == {}
    assert play_cfg.rewards['track_lateral_velocity'].params == {}
    assert play_cfg.rewards['track_angular_velocity'].params == {}
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


def test_gaussian_head_has_six_means_unit_std_and_finite_unclipped_log_probs(setup):
    _, _, alg, *_ = setup
    actor = alg.actor
    obs = TensorDict({'actor': torch.zeros(4096, 61)}, batch_size=[4096])
    torch.manual_seed(7)
    samples = actor(obs, stochastic_output=True)
    assert samples.shape == (4096, 6)
    assert (samples.abs() > 1.).any()
    torch.testing.assert_close(actor.distribution.std, torch.ones_like(samples))
    assert actor.mlp(torch.zeros(2, 61)).shape == (2, 6)
    log_prob = actor.distribution.log_prob(samples)
    assert torch.isfinite(log_prob).all()
    (-log_prob.mean()).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in actor.parameters())


def test_runner_rejects_beta_actor(setup):
    env, cfg, alg, *_ = setup
    alg.actor.distribution = BetaDistribution(6)
    with pytest.raises(TypeError, match='Gaussian'):
        SiteSnakeTeacherRunner(env, cfg)


def test_custom_control_limits_save_and_restore_without_fixed_contract(setup, tmp_path):
    env, cfg, *_ = setup
    action = env.unwrapped.cfg.actions['joint_pos']
    action.target_limit = 2*torch.pi/3
    action.snake_target_limit = torch.pi/2
    action.max_target_speed = 3.
    runner = SiteSnakeTeacherRunner(env, cfg)
    path = tmp_path/'custom.pt'
    saved = checkpoint(path, runner)
    SiteSnakeTeacherRunner._validate_checkpoint(saved)
    play_cfg = site_snake_env_cfg(play=True)
    SiteSnakeTeacherRunner.prepare_playback_env_cfg(play_cfg, path)
    restored = play_cfg.actions['joint_pos']
    assert restored.target_limit == action.target_limit
    assert restored.snake_target_limit == action.snake_target_limit
    assert restored.max_target_speed == action.max_target_speed
