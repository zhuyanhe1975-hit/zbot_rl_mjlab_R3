"""Compact console changes presentation only; the original writer still owns logging."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from rsl_rl.utils.logger import Logger

from zbot_rl_mjlab.training_console import install_compact_logging


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = (torch.as_tensor(value).item(), step)


def make_logger() -> Any:
    logger: Any = Logger(None, {"algorithm": {}, "num_steps_per_env": 24, "run_name": "four_teachers"},
                         {}, 2, False, 1, 0, "cpu")
    logger.writer = RecordingWriter()
    logger.rewbuffer.extend([1., 3.])
    logger.lenbuffer.extend([100., 200.])
    logger.ep_extras = [{"Episode_Reward/alive": 1., "hidden_metric": 42.}]
    return logger


def log(logger, losses=None):
    logger.log(it=3, start_it=0, total_it=10, collect_time=.125, learn_time=.075,
               loss_dict={} if losses is None else losses, learning_rate=.001,
               action_std=torch.tensor([.1, .3]), rnd_weight=None)


def test_metrics_and_state_match_unmodified_logger(capsys):
    baseline, compact = make_logger(), make_logger()
    losses = {"value": .7, "surrogate": -.01, "distillation": .001}
    log(baseline, losses)
    capsys.readouterr()
    install_compact_logging(compact)
    log(compact, losses)
    output = capsys.readouterr().out
    assert compact.writer.scalars == baseline.writer.scalars
    assert compact.tot_timesteps == baseline.tot_timesteps == 48
    assert compact.tot_time == baseline.tot_time == pytest.approx(.2)
    assert compact.ep_extras == baseline.ep_extras == []
    assert output == ("[four_teachers] 3/10 | 240 steps/s | collect 0.125s learn 0.075s\n"
                      "reward 2.00 | ep_len 150.0 | std 0.200\n")


def test_six_lines_include_phase_teacher_counts_gate_and_validation(capsys):
    logger = make_logger()
    install_compact_logging(logger)
    log(logger, {
        "distillation_warmup": 1., "distillation_coefficient": 1., "distillation": .01234,
        "teacher_snake_forward_candidates": 1234., "teacher_walking_candidates": 4321.,
        "teacher_snake_forward_validation_mse": .01,
        "warmup_blocked_snake_forward": 1., "warmup_blocked_walking": 0.,
        "warmup_blocked_behavior": 1., "warmup_validation_passes": 2.,
        "walking_success": .9, "snake_forward_success": .75,
    })
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 6
    assert "phase warmup | coef 1.0000 | imitation 0.0123" in output
    assert "teachers snake_forward=1234 walking=4321" in output
    assert "gate blocked=snake_forward,behavior | passes 2" in output
    assert "validation walking=90% snake_forward=75%" in output
    assert "validation_mse" not in output


def test_disabled_writer_remains_silent(capsys):
    logger = make_logger()
    logger.writer = None
    install_compact_logging(logger)
    log(logger)
    assert capsys.readouterr().out == ""
    assert logger.tot_timesteps == 0


def test_joint_guidance_counts_are_compact_without_phase_switch_messages(capsys):
    logger = make_logger()
    install_compact_logging(logger)
    log(logger, {'guidance': .12, 'guidance_descent_cached': 256.,
                 'guidance_snake_motion_cached': 64., 'guidance_ascent_cached': 256.})
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 3
    assert 'guide 0.1200 | labels down=256 snake=64 up=256' in output
    assert 'warmup' not in output and 'phase' not in output


def test_original_receives_identical_args_and_propagates_errors(capsys):
    loss, std = {}, torch.tensor([.1])
    received = []

    def failing_log(it, loss_dict, action_std, width=80):
        received.append((it, loss_dict, action_std))
        print("writer diagnostic")
        import warnings
        warnings.warn("writer warning", UserWarning, stacklevel=1)
        raise RuntimeError("writer failed")

    logger = SimpleNamespace(log=failing_log)
    install_compact_logging(logger)
    with (pytest.warns(UserWarning, match="writer warning"),
          pytest.raises(RuntimeError, match="writer failed")):
        logger.log(7, loss_dict=loss, action_std=std)
    assert received[0][0] == 7
    assert received[0][1] is loss
    assert received[0][2] is std
    assert capsys.readouterr().out == "writer diagnostic\n"


def test_custom_banner_width_and_writer_diagnostics(capsys):
    logger = make_logger()
    writer = logger.writer
    original = writer.add_scalar

    def write(*args):
        if args[0] == "Loss/learning_rate":
            print("writer upload complete")
        return original(*args)

    writer.add_scalar = write
    install_compact_logging(logger)
    logger.log(0, 0, 1, .1, .1, {"distillation_warmup": 0.}, .001,
               torch.tensor([.2]), None, width=60)
    output = capsys.readouterr().out
    assert output.startswith("writer upload complete\n[four_teachers]")
    assert "phase PPO" in output
    assert "###" not in output


def test_registered_student_defaults_to_compact_console():
    from mjlab.tasks.registry import load_rl_cfg

    from zbot_rl_mjlab.distillation import MultiTeacherRunnerCfg

    cfg = cast(MultiTeacherRunnerCfg, load_rl_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy'))
    assert cfg.console_mode == 'compact'


def test_invalid_console_mode_fails_before_creating_runner():
    from zbot_rl_mjlab.distillation import MultiTeacherRunner

    with pytest.raises(ValueError, match='console_mode'):
        MultiTeacherRunner(cast(Any, None), {'console_mode': 'silent'})


def test_omni_validation_collapses_all_heading_metrics(capsys):
    from zbot_rl_mjlab.universal_omni_evaluation import SCENARIOS

    logger = make_logger()
    install_compact_logging(logger)
    losses = {}
    for name, _, _ in SCENARIOS:
        losses[f'{name}_success'] = 1.
        for yaw in (0, 90, 180, 270):
            losses[f'{name}_yaw_{yaw}_success'] = 1.
    losses['biped_forward_success'] = .75
    losses['biped_forward_yaw_90_success'] = 0.
    log(logger, losses)
    output = capsys.readouterr().out
    assert 'validation biped=8/9 snake=9/9 transform=2/2' in output
    assert len(output.splitlines()) <= 6
    assert 'yaw_90' not in output
    assert logger.writer.scalars['Loss/biped_forward_yaw_90_success'] == (0., 3)


def test_site_snake_shows_live_rewards_without_completed_episodes(capsys):
    from zbot_rl_mjlab.training_console import install_site_snake_rollout_logging

    logger = make_logger()
    logger.rewbuffer.clear()
    logger.lenbuffer.clear()
    logger.ep_extras.clear()
    manager = SimpleNamespace(active_terms=['track_linear_velocity', 'track_angular_velocity'],
                              _step_reward=torch.tensor([[1., .2], [2., .4]]))
    env = SimpleNamespace(reward_manager=manager, step_dt=.02)
    install_compact_logging(logger)
    install_site_snake_rollout_logging(logger, env)
    with torch.inference_mode():
        for _ in range(2):
            logger.process_env_step(manager._step_reward.sum(-1)*env.step_dt, torch.zeros(2), {})
    original_losses = {'rollout_snake_vx_rmse': .01, 'rollout_snake_vy_rmse': .02, 'rollout_snake_wz_rmse': .3}
    log(logger, original_losses)
    output = capsys.readouterr().out
    assert 'reward_rate 1.800/s | episode_age 0.04s | resets 0 | timeouts 0 | policy_std 0.200' in output
    assert 'track_linear_velocity 1.500/s | track_angular_velocity 0.300/s' in output
    assert 'velocity_rmse vx 0.010 m/s | vy 0.020 m/s | wz 0.300 rad/s' in output
    assert '--' not in output
    assert logger.writer.scalars['Loss/rollout_reward_rate'] == pytest.approx((1.8, 3))
    assert 'rollout_reward_rate' not in original_losses
    assert not logger.rewbuffer and not logger.lenbuffer


def test_site_snake_rollout_stats_clear_after_each_log_and_count_collisions(capsys):
    from zbot_rl_mjlab.training_console import install_site_snake_rollout_logging

    logger = make_logger()
    logger.ep_extras.clear()
    manager = SimpleNamespace(active_terms=['track_linear_velocity', 'track_angular_velocity'],
                              _step_reward=torch.ones(2, 2))
    env = SimpleNamespace(reward_manager=manager, step_dt=.02)
    install_compact_logging(logger)
    install_site_snake_rollout_logging(logger, env)
    logger.process_env_step(torch.tensor([.04, .04]), torch.tensor([1, 0]), {})
    log(logger)
    assert 'resets 1' in capsys.readouterr().out
    assert len(logger.rewbuffer) == 3  # Two fixture episodes plus the real collision.
    manager._step_reward[:] = .25
    logger.process_env_step(torch.tensor([.01, .01]), torch.zeros(2), {})
    log(logger)
    output = capsys.readouterr().out
    assert 'reward_rate 0.500/s' in output and 'resets 0' in output
    assert 'track_linear_velocity 0.250/s | track_angular_velocity 0.250/s' in output


def test_reward_console_restores_training_and_episode_rows_without_changing_writer_metrics(capsys):
    from zbot_rl_mjlab.training_console import install_site_snake_rollout_logging

    logger = make_logger()
    logger.rewbuffer.clear()
    logger.lenbuffer.clear()
    logger.ep_extras.clear()
    manager = SimpleNamespace(active_terms=['track_linear_velocity', 'track_angular_velocity'],
                              _step_reward=torch.tensor([[1., .2], [2., .4]]))
    env = SimpleNamespace(reward_manager=manager, step_dt=.02)
    install_compact_logging(logger, rewards_only=True)
    install_site_snake_rollout_logging(logger, env)
    with torch.inference_mode():
        logger.process_env_step(manager._step_reward.sum(-1)*env.step_dt, torch.zeros(2), {})
    log(logger, {'value': .7, 'rollout_snake_vx_rmse': .01})
    output = capsys.readouterr().out
    assert 'Learning iteration 3/10' in output
    assert 'Mean reward rate (/s): 1.8000' in output
    assert 'Reward/track_linear_velocity (/s): 1.5000' in output
    assert 'Reward/track_angular_velocity (/s): 0.3000' in output
    assert 'Total steps: 48' in output
    assert 'Steps per second: 240' in output
    assert 'Mean action std: 0.200' in output
    assert 'Mean reward: --' in output
    assert 'Mean episode length: --' in output
    assert 'Current episode length (steps): 1.00' in output
    assert 'Current episode age (s): 0.02' in output
    assert 'Resets (rollout): 0' in output
    assert 'Terminations (no timeout): 0' in output
    assert 'Timeouts (rollout): 0' in output
    assert 'Iteration time: 0.20s' in output
    assert 'Time elapsed:' in output and 'ETA:' in output
    assert 'Mean value loss' not in output and 'rmse' not in output
    assert logger.writer.scalars['Loss/value'] == pytest.approx((.7, 3))
    assert logger.writer.scalars['Loss/rollout_snake_vx_rmse'] == pytest.approx((.01, 3))
    assert logger.writer.scalars['Loss/rollout_reward_rate'] == pytest.approx((1.8, 3))
    assert logger.tot_timesteps == 48
    assert logger.ep_extras == []


def test_reward_console_uses_native_completed_episode_average(capsys):
    logger = make_logger()
    baseline = make_logger()
    log(baseline)
    capsys.readouterr()
    install_compact_logging(logger, rewards_only=True)
    log(logger)
    output = capsys.readouterr().out
    assert 'Mean reward: 2.00' in output
    assert 'Mean episode length: 150.00' in output
    assert logger.writer.scalars == baseline.writer.scalars
    assert logger.tot_timesteps == baseline.tot_timesteps
    assert logger.lenbuffer == baseline.lenbuffer


@pytest.mark.parametrize('fallback_to_manager', [False, True])
def test_rollout_reset_and_timeout_counts_are_separate_and_clear_each_iteration(capsys, fallback_to_manager):
    from zbot_rl_mjlab.training_console import install_site_snake_rollout_logging

    logger = make_logger()
    logger.rewbuffer.clear()
    logger.lenbuffer.clear()
    logger.ep_extras.clear()
    manager = SimpleNamespace(active_terms=['tracking'], _step_reward=torch.ones(2, 1))
    termination = SimpleNamespace(time_outs=torch.tensor([False, True]))
    env = SimpleNamespace(reward_manager=manager, step_dt=.02, termination_manager=termination)
    install_compact_logging(logger, rewards_only=True)
    install_site_snake_rollout_logging(logger, env)
    # Ended robot 0 is a failure; a timeout flag on non-ended robot 1 must not count.
    extras = {} if fallback_to_manager else {'time_outs': termination.time_outs[:, None]}
    logger.process_env_step(torch.ones(2), torch.tensor([1, 0]), extras)
    termination.time_outs = torch.tensor([False, True])
    extras = {} if fallback_to_manager else {'time_outs': termination.time_outs}
    # Then both end: one failure, one timeout.
    logger.process_env_step(rewards=torch.ones(2), dones=torch.ones(2), extras=extras)
    log(logger)
    output = capsys.readouterr().out
    assert 'Resets (rollout): 3' in output
    assert 'Terminations (no timeout): 2' in output
    assert 'Timeouts (rollout): 1' in output
    assert 'Mean episode length: 1.33' in output
    for key, expected in (('reset', 3), ('termination', 2), ('timeout', 1)):
        assert logger.writer.scalars[f'Loss/rollout_{key}_count'] == pytest.approx((expected, 3))
    logger.process_env_step(torch.zeros(2), torch.zeros(2), {})
    log(logger)
    output = capsys.readouterr().out
    assert 'Resets (rollout): 0' in output
    assert 'Terminations (no timeout): 0' in output
    assert 'Timeouts (rollout): 0' in output


def test_reward_only_console_shows_negative_actual_and_target_limit_contributions(capsys):
    logger = make_logger()
    install_compact_logging(logger, rewards_only=True)
    log(logger, {'rollout_reward_rate': -.8, 'rollout_reward_track_linear_velocity': .4,
                 'rollout_reward_track_angular_velocity': .2, 'rollout_reward_joint_pos_limits': -.4,
                 'rollout_reward_joint_target_limits': -.5, 'rollout_reward_failure_reset': -.5})
    output = capsys.readouterr().out
    assert 'Mean reward rate (/s): -0.8000' in output
    assert 'Reward/joint_pos_limits (/s): -0.4000' in output
    assert 'Reward/joint_target_limits (/s): -0.5000' in output
    assert 'Reward/failure_reset (/s): -0.5000' in output


def test_live_reward_summary_includes_reset_penalty_even_without_resets(capsys):
    from zbot_rl_mjlab.experiments.site_snake import site_snake_env_cfg
    from zbot_rl_mjlab.training_console import install_site_snake_rollout_logging

    logger = make_logger()
    logger.rewbuffer.clear()
    logger.lenbuffer.clear()
    logger.ep_extras.clear()
    samples = {'alive': [1., 1.], 'track_forward_velocity': [-.1, -.5],
               'track_lateral_velocity': [-.3, -.9],
               'track_angular_velocity': [-.2, -.6], 'site_tilt': [-.03, -.07],
               'joint_pos_limits': [-.04, -.08], 'joint_target_limits': [-.05, 0.],
               'failure_reset': [0., 0.], 'action_rate': [-.1, -.2],
               'joint_pose_deviation': [-.02, -.06]}
    names = list(site_snake_env_cfg().rewards)
    manager = SimpleNamespace(active_terms=names,
                              _step_reward=torch.tensor([samples[name] for name in names]).T)
    env = SimpleNamespace(reward_manager=manager, step_dt=.02)
    install_compact_logging(logger, rewards_only=True)
    install_site_snake_rollout_logging(logger, env)
    with torch.inference_mode():
        logger.process_env_step(manager._step_reward.sum(-1)*env.step_dt, torch.zeros(2), {})
    log(logger)
    output = capsys.readouterr().out
    assert 'Mean reward rate (/s): -0.6250' in output
    for name, value in (('alive', 1.), ('track_forward_velocity', -.3), ('track_lateral_velocity', -.6), ('track_angular_velocity', -.4),
                        ('site_tilt', -.05), ('joint_pos_limits', -.06), ('joint_target_limits', -.025),
                        ('failure_reset', 0.), ('action_rate', -.15), ('joint_pose_deviation', -.04)):
        assert f'Reward/{name} (/s): {value:.4f}' in output
        assert logger.writer.scalars[f'Loss/rollout_reward_{name}'] == pytest.approx((value, 3))
    assert not logger.rewbuffer
    manager._step_reward.zero_()
    with torch.inference_mode():
        logger.process_env_step(torch.zeros(2), torch.zeros(2), {})
    log(logger)
    output = capsys.readouterr().out
    for name in manager.active_terms:
        assert f'Reward/{name} (/s): 0.0000' in output


def test_full_console_exposes_adaptive_weights_and_unweighted_score(capsys):
    logger = make_logger()
    install_compact_logging(logger, rewards_only=True)
    log(logger, {'velocity_weight_used_vy': 1., 'velocity_weight_next_vy': 1.05,
                 'velocity_score_ema_vy': -.4,
                 'velocity_reward_forward_mean': .4, 'velocity_reward_lateral_mean': .4,
                 'velocity_planar_reward_ratio_valid': 1., 'velocity_forward_to_lateral_reward_ratio': 1.,
                 'velocity_planar_target_limited': 0.,
                 'velocity_reward_planar_mean': .8, 'velocity_reward_yaw_mean': .2,
                 'velocity_reward_ratio_target': .25, 'velocity_reward_ratio_valid': 1.,
                 'velocity_yaw_to_planar_reward_ratio': .25, 'velocity_yaw_target_limited': 0.})
    output = capsys.readouterr().out
    assert 'Adaptive vy weight (used -> next):' in output
    assert '1.000 -> 1.050 | score EMA -0.400' in output
    assert 'Yaw/planar reward (actual / target):' in output and '0.250 / 0.250' in output
    assert 'Vx/vy reward (actual / target):' in output and '1.000 / 1.000' in output
    assert logger.writer.scalars['Loss/velocity_weight_next_vy'] == pytest.approx((1.05, 3))


def test_reward_ratio_console_handles_zero_planar_reward_without_inf(capsys):
    logger = make_logger()
    install_compact_logging(logger, rewards_only=True)
    log(logger, {'velocity_reward_planar_mean': 0., 'velocity_reward_yaw_mean': .2,
                 'velocity_reward_ratio_target': .25, 'velocity_reward_ratio_valid': 0.,
                 'velocity_yaw_to_planar_reward_ratio': 0., 'velocity_yaw_target_limited': 1.})
    output = capsys.readouterr().out
    assert '-- / 0.250 | weight bound active' in output


@pytest.mark.parametrize('stage,description', [(1, 'planar first'), (2, 'yaw learning')])
def test_console_reports_velocity_curriculum_progress(capsys, stage, description):
    logger = make_logger()
    install_compact_logging(logger, rewards_only=True)
    log(logger, {'velocity_learning_stage': float(stage),
                 'velocity_planar_positive_updates': 7., 'velocity_planar_regression_updates': 2.})
    output = capsys.readouterr().out
    assert f'{stage} ({description}) | positive 7/20 | regression 2/10' in output


@pytest.mark.parametrize('with_task_denominators', [False, True])
def test_dagger_console_separates_task_outcomes_and_preserves_tensorboard(capsys, with_task_denominators):
    baseline, compact = make_logger(), make_logger()
    losses = {
        'dagger_teacher_fraction': .375,
        'dagger_teacher_beta': .4,
        'dagger_optimizer_steps': 32.,
        'dagger_supervised_rows': 65536.,
        'dagger_anchor_samples': 16384.,
    }
    tasks = ('biped', 'descending', 'snake', 'ascending')
    for task_index, task in enumerate(tasks, start=1):
        losses[f'dagger_{task}_mse'] = task_index / 100
        prefix = f'task_{task}_'
        losses[prefix + 'success_count'] = float(task_index)
        losses[prefix + 'biped_fallen_count'] = float(task_index + 10)
        losses[prefix + 'unsafe_count'] = float(task_index + 20)
        losses[prefix + 'time_out_count'] = float(task_index + 30)
        if with_task_denominators:
            losses[prefix + 'reward_rate'] = task_index * .125
            losses[prefix + 'mean_episode_s'] = task_index * .5
    log(baseline, losses)
    capsys.readouterr()
    install_compact_logging(compact)
    log(compact, losses)
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 6
    assert ('DAgger teacher 38% (beta 0.40) | updates 32 | rows 65536 | anchor 16384'
            in output)
    for task_index, task in enumerate(tasks, start=1):
        rate = f'{task_index * .125:.3f}' if with_task_denominators else '--'
        duration = f'{task_index * .5:.2f}s' if with_task_denominators else '--'
        assert (f'{task}: mse {task_index / 100:.4f} | reward/s {rate} | episode {duration} | '
                f'success {task_index} | falls {task_index + 10} | unsafe {task_index + 20} | '
                f'timeout {task_index + 30}') in output
    # Global completed-episode averages remain in TensorBoard, not this task table.
    assert 'reward 2.00' not in output and 'ep_len 150.0' not in output
    assert 'Mean reward:' not in output and 'Mean episode length:' not in output
    assert compact.writer.scalars == baseline.writer.scalars
    assert compact.writer.scalars['Train/mean_reward'] == (2., 3)
    assert compact.writer.scalars['Loss/dagger_supervised_rows'] == (65536., 3)
    assert compact.tot_timesteps == baseline.tot_timesteps == 48
    assert compact.ep_extras == baseline.ep_extras == []
