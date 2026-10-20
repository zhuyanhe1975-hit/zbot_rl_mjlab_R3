"""终端显示全部奖励与少量诊断，完整指标仍写入框架日志。"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.utils.logger import Logger

from zbot_rl_mjlab.experiments.snake import CompactSnakeRunner


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars[name] = (torch.as_tensor(value).item(), step)


def make_runner(monkeypatch, writer, reward_names=('translation_tracking', 'yaw_tracking', 'head_tilt'), observation_reference='command'):
    cfg = {'algorithm': {}, 'num_steps_per_env': 24, 'run_name': 'console_check'}

    def init(self, env, train_cfg, log_dir=None, device='cpu'):
        self.logger = Logger(log_dir, train_cfg, {}, 2, False, 1, 0, device)
        self.logger.writer = writer

    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', init)
    wrapped = SimpleNamespace(unwrapped=SimpleNamespace(cfg=SimpleNamespace(
        rewards=dict.fromkeys(reward_names), observation_reference=observation_reference,
    )))
    return CompactSnakeRunner(cast(VecEnv, wrapped), cfg)


def log(logger):
    logger.log(it=3, start_it=0, total_it=10, collect_time=.1, learn_time=.2,
               loss_dict={'value': .7, 'surrogate': -.01}, learning_rate=.001,
               action_std=torch.ones(6) * .5, rnd_weight=None)


def test_console_shows_all_rewards_but_keeps_detailed_metrics_in_writer(monkeypatch, capsys):
    writer = RecordingWriter()
    logger = make_runner(monkeypatch, writer).logger
    logger.rewbuffer.extend([1., 3.])
    logger.lenbuffer.extend([100., 200.])
    logger.ep_extras = [{
        'Episode_Reward/translation_tracking': torch.tensor([1., 3.]),
        'Episode_Reward/yaw_tracking': .5,
        'Episode_Reward/head_tilt': -.125,
        'Episode_Reward/future_reward': -.0002,
        'Episode_Metrics/forward_velocity': .004,
        'Episode_Metrics/group_translation_fraction': .5,
        'Episode_Metrics/group_translation_actual_speed': .002,
        'Episode_Metrics/group_stop_fraction': 0.,
        'Episode_Metrics/group_stop_linear_reward': 0.,
        'Episode_Metrics/translation_curriculum_cap': .06,
        'Episode_Termination/head_tilt': 2.,
        'Episode_Metrics/previously_hidden': 17.,
    }]
    expected_keys = set(logger.ep_extras[0])
    log(logger)
    shown = capsys.readouterr().out
    assert 'Learning iteration 3/10' in shown
    assert 'Mean value loss:' in shown
    assert 'Mean episode length:' in shown
    for key in expected_keys:
        if key.startswith('Episode_Reward/'):
            assert f'{key}:' in shown
        else:
            assert f'{key}:' not in shown
    assert 'Episode_Reward/translation_tracking: 2.0000' in shown
    assert 'Episode_Reward/future_reward: -0.0002' in shown
    assert 'Episode_Metrics/group_stop_fraction:' not in shown
    assert '奖励分项：' not in shown
    assert '指令分组（' not in shown
    assert '纯平移 m/s：目标-- 实际0.0040 误差-- 课程上限0.0600' in shown
    assert '前0.0040' in shown
    assert '复位计数：头倾2.00' in shown
    assert len(shown.splitlines()) < 40
    assert expected_keys <= writer.scalars.keys()
    assert writer.scalars['Episode_Metrics/previously_hidden'] == (17., 3)
    assert writer.scalars['Episode_Reward/translation_tracking'] == (2., 3)
    assert writer.scalars['Train/mean_reward'] == (2., 3)
    assert writer.scalars['Loss/value'] == pytest.approx((.7, 3))
    assert logger.tot_timesteps == 48
    assert logger.tot_time == pytest.approx(.3)
    assert logger.ep_extras == []


def test_no_completed_episode_uses_framework_output_and_no_writer_stays_silent(monkeypatch, capsys):
    logger = make_runner(monkeypatch, RecordingWriter()).logger
    log(logger)
    shown = capsys.readouterr().out
    assert 'Learning iteration 3/10' in shown
    assert 'Mean action std:' in shown
    assert 'Episode_Reward/' not in shown
    assert 'Episode_Metrics/' not in shown
    logger.writer = None
    log(logger)
    assert capsys.readouterr().out == ''


def test_group_conditional_metrics_use_exposure_weighted_aggregation(monkeypatch, capsys):
    writer = RecordingWriter()
    logger = make_runner(monkeypatch, writer).logger
    logger.ep_extras = [
        {'Episode_Metrics/group_translation_fraction': .1,
         'Episode_Metrics/group_translation_linear_reward': .5,
         'Episode_Metrics/group_translation_actual_speed': .001,
         'Episode_Metrics/group_stop_fraction': 0.,
         'Episode_Metrics/group_stop_linear_reward': 0.},
        {'Episode_Metrics/group_translation_fraction': .9,
         'Episode_Metrics/group_translation_linear_reward': .9,
         'Episode_Metrics/group_translation_actual_speed': .027,
         'Episode_Metrics/group_stop_fraction': 0.,
         'Episode_Metrics/group_stop_linear_reward': 0.},
    ]
    log(logger)
    shown = capsys.readouterr().out
    assert '实际0.0280' in shown
    assert 'Episode_Metrics/group_translation_conditional_' not in shown
    assert 'Episode_Metrics/group_stop_conditional_' not in shown
    assert writer.scalars['Episode_Metrics/group_translation_linear_reward'][0] == pytest.approx(.7)
    assert writer.scalars['Episode_Metrics/group_translation_fraction'][0] == pytest.approx(.5)
    assert writer.scalars['Episode_Metrics/group_translation_conditional_linear_reward'][0] == pytest.approx(1.4)
    assert logger.ep_extras == []
    assert logger.tot_timesteps == 48


def test_group_conditional_metrics_include_every_tensor_sample(monkeypatch, capsys):
    writer = RecordingWriter()
    logger = make_runner(monkeypatch, writer).logger
    logger.ep_extras = [
        {'Episode_Metrics/group_mixed_fraction': torch.tensor([.2, .4]),
         'Episode_Metrics/group_mixed_linear_error': torch.tensor([.02, .08])},
        {'Episode_Metrics/group_mixed_fraction': .4,
         'Episode_Metrics/group_mixed_linear_error': .12},
    ]
    log(logger)
    assert writer.scalars['Episode_Metrics/group_mixed_conditional_linear_error'][0] == pytest.approx(.22)
    assert 'Episode_Metrics/group_mixed_conditional_' not in capsys.readouterr().out


def test_detailed_metrics_do_not_expand_console_output(monkeypatch, capsys):
    writer = RecordingWriter()
    logger = make_runner(monkeypatch, writer).logger
    logger.ep_extras = [{
        'Episode_Reward/translation_tracking': 1.,
        **{f'Episode_Metrics/detail_{i}': float(i) for i in range(100)},
    }]
    log(logger)
    shown = capsys.readouterr().out
    assert 'Episode_Reward/translation_tracking: 1.0000' in shown
    assert 'Episode_Metrics/detail_' not in shown
    assert len(shown.splitlines()) < 40
    assert writer.scalars['Episode_Metrics/detail_99'] == (99., 3)


def test_world_task_shows_signed_velocity_and_all_reward_parts(monkeypatch, capsys):
    writer = RecordingWriter()
    names = ('head_forward_speed', 'head_lateral_speed_cost', 'head_vertical_speed_cost', 'joint_pose_deviation', 'target_saturation_fraction', 'head_tilt_reset_cost', 'head_angular_velocity_cost', 'self_collision_reset_cost', 'action_rate', 'head_yaw_cost', 'alive', 'head_yaw_reset_cost', 'body_yaw_cost', 'body_yaw_reset_cost', 'body_yaw_integral_cost', 'head_yaw_integral_cost', 'joint_fold_cost')
    logger = make_runner(monkeypatch, writer, names, observation_reference='world_x').logger
    logger.ep_extras = [{
        **{f'Episode_Reward/{name}': value for name, value in zip(names, (.04, -.002, -.001, -.003, -.05, -.0167, -.002, -.0167, -.01, -.001, .02, -.1667, -.001, 0., -.001, -.001, -.002), strict=True)},
        'Episode_Metrics/world_velocity_x': .04,
        'Episode_Metrics/world_velocity_y': -.01,
        'Episode_Metrics/world_velocity_z': .002,
        'Episode_Metrics/head_angular_speed_rad_s': .3,
    }]
    log(logger)
    shown = capsys.readouterr().out
    for name in names:
        assert f'Episode_Reward/{name}:' in shown
    assert '世界速度 m/s：X 0.0400 Y -0.0100 Z 0.0020' in shown
    assert '课程' not in shown
    assert '纯平移' not in shown
    assert len(shown.splitlines()) < 40
