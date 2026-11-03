from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.initial_closure_logging import (
    EPISODE_FIELDS,InitialClosureLogger,aggregate_episode_logs,
)


def episode_records():
    return [
        {'Metrics/ended_episodes':1,'Metrics/success_episodes':0,
         'Metrics/success_fraction':0.,'Metrics/failure_fraction':1.,
         'Metrics/load_guard_fraction':1.,'Metrics/non_target_batch_peak_N':84.,
         'Metrics/headtail_collision_fraction':1.,'Metrics/headtail_penetration_batch_peak_mm':1.,
         'Metrics/episode_duration_s':2.,'Penalty/failure':-20.,'Reward/time':0.,
         'Penalty/elapsed_time':-.4,'Reward/episode_return':-20.4},
        {'Metrics/ended_episodes':9,'Metrics/success_episodes':9,
         'Metrics/success_fraction':1.,'Metrics/failure_fraction':0.,
         'Metrics/load_guard_fraction':0.,'Metrics/non_target_batch_peak_N':65.,
         'Metrics/headtail_collision_fraction':0.,'Metrics/headtail_penetration_batch_peak_mm':.05,
         'Metrics/episode_duration_s':5.,'Metrics/success_duration_s':5.,
         'Penalty/failure':0.,'Reward/time':4.,'Penalty/elapsed_time':-1.,'Reward/episode_return':3.},
    ]


def test_episode_statistics_weight_completed_worlds_and_preserve_peak():
    result=aggregate_episode_logs(episode_records())
    assert result['Metrics/ended_episodes']==10
    assert result['Metrics/success_fraction']==pytest.approx(.9)
    assert result['Metrics/failure_fraction']==pytest.approx(.1)
    assert result['Metrics/non_target_peak_N']==84.
    assert result['Metrics/headtail_collision_fraction']==pytest.approx(.1)
    assert result['Metrics/headtail_penetration_peak_mm']==1.
    assert result['Metrics/episode_duration_s']==pytest.approx(4.7)
    assert result['Metrics/success_duration_s']==5.
    assert result['Penalty/failure']==pytest.approx(-2.)
    assert result['Penalty/elapsed_time']==pytest.approx(-.94)
    assert result['Reward/episode_return']==pytest.approx(.66)


def test_console_schema_stays_fixed_with_and_without_completions(capsys):
    logger=InitialClosureLogger(log_dir=None,cfg={'algorithm':{},'num_steps_per_env':24},
        env_cfg=None,num_envs=2,is_distributed=False,gpu_world_size=1,gpu_global_rank=0,
        device='cpu',step_dt=.02)
    written=[]
    logger.writer=SimpleNamespace(add_scalar=lambda *args:written.append(args))
    args=dict(start_it=0,total_it=2,collect_time=1.,learn_time=.1,
              loss_dict={'value':.1,'surrogate':.2,'entropy':.3},learning_rate=.001,
              action_std=torch.full((7,),.3))
    logger.log(it=0,**args)
    empty=capsys.readouterr().out
    for key in EPISODE_FIELDS:assert f'{key:>42}:' in empty
    assert 'Reward/time: n/a' in empty
    assert not any(k=='Reward/time' for k,_,_ in written)
    logger.ep_extras=episode_records()
    logger.rewbuffer.append(-5.);logger.lenbuffer.append(250)
    logger.log(it=1,**args)
    completed=capsys.readouterr().out
    for key in EPISODE_FIELDS:assert f'{key:>42}:' in completed
    assert 'Reward/episode_return: 0.6600' in completed
    assert 'Recent100/episode_return: -5.0000' in completed
    assert 'Mean reward:' not in completed
    assert logger.ep_extras==[]
    assert ('Metrics/success_fraction',pytest.approx(.9),1) in written
