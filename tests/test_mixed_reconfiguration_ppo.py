"""Guidance source, task isolation and checkpoint mode for full reconfiguration."""
import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.mixed_reconfiguration_ppo import MixedReconfigurationRunner
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_guide import planned_reference
from zbot_rl_mjlab.tasks.reconfiguration.mixed_ppo import MODE

TASK = 'Mjlab-Zbot-6dof-Reconfigure-Mixed-270-270-180-90-90-PPO'


def test_guide_contains_every_motion_segment_and_keeps_connections():
    segments, report = planned_reference(1.)
    assert set(segments) == {(i,p) for i in (0,1,3,4) for p in (1,2,3)}
    assert all(v.shape == (21,6) and np.isfinite(v).all() for v in segments.values())
    for i in (0,1,3,4):
        np.testing.assert_allclose(segments[i,1][-1], segments[i,2][0])
        np.testing.assert_allclose(segments[i,2][-1], segments[i,3][0])
    assert report['collision_free'] is False


def test_independent_task_masked_algorithm_and_safety_config():
    cfg = load_env_cfg(TASK)
    assert cfg.mixed_reconfiguration_ppo and cfg.scene.num_envs == 4096
    assert cfg.auto_reset and cfg.connection_order == (0,1,3,4)
    assert cfg.break_force == 80 and cfg.minimum_other_margin == .2
    assert cfg.continuous_torque_limit == 14 and cfg.torque_limit == 20
    assert cfg.motion_duration_s == 1.
    assert {'success', 'failure', 'time_out'} == set(cfg.terminations)
    assert all(k.startswith('penalty_') for k,v in cfg.rewards.items() if v.weight < 0.)
    agent = load_rl_cfg(TASK)
    assert agent.algorithm.class_name.endswith(':MotionMaskedPPO')
    assert load_runner_cls(TASK) is MixedReconfigurationRunner


def test_old_loading_and_unguided_checkpoints_rejected(tmp_path):
    cfg = load_env_cfg(TASK, play=True)
    for mode in (None, 'mixed_closed_motion_residual_native_pd_v1', 'mixed_closed_kinematic_guided_motion_residual_native_pd_v2', 'mixed_closed_batched_gpu_kinematic_guided_motion_residual_v3'):
        path = tmp_path/'old.pt'
        torch.save({'infos': {'reconfiguration_control_mode': mode}},path)
        with pytest.raises(ValueError):
            MixedReconfigurationRunner.prepare_playback_env_cfg(cfg,path)
    path = tmp_path/'new.pt'
    torch.save({'infos': {'reconfiguration_control_mode': MODE, 'observation_recipe': 'full_state_guide_velocity_v1'}},path)
    MixedReconfigurationRunner.prepare_playback_env_cfg(cfg,path)
