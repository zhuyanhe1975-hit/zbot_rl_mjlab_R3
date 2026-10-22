"""教师超参数显式对齐成功归档，生产训练不动态加载bests。"""

from pathlib import Path
from typing import cast

import pytest
import yaml
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.experiments.omni_teacher import omni_teacher_ppo_cfg
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunnerCfg
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg

ROOT = Path(__file__).resolve().parents[1]
TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'


def archive(name, file='agent.yaml'):
    return yaml.load((ROOT / 'bests' / name / file).read_text(), Loader=yaml.BaseLoader)


def test_ppo_matches_successful_privileged_and_bidirectional_transform_settings():
    cfg = omni_teacher_ppo_cfg()
    for name in ('transform_snake_to_biped_teacher_2026-10-17_iter200', 'transform_snake_biped_bidirectional_2026-10-17_iter599'):
        saved = archive(name)
        for key in ('learning_rate', 'entropy_coef', 'gamma', 'lam', 'clip_param', 'desired_kl',
                    'max_grad_norm', 'value_loss_coef', 'num_learning_epochs', 'num_mini_batches'):
            assert getattr(cfg.algorithm, key) == pytest.approx(float(saved['algorithm'][key])), (name, key)
        for key in ('schedule', 'optimizer', 'class_name'):
            assert getattr(cfg.algorithm, key) == saved['algorithm'][key]
        assert cfg.num_steps_per_env == int(saved['num_steps_per_env'])
        assert cfg.save_interval == int(saved['save_interval'])
        assert cfg.actor.activation == saved['actor']['activation']
        assert cfg.actor.distribution_cfg is not None
        assert cfg.actor.distribution_cfg['class_name'] == saved['actor']['distribution_cfg']['class_name']


def test_teacher_uses_archived_network_capacity_and_shared_privileged_schema():
    cfg = cast(OmniTeacherRunnerCfg, load_rl_cfg(TASK))
    assert cfg.actor.hidden_dims == cfg.critic.hidden_dims == (256, 128)
    saved = archive('omni_velocity_energy_2026-10-10_iter499')
    assert cfg.actor.hidden_dims == tuple(int(x) for x in saved['actor']['hidden_dims'])
    assert cfg.actor.obs_normalization and cfg.critic.obs_normalization
    assert cfg.obs_groups == {'actor': ('actor',), 'critic': ('actor',)}


def test_control_frequency_and_native_rate_match_archived_snake():
    cfg = load_env_cfg(TASK)
    saved = archive('snake_world_x_2026-10-19_iter1499', 'env.yaml')
    assert cfg.decimation == int(saved['decimation'])
    assert cfg.sim.mujoco.timestep == float(saved['sim']['mujoco']['timestep'])
    action = cast(SnakeCrawlJointTargetRateActionCfg, cfg.actions['joint_pos'])
    assert action.max_target_speed == float(saved['actions']['joint_pos']['max_target_speed'])
    assert action.snake_target_limit == float(saved['actions']['joint_pos']['target_limit'])


def test_teacher_initial_difficulty_uses_archived_pd_and_foot_randomization():
    cfg = load_env_cfg(TASK)
    assert not any(name in cfg.events for name in ('randomize_mass', 'randomize_gains', 'randomize_pd_gains'))
    event = cfg.events['randomize_foot_friction']
    assert event.params['asset_cfg'].geom_names == ('a1_geom', 'b6_geom')
    assert event.params['ranges'] == (.3, 1.2)
    impulse = cfg.events['base_disturbance'].params
    saved = archive('velocity_energy_disturbed_2026-10-10_iter499', 'env.yaml')['events']['base_disturbance']['params']
    for name in ('force_range', 'torque_range', 'duration_s', 'cooldown_s'):
        assert impulse[name] == tuple(float(value) for value in saved[name])
    assert impulse['disturbed_fraction'] == float(saved['disturbed_fraction'])
    assert impulse['pretraining_scale'] == 0.


def test_teacher_phase_and_snake_costs_are_wired_from_successful_recipes():
    cfg = load_env_cfg(TASK)
    snake = archive('snake_world_x_2026-10-19_iter1499', 'env.yaml')
    mapping = {'snake_vertical_speed': 'head_vertical_speed_cost',
               'snake_joint_deviation': 'joint_pose_deviation',
               'snake_target_saturation': 'target_saturation_fraction'}
    for current, old in mapping.items():
        assert cfg.rewards[current].weight == float(snake['rewards'][old]['weight'])
    command = cast(PrivilegedOmniCommandCfg, cfg.commands['transform'])
    transform = archive('transform_snake_biped_bidirectional_2026-10-17_iter599', 'env.yaml')
    assert command.hold_s == float(transform['success_hold_s'])
    assert command.transition_joint_tolerance == float(transform['commands']['transform']['auto_switch_joint_tolerance'])
    action = cast(SnakeCrawlJointTargetRateActionCfg, cfg.actions['joint_pos'])
    assert action.transition_end_joint_radius is not None
    joints = transform['standing_joint_pos']
    bounds = transform['actions']['joint_pos']['joint_target_bounds']
    for i, name in ((0, 'joint1'), (5, 'joint6')):
        center = float(joints[i])
        assert (center - action.transition_end_joint_radius, center + action.transition_end_joint_radius) == pytest.approx(tuple(float(x) for x in bounds[name]))
