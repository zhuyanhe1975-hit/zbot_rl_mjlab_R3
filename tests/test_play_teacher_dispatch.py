"""Exercise the actual play entry point up to environment construction."""

import math
import sys

import pytest
import torch

import play
from zbot_rl_mjlab.task_ppo import TASK_NAMES
from zbot_rl_mjlab.tasks.universal.reward_groups import GROUP_NAMES


@pytest.mark.parametrize('kind', ['site_snake', 'guided', 'routed', 'task', 'grouped', 'legacy_teacher', 'walking'])
def test_play_dispatches_saved_layout_to_registered_runner_before_env_creation(monkeypatch, tmp_path, kind):
    base = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'
    metadata = {
        'role': 'privileged_omni_teacher', 'schema_version': 2,
        'actor_obs_dim': 59, 'critic_obs_dim': 59, 'action_dim': 6,
    }
    task = base + '-Wide-Biped'
    if kind == 'site_snake':
        from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner

        task = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'
        metadata = SiteSnakeTeacherRunner._metadata_schema() | {
            'velocity_limits': (.15, .15, .6), 'joint_target_limit': math.pi / 3,
            'max_target_speed': 2 * math.pi,
        }
    elif kind in ('task', 'routed', 'guided'):
        task = base + ('-Guided' if kind == 'guided' else '-Task-Routed' if kind == 'routed' else '-Task-Critic')
        metadata.update(
            role='privileged_omni_task_teacher', schema_version=3,
            actor_obs_dim=56, critic_obs_dim=56,
            training_algorithm='task_ppo_v1', task_names=TASK_NAMES,
        )
        if kind == 'routed':
            metadata.update(
                role='privileged_omni_task_routed_teacher', training_algorithm='routed_task_ppo_v1',
                actor_branches=TASK_NAMES, actor_normalization='per_task_v1', critic_normalization='per_task_v1',
            )
        if kind == 'guided':
            from zbot_rl_mjlab.guided_teacher import GuidedTeacherRunner

            metadata.update(GuidedTeacherRunner._metadata_schema())
    elif kind == 'grouped':
        task = base + '-Multi-Critic'
        metadata.update(
            role='privileged_omni_reward_group_teacher', schema_version=3,
            actor_obs_dim=56, critic_obs_dim=56,
            training_algorithm='reward_grouped_ppo_v1', reward_groups=GROUP_NAMES,
        )
    elif kind == 'walking':
        task = 'Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking'
    checkpoint = tmp_path / 'model.pt'
    torch.save({'infos': {'privileged_omni_teacher': metadata}} if kind != 'walking' else {}, checkpoint)
    monkeypatch.setattr(sys, 'argv', ['play.py', task, '--checkpoint-file', str(checkpoint), '--device', 'cpu'])

    class EnvironmentReached(Exception):
        pass

    def create_environment(cfg, **kwargs):
        if kind != 'walking':
            assert cfg.commands['transform'].task_observation_encoding == ('scalar' if kind in ('site_snake', 'guided', 'routed', 'task', 'grouped') else 'one_hot')
        raise EnvironmentReached

    monkeypatch.setattr(play, 'ManagerBasedRlEnv', create_environment)
    with pytest.raises(EnvironmentReached):
        play.main()
