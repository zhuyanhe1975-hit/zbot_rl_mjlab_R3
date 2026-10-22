"""The v3 experiment opts into new behavior without replacing the v2 recipe."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import cast

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.experiments.student_ppo import UniversalOmniStudentPPOV3
from zbot_rl_mjlab.tasks.universal.student_ppo_env import StudentPPOCommandCfg
from zbot_rl_mjlab.tasks.universal.student_ppo_rewards import student_velocity_progress
from zbot_rl_mjlab.tasks.universal.student_rewards import student_velocity_tracking


def test_v3_recipe_is_independent_and_preserves_original_v2_card():
    root = Path(__file__).resolve().parents[1]
    card = root / 'src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student_ppo_v2.json'
    assert hashlib.sha256(card.read_bytes()).hexdigest() == '9e09e93adbb4eab2fee0524bcb80a9b3f0096a81be9bebeb5424105cce70808a'
    v2 = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO'
    v3 = UniversalOmniStudentPPOV3.task_id
    before = asdict(load_rl_cfg(v2))
    agent = asdict(load_rl_cfg(v3))
    assert agent['experiment_name'].endswith('_v3')
    assert agent['initial_student_checkpoint'] == before['initial_student_checkpoint']
    assert agent['initial_student_sha256'] == before['initial_student_sha256']
    assert agent['validation_selection_enabled'] is True
    assert agent['validation_interval_iterations'] == 10
    assert agent['algorithm']['reference_tasks'] == (0, 1, 2, 3)
    assert agent['algorithm']['reference_start_coef'] == agent['algorithm']['reference_floor_coef'] == .2
    assert agent['algorithm']['task_balanced_advantages'] is True
    for play in (False, True):
        old, new = load_env_cfg(v2, play=play), load_env_cfg(v3, play=play)
        assert old.rewards['velocity_tracking'].func is student_velocity_tracking
        assert new.rewards['velocity_tracking'].func is student_velocity_progress
        assert 'axis_weights' not in new.rewards['velocity_tracking'].params
        new_command = cast(StudentPPOCommandCfg, new.commands['transform'])
        old_command = cast(StudentPPOCommandCfg, old.commands['transform'])
        assert new_command.balanced_motion_commands
        assert not old_command.balanced_motion_commands
        assert new_command.snake_command_initial_scale == 1.
        assert old_command.snake_command_initial_scale == .5
        for name, reward in old.rewards.items():
            if name != 'velocity_tracking':
                assert asdict(new.rewards[name]) == asdict(reward)
    assert asdict(load_rl_cfg(v2)) == before


def test_v3_launch_example_and_v2_example_are_preserved():
    script = (Path(__file__).resolve().parents[1] / 'train.sh').read_text()
    assert '# 22.c ' in script
    assert '# python train.py Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-PPO \\' in script
    assert 'python train.py ' + UniversalOmniStudentPPOV3.task_id + ' \\' in script
