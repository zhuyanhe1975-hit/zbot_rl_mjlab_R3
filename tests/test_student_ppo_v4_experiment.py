"""The v4 reward and command limits remain isolated from earlier student recipes."""

from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from test_task_contracts import canonical

from zbot_rl_mjlab.experiments.cards import load_card
from zbot_rl_mjlab.experiments.student_ppo import (
    UniversalOmniStudentPPO,
    UniversalOmniStudentPPOV3,
    UniversalOmniStudentPPOV4,
)
from zbot_rl_mjlab.tasks.universal.mdp import smoothed_body_velocity_tracking
from zbot_rl_mjlab.tasks.universal.student_ppo_env import StudentPPOCommandCfg
from zbot_rl_mjlab.tasks.universal.student_ppo_rewards import student_velocity_progress
from zbot_rl_mjlab.tasks.universal.student_rewards import student_velocity_tracking


@pytest.mark.parametrize('play', [False, True])
def test_v4_only_changes_tracking_reward_and_snake_limits(play):
    v3 = load_env_cfg(UniversalOmniStudentPPOV3.task_id, play=play)
    v4 = load_env_cfg(UniversalOmniStudentPPOV4.task_id, play=play)
    tracking = v4.rewards['velocity_tracking']
    assert tracking.func is smoothed_body_velocity_tracking
    assert tracking.weight == v3.rewards['velocity_tracking'].weight
    assert tracking.params == {
        'limits': {'biped': (.5, .2, .5), 'snake': (.06, .06, .5)},
        'time_constant': .75,
        'w_lat': .3,
    }
    command = cast(StudentPPOCommandCfg, v4.commands['transform'])
    old_command = cast(StudentPPOCommandCfg, v3.commands['transform'])
    assert command.snake_velocity_limits == (.06, .06, .5)
    assert command.biped_velocity_ranges == old_command.biped_velocity_ranges
    assert tracking.params['limits']['biped'] == tuple(
        max(abs(low), abs(high)) for low, high in command.biped_velocity_ranges
    )
    # Normalizing exactly the two intended differences must recover all of v3.
    v4.rewards['velocity_tracking'] = v3.rewards['velocity_tracking']
    command.snake_velocity_limits = old_command.snake_velocity_limits
    assert canonical(v4) == canonical(v3)
    for experiment, func in (
        (UniversalOmniStudentPPO, student_velocity_tracking),
        (UniversalOmniStudentPPOV3, student_velocity_progress),
    ):
        old = load_env_cfg(experiment.task_id, play=play)
        assert old.rewards['velocity_tracking'].func is func
        assert cast(StudentPPOCommandCfg, old.commands['transform']).snake_velocity_limits == (.2, .2, .5)


def test_v4_inherits_initialization_validation_selection_and_iteration_count():
    v3 = asdict(load_rl_cfg(UniversalOmniStudentPPOV3.task_id))
    v4 = asdict(load_rl_cfg(UniversalOmniStudentPPOV4.task_id))
    assert v4['experiment_name'] == 'zbot_universal_snake_leg_omni_student_ppo_v4'
    assert v4['run_name'] == 'student150_ppo_v4'
    for name in ('experiment_name', 'run_name'):
        v4[name] = v3[name]
    assert v4 == v3
    assert load_runner_cls(UniversalOmniStudentPPOV4.task_id) is load_runner_cls(UniversalOmniStudentPPOV3.task_id)
    params = load_card('universal_snake_leg_omni_student_ppo_v4')['rewards']['velocity_tracking']['params']
    assert not {'std', 'snake_std', 'axis_weights'} & params.keys()


def test_v4_launch_example_is_available_and_v3_example_is_preserved():
    script = (Path(__file__).resolve().parents[1] / 'train.sh').read_text()
    block = script.split('# 22.c ', 1)[1]
    assert 'python train.py ' + UniversalOmniStudentPPOV4.task_id in block
    assert '--log-root logs/rsl_rl_universal_student_ppo_v4' in block
    # The card fixes the recipe; users may shorten a launch via this CLI override.
    assert int(block.split('--agent.max-iterations ', 1)[1].split()[0]) > 0
    assert 'python train.py ' + UniversalOmniStudentPPOV3.task_id + ' \\' in script
