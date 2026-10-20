"""Keep the successful walking action range experiment independent of its baseline."""

from dataclasses import asdict

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg

BASE = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'
WIDE = BASE + '-Wide-Biped'


def test_current_task_is_biped_only_and_keeps_legacy_task():
    for play in (False, True):
        before, after = load_env_cfg(BASE, play=play), load_env_cfg(WIDE, play=play)
        narrow = before.actions['joint_pos']
        wide = after.actions['joint_pos']
        assert isinstance(narrow, SnakeCrawlJointTargetRateActionCfg)
        assert isinstance(wide, SnakeCrawlJointTargetRateActionCfg)
        assert narrow.biped_residual_scale == .25
        assert wide.biped_residual_scale == .375
        old_command, new_command = before.commands['transform'], after.commands['transform']
        assert isinstance(old_command, PrivilegedOmniCommandCfg)
        assert isinstance(new_command, PrivilegedOmniCommandCfg)
        assert old_command.task_observation_encoding == 'one_hot'
        assert new_command.task_observation_encoding == 'scalar'
        assert old_command.mode == 'source'
        assert new_command.mode == 'stand'
        narrow.biped_residual_scale = .375
        old_command.task_observation_encoding = 'scalar'
        old_command.mode = 'stand'
        before_data, after_data = asdict(before), asdict(after)
        # Terrain configuration allocates a new default factory lambda per scene.
        before_factory = before_data['scene']['terrain'].pop('spec_fn')
        after_factory = after_data['scene']['terrain'].pop('spec_fn')
        assert before_factory.__qualname__ == after_factory.__qualname__
        assert before_data == after_data


def test_wide_has_independent_checkpoint_directory_with_same_ppo_and_capacity():
    before, after = load_rl_cfg(BASE), load_rl_cfg(WIDE)
    assert before.experiment_name != after.experiment_name
    before.experiment_name = after.experiment_name
    assert asdict(before) == asdict(after)
    assert load_runner_cls(WIDE) is OmniTeacherRunner
