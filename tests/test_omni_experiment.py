"""全向实验与旧世界X/行走教师配方隔离。"""

from typing import cast

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.distillation import MultiTeacherPpoCfg, MultiTeacherRunnerCfg
from zbot_rl_mjlab.tasks.universal.env_cfg import UniversalEnvCfg
from zbot_rl_mjlab.tasks.universal.omni import OmniCommandCfg, reset_omni

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy'


def test_new_omni_task_uses_site_frame_and_exact_transform_only_teacher():
    env = cast(UniversalEnvCfg, load_env_cfg(TASK))
    agent = cast(MultiTeacherRunnerCfg, load_rl_cfg(TASK))
    algorithm = cast(MultiTeacherPpoCfg, agent.algorithm)
    assert env.omni_site_frame and env.motion_site == 'foot_0_sole'
    assert not env.snake_teacher_enabled
    assert isinstance(env.commands['transform'], OmniCommandCfg)
    assert env.events['reset_transition'].func is reset_omni
    assert env.observations['actor'].history_length == 5
    assert algorithm.enabled_teachers == ('transform',)
    assert not algorithm.teacher_warmup_enabled
    assert algorithm.teacher_checkpoints is None
    assert agent.experiment_name == 'zbot_universal_snake_leg_omni_deploy'


def test_archived_experiment_keeps_its_original_frame_and_teacher_recipe():
    old_env = cast(UniversalEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy'))
    old_agent = cast(MultiTeacherRunnerCfg, load_rl_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy'))
    old_algorithm = cast(MultiTeacherPpoCfg, old_agent.algorithm)
    assert not old_env.omni_site_frame
    assert old_env.snake_teacher_enabled
    assert old_algorithm.enabled_teachers is None
    assert old_algorithm.teacher_warmup_enabled
