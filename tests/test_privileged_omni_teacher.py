"""一次联合PPO的特权教师，与部署学生/冻结教师完全隔离。"""

from types import SimpleNamespace
from typing import cast

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.distillation import MultiTeacherPpoCfg, MultiTeacherRunner
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner, OmniTeacherRunnerCfg
from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.env_cfg import UniversalEnvCfg
from zbot_rl_mjlab.tasks.universal.privileged import (
    PRIVILEGED_OMNI_OBS_DIM,
    PrivilegedOmniCommandCfg,
    privileged_state,
)

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'


def test_teacher_has_shared_privileged_actor_critic_and_plain_ppo():
    env = cast(UniversalEnvCfg, load_env_cfg(TASK))
    agent = cast(OmniTeacherRunnerCfg, load_rl_cfg(TASK))
    assert env.omni_site_frame and env.motion_site == 'foot_0_sole'
    assert set(env.observations) == {'actor'}
    assert env.observations['actor'].history_length is None
    assert not env.observations['actor'].enable_corruption
    assert agent.obs_groups == {'actor': ('actor',), 'critic': ('actor',)}
    assert agent.actor.obs_normalization and agent.critic.obs_normalization
    assert not isinstance(agent.algorithm, MultiTeacherPpoCfg)
    assert 'MultiTeacher' not in agent.algorithm.class_name
    assert load_runner_cls(TASK) is OmniTeacherRunner
    assert not issubclass(OmniTeacherRunner, MultiTeacherRunner)
    assert isinstance(env.commands['transform'], PrivilegedOmniCommandCfg)
    assert env.commands['transform'].task_observation_encoding == 'one_hot'
    assert agent.max_iterations == 6000


def test_privileged_state_contains_true_state_com_velocity_and_pending_request(monkeypatch):
    state = torch.arange(106, dtype=torch.float32).reshape(2, 53)
    com_velocity = torch.tensor([[.1, -.05, .3], [-.1, .05, -.3]])
    requested = torch.tensor([[.2, .1, .6], [-.2, -.1, -.6]])
    command = SimpleNamespace(
        teacher_cfg=PrivilegedOmniCommandCfg(task_observation_encoding='one_hot', resampling_time_range=(6., 12.)),
        requested_velocity=requested, velocity_command=torch.zeros(2, 3),
    )
    monkeypatch.setattr(mdp, 'critic_state', lambda env: state)
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda env: com_velocity)
    monkeypatch.setattr(mdp, 'command_term', lambda env: command)
    observed = privileged_state(None)
    assert PRIVILEGED_OMNI_OBS_DIM == 56
    assert observed.shape == (2, 59)
    torch.testing.assert_close(observed[:, :53], state)
    torch.testing.assert_close(observed[:, 53:56], com_velocity)
    torch.testing.assert_close(observed[:, 56:], requested)
    observed.zero_()
    assert state.abs().sum() > 0 and requested.abs().sum() > 0


def test_teacher_does_not_use_deployment_straightening_stage():
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0)
    command = PrivilegedOmniCommandCfg(resampling_time_range=(6., 12.)).build(env)
    command._clear(torch.arange(2))
    command.straightening_mask[:] = True
    command._update_straightening(torch.arange(2))
    assert not command.straightening_mask.any()
    assert not command._straightening_pending.any()
    torch.testing.assert_close(command.observation_mode, command.command[:, 0])


def test_privileged_observation_explicitly_encodes_current_morphology_and_fsm_task():
    from test_universal_task import state_env

    env, _ = state_env()
    env._omni_initial_standing = torch.tensor([True, False])
    env._omni_play_standing = env._omni_initial_standing.clone()
    command = PrivilegedOmniCommandCfg(
        training=False, mode='source', resampling_time_range=(6., 12.), task_observation_encoding='scalar',
    ).build(env)
    env.command_manager.get_term = lambda _: command
    command.reset(torch.arange(2))
    observation = privileged_state(env)
    torch.testing.assert_close(observation[:, 38], torch.tensor([1., -1.]))
    torch.testing.assert_close(observation[:, 39], command.current_task.float())
    assert command.current_task.tolist() == [0, 2]
    command.set_target('snake', torch.tensor([0]))
    observation = privileged_state(env)
    assert observation[0, 38] == 0
    torch.testing.assert_close(observation[:, 39], command.current_task.float())
    assert command.current_task.tolist() == [1, 2]
    assert command.current_morphology.tolist() == [0, -1]


def test_student_recipe_is_preserved_while_teacher_is_separate():
    student = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy')
    assert student.observations['actor'].history_length == 5
    assert set(student.observations) == {'actor', 'critic'}
