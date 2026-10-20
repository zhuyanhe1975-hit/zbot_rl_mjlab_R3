"""教师回放同时展示两种构型，初始目标跟随实际姿态。"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.universal import mdp, omni
from zbot_rl_mjlab.tasks.universal.env_cfg import UniversalEnvCfg
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'


def test_current_routed_play_enables_random_commands_without_changing_training_or_old_play():
    routed = cast(PrivilegedOmniCommandCfg, load_env_cfg(TASK + '-Task-Routed', play=True).commands['transform'])
    assert routed.play_random_velocity
    assert routed.resampling_time_range == (6., 12.)
    assert not cast(PrivilegedOmniCommandCfg, load_env_cfg(TASK + '-Task-Routed').commands['transform']).play_random_velocity
    assert not cast(PrivilegedOmniCommandCfg, load_env_cfg(TASK, play=True).commands['transform']).play_random_velocity


@pytest.mark.parametrize('mode', ['source', 'stand', 'snake'])
def test_play_timers_resample_velocity_without_switching_tasks_or_overwriting_keyboard(monkeypatch, mode):
    torch.manual_seed(42)
    n = 32
    standing = torch.arange(n) % 2 == 0
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0,
                          _omni_play_standing=standing, _omni_initial_standing=standing)
    cfg = PrivilegedOmniCommandCfg(training=False, mode=mode, play_random_velocity=True,
                                  resampling_time_range=(.06, .06))
    term = cfg.build(env)
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.ones(n, dtype=torch.bool))
    term.reset(torch.arange(n))
    if mode != 'source':
        standing = torch.full((n,), mode == 'stand')
    assert torch.equal(term.standing_mode, standing)
    assert torch.isfinite(term.time_left).all()
    first = term.requested_velocity.clone()
    task, morphology = term.current_task.clone(), term.current_morphology.clone()
    term.stop(torch.tensor([0]))
    term.set_velocity(.1, -.05, .2, torch.tensor([1]))
    for _ in range(4):
        env.common_step_counter += 1
        term.compute(.02)
    assert (term.command_counter[2:] >= 2).all()
    assert not torch.equal(term.requested_velocity[2:], first[2:])
    torch.testing.assert_close(term.requested_velocity[0], torch.zeros(3))
    torch.testing.assert_close(term.requested_velocity[1], torch.tensor([.1, -.05, .2]))
    assert term.time_left[:2].isinf().all()
    assert torch.equal(term.standing_mode, standing)
    assert torch.equal(term.current_task, task)
    assert torch.equal(term.current_morphology, morphology)
    assert (term.requested_velocity.abs() <= torch.tensor(cfg.velocity_limits)).all()


@pytest.mark.parametrize('suffix', ['', '-Wide-Biped', '-Multi-Critic', '-Task-Critic', '-Task-Routed', '-Guided'])
def test_continuous_teacher_play_cannot_require_an_automatic_episode_reset(suffix):
    play = load_env_cfg(TASK + suffix, play=True)
    training = load_env_cfg(TASK + suffix)
    assert not play.auto_reset
    # Bundled viewers keep stepping; any terminal result would block the next step.
    assert not play.terminations
    assert training.auto_reset
    assert {'unsafe', 'time_out', 'biped_fallen', 'transition_failure', 'skill_complete'} <= training.terminations.keys()


def test_teacher_play_defaults_to_mixed_source_while_student_keeps_stand():
    teacher = load_env_cfg(TASK, play=True)
    student = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy', play=True)
    assert cast(omni.OmniCommandCfg, teacher.commands['transform']).mode == 'source'
    assert cast(omni.OmniCommandCfg, student.commands['transform']).mode == 'stand'
    assert not teacher.auto_reset
    assert load_env_cfg(TASK).auto_reset
    assert not cast(PrivilegedOmniCommandCfg, teacher.commands['transform']).play_auto_recovery


def test_play_reset_mixes_sixteen_poses_and_matching_persistent_targets(monkeypatch):
    class Scene(dict):
        env_origins: torch.Tensor

    cfg = cast(UniversalEnvCfg, load_env_cfg(TASK, play=True))
    cfg.scene.num_envs = 16
    q, roots = torch.zeros(16, 6), torch.zeros(16, 13)
    roots[:, 3] = 1
    data = SimpleNamespace(default_root_state=roots.clone())
    robot = SimpleNamespace(
        data=data,
        write_joint_state_to_sim=lambda pos, vel, env_ids: q.__setitem__(env_ids, pos),
        write_root_state_to_sim=lambda root, env_ids: roots.__setitem__(env_ids, root),
    )
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(16, 3)
    env = SimpleNamespace(cfg=cfg, scene=scene, num_envs=16, device='cpu', step_dt=.02, common_step_counter=0)
    command_cfg = cast(PrivilegedOmniCommandCfg, cfg.commands['transform'])
    term = command_cfg.build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: term)
    monkeypatch.setattr(omni, 'collision_radial_envelope', lambda pose: ([], -.26, .26))
    omni.reset_omni(env, None)
    term.reset(torch.arange(16))
    standing = q.abs().amax(-1) > 0
    assert standing.sum().item() == 8
    assert torch.equal(term.standing_mode, standing)
    torch.testing.assert_close(term.requested_velocity, torch.tensor([[.05, 0., 0.]]).expand(16, -1))
    assert term.time_left.isinf().all()
    old = term.command.clone()
    term._resample_command(torch.arange(16))
    torch.testing.assert_close(term.command, old)
    term.set_target('snake', torch.tensor([0]))
    term.set_velocity(.05, .02, .1, torch.tensor([0]))
    term._resample_command(torch.arange(16))
    torch.testing.assert_close(term.requested_velocity[0], torch.tensor([.05, .02, .1]))
    assert not term.standing_mode[0]
    term.stop()
    term._resample_command(torch.arange(16))
    assert term.requested_velocity.eq(0).all()


@pytest.mark.parametrize('mode', ['source', 'stand', 'snake'])
def test_auto_motion_for_all_play_modes_and_training_unchanged(mode):
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0)
    cfg = PrivilegedOmniCommandCfg(training=False, mode=mode, resampling_time_range=(6., 12.))
    command = cfg.build(env)
    command.reset(torch.arange(2))
    torch.testing.assert_close(command.requested_velocity, torch.tensor([[.05, 0., 0.]]).expand(2, -1))
    if mode != 'source':
        assert command.standing_mode.eq(mode == 'stand').all()
    cfg.training = True
    command.set_pretraining(True)
    command.reset(torch.arange(2))
    # 原训练预热分组不使用回放的默认速度。
    torch.testing.assert_close(command.requested_velocity, torch.tensor([[.08, 0., 0.], [0., .05, 0.]]))


def test_play_only_keyboard_enters_transform_and_only_for_selected_robot(monkeypatch):
    from test_universal_task import state_env

    env, _ = state_env()
    env._omni_initial_standing = torch.ones(2, dtype=torch.bool)
    command = PrivilegedOmniCommandCfg(training=False, mode='stand', resampling_time_range=(6., 12.)).build(env)
    env.command_manager.get_term = lambda _: command
    command.reset(torch.arange(2))
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.zeros(2, dtype=torch.bool))
    for _ in range(110):
        env.common_step_counter += 1
        command._update_command(None)
    assert command.phase.eq(0).all()  # No automatic recovery despite loss of readiness.
    assert command.standing_mode.all()
    assert command.velocity_command[:, 0].gt(.049).all()
    command.set_target('snake', torch.tensor([0]))
    assert command.transition_mask.tolist() == [True, False]
    assert command.standing_mode.tolist() == [False, True]
    assert command.manual_control.tolist() == [True, False]
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.tensor([True, False]))
    for _ in range(110):
        env.common_step_counter += 1
        command._update_command(None)
    assert command.phase.tolist() == [2, 0]
    assert not command.transition_mask.any()
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.zeros(2, dtype=torch.bool))
    env.common_step_counter += 1
    command._update_command(None)
    assert command.phase.tolist() == [2, 0]


def test_training_preserves_motion_on_loss_of_readiness(monkeypatch):
    from test_universal_task import state_env

    env, _ = state_env()
    env._omni_initial_standing = torch.ones(2, dtype=torch.bool)
    command = PrivilegedOmniCommandCfg(training=True, mode='stand', resampling_time_range=(6., 12.)).build(env)
    env.command_manager.get_term = lambda _: command
    command.reset(torch.arange(2))
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.zeros(2, dtype=torch.bool))
    env.common_step_counter += 1
    command._update_command(None)
    assert command.phase.eq(0).all()
