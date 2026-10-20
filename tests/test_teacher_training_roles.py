"""Motion gets dedicated environment slots instead of competing with failed ascent."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.universal import omni
from zbot_rl_mjlab.tasks.universal.privileged import (
    PrivilegedOmniCommandCfg,
    training_skill_complete,
)

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher'


def environment(monkeypatch, mode=None, *, task=TASK, play=False):
    class Scene(dict):
        env_origins: torch.Tensor

    cfg = load_env_cfg(task, play=play)
    root, q = torch.zeros(100, 13), torch.zeros(100, 6)
    root[:, 3] = 1
    robot = SimpleNamespace(
        data=SimpleNamespace(default_root_state=root.clone()),
        write_joint_state_to_sim=lambda pos, vel, env_ids: q.__setitem__(env_ids, pos),
        write_root_state_to_sim=lambda pos, env_ids: root.__setitem__(env_ids, pos),
    )
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(100, 3)
    env = SimpleNamespace(cfg=cfg, num_envs=100, device='cpu', step_dt=.02, common_step_counter=0, scene=scene)
    command_cfg = cast(PrivilegedOmniCommandCfg, cfg.commands['transform'])
    if mode is not None:
        command_cfg.mode = mode
    command = command_cfg.build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    monkeypatch.setattr(omni, 'collision_radial_envelope', lambda pose: ([], -.26, .26))
    omni.reset_omni(env, None)
    command.reset(torch.arange(100))
    return env, command, q


def test_roles_match_motion_and_transformation_source_poses(monkeypatch):
    _, command, q = environment(monkeypatch)
    assert command.phase.tolist() == [0] * 40 + [2] * 40 + [3] * 10 + [1] * 10
    assert (q.abs().amax(-1) > 0).tolist() == [True] * 40 + [False] * 50 + [True] * 10
    assert command.standing_mode.tolist() == [True] * 40 + [False] * 40 + [True] * 10 + [False] * 10
    assert command.requested_velocity[80:].eq(0).all()


def test_resampling_keeps_dedicated_modes_and_partial_reset_does_not_reassign_roles(monkeypatch):
    env, command, q = environment(monkeypatch)
    old_modes = command.standing_mode.clone()
    command._resample_command(torch.arange(100))
    torch.testing.assert_close(command.standing_mode, old_modes)
    ids = torch.tensor([0, 40, 80, 90])
    expected = command.phase[ids].clone()
    rest = torch.ones(100, dtype=torch.bool)
    rest[ids] = False
    q_before = q.clone()
    env.common_step_counter = 10
    omni.reset_omni(env, ids)
    command.reset(ids)
    torch.testing.assert_close(command.phase[ids], expected)
    torch.testing.assert_close(q[rest], q_before[rest])


def test_only_successful_transform_trials_recycle_and_play_never_does(monkeypatch):
    env, command, _ = environment(monkeypatch)
    assert not training_skill_complete(env).any()
    command.transition_mask[80:90] = False
    assert training_skill_complete(env).nonzero().flatten().tolist() == list(range(80, 90))
    command.cfg.training = False
    assert not training_skill_complete(env).any()


def test_teacher_sampling_does_not_require_a_nonzero_course_score(monkeypatch):
    _, command, _ = environment(monkeypatch)
    ids = torch.arange(80)
    command.tracking_ema.zero_()
    torch.manual_seed(123)
    command._sample_velocity(ids)
    low_course = command.requested_velocity.clone()
    command.tracking_ema.fill_(1.)
    torch.manual_seed(123)
    command._sample_velocity(ids)
    torch.testing.assert_close(command.requested_velocity, low_course)


@pytest.mark.parametrize('mode,phase', [('stand', 0), ('snake', 2)])
def test_single_motion_control_starts_every_environment_in_its_requested_morphology(monkeypatch, mode, phase):
    env, command, q = environment(monkeypatch, mode=mode)
    assert command.phase.eq(phase).all()
    assert not command.transition_mask.any()
    assert (q.abs().amax(-1) > 0).eq(mode == 'stand').all()
    assert command.training_roles(torch.arange(100)).eq(-1).all()
    command._resample_command(torch.arange(100))
    assert command.standing_mode.eq(mode == 'stand').all()
    assert not training_skill_complete(env).any()


@pytest.mark.parametrize('play', [False, True])
def test_current_biped_only_task_never_samples_other_skills_on_reset_or_resampling(monkeypatch, play):
    env, command, q = environment(monkeypatch, task=TASK + '-Wide-Biped', play=play)
    ids = torch.arange(100)
    for step in range(5):
        env.common_step_counter = step * 600
        reset_ids = ids if step == 0 else ids[::3]
        omni.reset_omni(env, reset_ids)
        command.reset(reset_ids)
        command._resample_command(ids)
        assert command.standing_mode.all()
        assert command.current_task.eq(0).all()
        assert command.current_morphology.eq(1).all()
        assert not command.transition_mask.any()
        assert (q.abs().amax(-1) > 0).all()
        assert not training_skill_complete(env).any()
        if not play:
            # All three motion axes still receive training demands.
            assert command.requested_velocity.abs().amax(0).gt(0).all()
            assert command.training_roles(ids).eq(-1).all()
