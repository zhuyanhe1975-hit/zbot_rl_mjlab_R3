"""四教师综合实验的隔离、指令与前爬坐标约定。"""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.distillation import MultiTeacherPpoCfg, MultiTeacherRunnerCfg
from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.commands import UniversalCommandCfg
from zbot_rl_mjlab.tasks.universal.deployment import DeploymentCommandCfg
from zbot_rl_mjlab.tasks.universal.env_cfg import UniversalEnvCfg

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy'


def test_new_experiment_is_isolated_and_keeps_deployment_observations():
    cfg = cast(UniversalEnvCfg, load_env_cfg(TASK))
    old = cast(UniversalEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Universal-Locomotion-Deploy'))
    assert cfg.snake_teacher_enabled and cast(UniversalCommandCfg, cfg.commands['transform']).snake_teacher_enabled
    assert not old.snake_teacher_enabled and not cast(UniversalCommandCfg, old.commands['transform']).snake_teacher_enabled
    assert cfg.observations['actor'].history_length == 5
    assert cfg.observations['actor'].terms.keys() == old.observations['actor'].terms.keys()
    ppo = cast(MultiTeacherRunnerCfg, load_rl_cfg(TASK))
    assert ppo.experiment_name == 'zbot_universal_snake_leg_deploy'
    checkpoints = cast(MultiTeacherPpoCfg, ppo.algorithm).teacher_checkpoints
    assert checkpoints is not None and Path(checkpoints['snake']).is_file()
    old_ppo = cast(MultiTeacherRunnerCfg, load_rl_cfg('Mjlab-Zbot-6dof-Universal-Locomotion-Deploy'))
    assert cast(MultiTeacherPpoCfg, old_ppo.algorithm).teacher_checkpoints is None


def test_pretraining_balances_five_capabilities_and_crawl_is_not_shortened():
    env = SimpleNamespace(num_envs=10, device='cpu', step_dt=.02, common_step_counter=0)
    term = UniversalCommandCfg(snake_teacher_enabled=True, resampling_time_range=(6., 12.)).build(cast(Any, env))
    term.set_pretraining(True)
    term.reset(torch.arange(10))
    assert term.standing_mode.tolist() == [True, True, False, True, False] * 2
    torch.testing.assert_close(term.requested_velocity[4], torch.tensor([.08, 0., 0.]))
    env.command_manager = SimpleNamespace(get_term=lambda _: term)
    env.episode_length_buf = torch.full((10,), 100)
    term.transition_mask.zero_()
    assert mdp.demonstration_time_out(env).tolist() == [False, False, True, True, False] * 2


def test_snake_curriculum_keeps_forward_or_stop_commands():
    env = SimpleNamespace(num_envs=100, device='cpu', step_dt=.02, common_step_counter=0)
    term = DeploymentCommandCfg(snake_teacher_enabled=True, resampling_time_range=(6., 12.)).build(env)
    term._command[:, 0] = -1
    term._sample_velocity(torch.arange(100))
    assert term.requested_velocity[:, 1:].eq(0).all()
    assert set(term.requested_velocity[:, 0].tolist()) <= {0., torch.tensor(.08).item()}


def test_crawl_heading_matches_archived_teacher_world_x():
    n = 2
    q = torch.tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.]).expand(n, 1, 4)
    data = SimpleNamespace(body_link_pos_w=torch.zeros(n, 1, 3), body_link_quat_w=q)
    robot = SimpleNamespace(data=data, find_bodies=lambda _: ([0], []))
    term = SimpleNamespace(standing_mode=torch.zeros(n, dtype=torch.bool), heading=torch.zeros(n, 2), initial_heading=torch.zeros(n, 2), heading_initialized=torch.zeros(n, dtype=torch.bool))
    env = SimpleNamespace(cfg=SimpleNamespace(snake_teacher_enabled=True), scene={'robot': robot}, command_manager=SimpleNamespace(get_term=lambda _: term))
    torch.testing.assert_close(mdp.heading(env), torch.tensor([[1., 0.], [1., 0.]]))
    # 合法的头部左右摆动不得旋转世界前爬目标。
    data.body_link_quat_w = torch.tensor([1., 0., 0., 0.]).expand(n, 1, 4)
    torch.testing.assert_close(mdp.heading(env), torch.tensor([[1., 0.], [1., 0.]]))


def test_student_can_observe_the_snake_teachers_clock():
    term = SimpleNamespace(gait_frequency=torch.full((2,), .5), standing_mode=torch.tensor([True, False]))
    env = SimpleNamespace(cfg=SimpleNamespace(snake_teacher_enabled=True), episode_length_buf=torch.tensor([25, 25]), step_dt=.02, command_manager=SimpleNamespace(get_term=lambda _: term))
    phase = torch.tensor([.5, .7]) * math.pi
    torch.testing.assert_close(mdp.gait_observation(env), torch.stack((phase.sin(), phase.cos(), torch.tensor([.5, .7])), -1))
