"""Keyboard transitions must not feed walking commands or accept a collapsed biped."""
from typing import cast

import torch
from mjlab.tasks.registry import load_env_cfg
from test_guided_native_sampling import make_command
from test_universal_task import state_env

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.guided_transitions import GuidedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.task_fsm import TaskState


def test_play_request_immediately_gates_effective_command_and_clips_pending_destination():
    command = make_command(2, training=False)
    ids = torch.arange(2)
    command.task_fsm.reset(ids, torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    command._publish_task_state()
    command.requested_velocity[:] = torch.tensor([.5, .2, .5])
    command.velocity_command.copy_(command.requested_velocity)
    command.gait_frequency[:] = .9
    other = command.command[1].clone()
    command.set_target('snake', torch.tensor([0]))
    assert command.current_task.tolist() == [TaskState.DESCENDING, TaskState.BIPED_MOTION]
    assert torch.equal(command.velocity_command[0], torch.zeros(3))
    torch.testing.assert_close(command.requested_velocity[0], torch.tensor([.2, .2, .5]))
    assert float(command.gait_frequency[0]) == float(torch.tensor(.7))
    torch.testing.assert_close(command.command[1], other)
    command.start_step[0] = 123
    command.set_target('snake', torch.tensor([0]))
    assert int(command.start_step[0]) == 123


def test_collapsed_biped_cannot_finish_descent_but_crawling_stays_free(monkeypatch):
    env, _ = state_env()
    env.cfg = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided', play=True)
    cfg = cast(GuidedOmniCommandCfg, env.cfg.commands['transform'])
    command = cfg.build(env)
    env.command_manager.get_term = lambda _: command
    command.task_fsm.reset(torch.arange(2), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    command._publish_task_state()
    command.set_target('snake', torch.tensor([0]))
    monkeypatch.setattr(mdp, 'base_height', lambda _: torch.full((2,), .055))
    monkeypatch.setattr(mdp.transform, 'com_state', lambda _: torch.tensor([[.055, 0.], [.055, 0.]]))
    monkeypatch.setattr(mdp.standing, 'base_gravity_alignment_cost', lambda _: torch.zeros(2))
    monkeypatch.setattr(mdp, 'contacts', lambda _: torch.ones(2, 3))
    monkeypatch.setattr(mdp, 'normalized_force', lambda _: torch.ones(2))
    action = env.action_manager.get_term('joint_pos')
    data = env.scene['robot'].data
    data.joint_pos[0] = torch.tensor([.3, .8, -2., 2., -.8, -.3])
    action.commanded_target[0] = data.joint_pos[0]
    assert not bool(mdp.configuration_ready(env)[0])
    data.joint_pos[0] = 0
    action.commanded_target[0] = .3
    assert not bool(mdp.configuration_ready(env)[0])
    action.commanded_target[0] = 0
    assert bool(mdp.configuration_ready(env)[0])
    command.task_fsm.update_completion(torch.tensor([True, False]))
    command._publish_task_state()
    data.joint_pos[0] = .5
    action.commanded_target[0] = .5
    assert command.current_task[0] == TaskState.SNAKE_MOTION
    assert bool(mdp.configuration_ready(env)[0])


def test_policy_routes_immediately_to_down_snake_and_up_after_requests(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from zbot_rl_mjlab.gaussian_action_models import (
        GaussianActionRoutedTaskNormalizedMLP,
    )
    from zbot_rl_mjlab.task_models import TaskRoutedNetwork
    from zbot_rl_mjlab.tasks.universal.guided_transitions import guided_privileged_state

    command = make_command(2, training=False)
    env = command._env
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    monkeypatch.setattr(mdp, 'critic_state', lambda _: torch.zeros(2, 53))
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda _: torch.zeros(2, 3))
    monkeypatch.setattr(mdp, 'configuration_ready', lambda _, relaxed=False: torch.zeros(2, dtype=torch.bool))
    command.task_fsm.reset(torch.arange(2), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    command._publish_task_state()

    def observe():
        return TensorDict({'actor': guided_privileged_state(env)}, batch_size=[2])

    actor = GaussianActionRoutedTaskNormalizedMLP(observe(), {'actor': ['actor']}, 'actor', 6,
                                                 hidden_dims=(4,), obs_normalization=True)
    router = cast(TaskRoutedNetwork, actor.mlp)
    with torch.no_grad():
        for task, head in enumerate(router.heads):
            sequence = cast(torch.nn.Sequential, head)
            mlp = cast(torch.nn.Sequential, sequence[1])
            last = cast(torch.nn.Linear, mlp[-1])
            last.weight.zero_()
            last.bias.fill_(task+1)
    assert float(actor(observe())[0, 0].detach()) == 1.
    command.set_target('snake', torch.tensor([0]))
    assert float(observe()['actor'][0, 39]) == 1.
    assert float(actor(observe())[0, 0].detach()) == 2.
    command.task_fsm.update_completion(torch.tensor([True, False]))
    command._publish_task_state()
    assert float(observe()['actor'][0, 39]) == 2.
    assert float(actor(observe())[0, 0].detach()) == 3.
    assert float(actor(observe())[1, 0].detach()) == 1.
    command.set_target('stand', torch.tensor([0]))
    assert float(observe()['actor'][0, 39]) == 3.
    assert float(actor(observe())[0, 0].detach()) == 4.
