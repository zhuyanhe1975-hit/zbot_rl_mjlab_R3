"""Command-relative adaptations preserve the successful snake's physical units."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from zbot_rl_mjlab.tasks.snake import mdp as archived
from zbot_rl_mjlab.tasks.universal import teacher_snake_rewards as rewards


def test_forward_branch_matches_archived_capped_progress_and_lateral_cost(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'bests/snake_world_x_2026-10-19_iter1499/env.yaml'
    cfg = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    actual = torch.tensor([[.1, .04, 0.], [.3, -.05, 0.], [-.1, .02, 0.], [0., .1, 0.]])
    env = SimpleNamespace(cfg=SimpleNamespace(forward_speed_cap=float(cfg['forward_speed_cap']),
                          motion_penalty_min_scale=float(cfg['motion_penalty_min_scale']),
                          motion_penalty_speed_gain=float(cfg['motion_penalty_speed_gain'])))
    monkeypatch.setattr(archived, 'world_velocity_component', lambda env, axis: actual[:, axis])
    expected = (float(cfg['rewards']['head_forward_speed']['weight']) * archived.head_forward_speed(env)
                + float(cfg['rewards']['head_lateral_speed_cost']['weight']) * archived.head_lateral_speed_cost(env))
    command = torch.tensor([[.2, 0., 0.]]).expand_as(actual)
    torch.testing.assert_close(rewards.snake_motion_score(actual, command), expected)


@pytest.mark.parametrize('direction', [(-1., 0.), (0., 1.), (0., -1.), (.6, .8)])
def test_rotating_translation_direction_preserves_progress_and_cross_track_cost(direction):
    x, y = direction
    actual = torch.tensor([[.1 * x - .03 * y, .1 * y + .03 * x, 0.]])
    command = torch.tensor([[.2 * x, .2 * y, 0.]])
    expected = rewards.snake_motion_score(torch.tensor([[.1, .03, 0.]]), torch.tensor([[.2, 0., 0.]]))
    torch.testing.assert_close(rewards.snake_motion_score(actual, command), expected)


def test_reverse_progress_negative_and_credit_capped_at_command_speed():
    command = torch.tensor([[.05, 0., 0.], [.05, 0., 0.], [.05, 0., 0.]])
    actual = torch.tensor([[-.1, 0., 0.], [.05, 0., 0.], [.3, 0., 0.]])
    torch.testing.assert_close(rewards.snake_motion_score(actual, command), torch.tensor([-1.5, .75, .75]))


def test_zero_command_has_no_motion_baseline_or_drift_penalty():
    actual = torch.tensor([[.2, -.1, .8], [0., 0., 0.]])
    torch.testing.assert_close(rewards.snake_motion_score(actual, torch.zeros_like(actual)), torch.zeros(2))


def test_pure_yaw_needs_no_translation_and_mixed_yaw_is_additive():
    actual = torch.tensor([[0., 0., .4], [.1, .02, .4]])
    command = torch.tensor([[0., 0., .4], [.2, 0., .4]])
    yaw = rewards.snake_motion_score(actual[:1], command[:1])
    assert yaw.item() == pytest.approx(4.)
    translation_actual = actual[1:].clone()
    translation_actual[:, 2] = 0.
    translation_command = command[1:].clone()
    translation_command[:, 2] = 0.
    torch.testing.assert_close(rewards.snake_motion_score(actual[1:], command[1:]),
                               rewards.snake_motion_score(translation_actual, translation_command) + yaw)


def test_nonfinite_inputs_do_not_poison_reward():
    actual = torch.tensor([[float('nan'), float('inf'), -float('inf')]])
    command = torch.tensor([[.2, 0., .4]])
    assert torch.isfinite(rewards.snake_motion_score(actual, command)).all()


def test_phase_gates_and_energy_exploration_schedule(monkeypatch):
    actual = torch.tensor([[.1, 0., 0.], [0., 0., .4], [.2, 0., 0.], [.1, 0., 0.], [0., 0., 0.]])
    term = SimpleNamespace(velocity_command=torch.tensor([[.2, 0., 0.], [0., 0., .4], [.2, 0., 0.], [.2, 0., 0.], [0., 0., 0.]]),
                           standing_mode=torch.tensor([False, False, True, False, False]),
                           transition_mask=torch.tensor([False, False, False, True, False]),
                           cfg=SimpleNamespace(velocity_limits=(.25, .20, .8)))
    monkeypatch.setattr(rewards.mdp, 'command_term', lambda env: term)
    monkeypatch.setattr(rewards.mdp, 'tracking_velocity', lambda env: actual)
    torch.testing.assert_close(rewards.exploration_scale(None), torch.tensor([.75, .25, 1., 1., 1.]))
    score = rewards.snake_score(None)
    assert score[0] > 0 and score[1] > 0
    torch.testing.assert_close(score[2:], torch.zeros(3))


def test_auxiliary_snake_costs_preserve_archive_units_and_phase_gates(monkeypatch):
    term = SimpleNamespace(standing_mode=torch.tensor([False, False, True, False]),
                           transition_mask=torch.tensor([False, False, False, True]))
    monkeypatch.setattr(rewards.mdp, 'command_term', lambda env: term)
    q = torch.tensor([[.2, -.4], [0., 0.], [.3, -.3], [.2, -.2]])
    velocity = torch.tensor([[[1., 0., .2], [0., 0., -.1]],
                             [[0., 0., -.4], [0., 0., -.2]],
                             [[0., 0., .2], [0., 0., .2]],
                             [[0., 0., .4], [0., 0., .4]]])
    masses = torch.tensor([[1., 3.]]).expand(4, -1)
    data = SimpleNamespace(joint_pos=q, default_joint_pos=torch.ones_like(q),
                           body_com_lin_vel_w=velocity, body_com_pos_w=torch.zeros_like(velocity))
    limit = torch.pi / 3
    target = torch.tensor([[limit, 0.], [-limit + .0005, limit - .002], [limit, limit], [limit, limit]])
    action = SimpleNamespace(commanded_target=target)
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)},
                          cfg=SimpleNamespace(actions={'joint_pos': SimpleNamespace(snake_target_limit=limit)}),
                          action_manager=SimpleNamespace(get_term=lambda name: action))
    # Exercise both source COM definitions using the same nonuniform body masses.
    monkeypatch.setattr(rewards.transform, 'body_masses', lambda env: masses)
    monkeypatch.setattr(archived, 'body_masses', lambda env: masses)
    env.scene = SceneWithOrigins(env.scene, torch.zeros(4, 3))
    expected_vertical = archived.head_vertical_speed_cost(env)
    torch.testing.assert_close(rewards.snake_vertical_speed(env), torch.cat((expected_vertical[:2], torch.zeros(2))))
    # Source snake defaults are zero, even when the shared scene has standing defaults.
    source_data = SimpleNamespace(joint_pos=q, default_joint_pos=torch.zeros_like(q))
    source_env = SimpleNamespace(scene={'robot': SimpleNamespace(data=source_data)},
                                 action_manager=env.action_manager,
                                 cfg=SimpleNamespace(actions={'joint_pos': SimpleNamespace(target_limit=limit)}))
    expected_pose = archived.joint_pose_deviation_cost(source_env)
    expected_saturation = archived.target_saturation_fraction(source_env)
    torch.testing.assert_close(rewards.snake_joint_deviation(env), torch.cat((expected_pose[:2], torch.zeros(2))))
    torch.testing.assert_close(rewards.snake_target_saturation(env), torch.cat((expected_saturation[:2], torch.zeros(2))))


class SceneWithOrigins(dict):
    def __init__(self, values, origins):
        super().__init__(values)
        self.env_origins = origins
