"""蛇足学生须独立通过持续蛇爬验收，不能以变形成功代替。"""
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict

from zbot_rl_mjlab import universal_evaluation as evaluation


def trace():
    velocity = torch.zeros(10, 20, 3)
    velocity[:, :4, 0] = .08
    velocity[:, 4:8, 1] = .05
    velocity[:, 16:, 0] = .08
    return velocity, torch.ones(10, 20, dtype=torch.bool), torch.ones(20, dtype=torch.bool)


def test_snake_is_a_separate_required_capability():
    velocity, ready, finite = trace()
    result = evaluation.evaluate_capabilities(velocity, ready, finite, include_snake=True)
    assert result['validation_passed'] == result['snake_success'] == 1.
    assert result['snake_motion_fraction'] == 1.
    velocity[:, 16:] = 0.
    result = evaluation.evaluate_capabilities(velocity, ready, finite, include_snake=True)
    assert result['ascent_success'] == 1.
    assert result['snake_success'] == result['validation_passed'] == 0.


@pytest.mark.parametrize('failure', ['burst', 'backward', 'drift', 'yaw', 'unready', 'unsafe', 'nonfinite'])
def test_snake_gate_rejects_invalid_motion(failure):
    velocity, ready, finite = trace()
    if failure == 'burst':
        velocity[:, 16:, 0] = 0.
        velocity[0, 16:, 0] = .8
    elif failure == 'backward':
        velocity[:, 16:, 0] = -.08
    elif failure == 'drift':
        velocity[:, 16:, 1] = .08
    elif failure == 'yaw':
        velocity[::2, 16:, 2] = 1.
        velocity[1::2, 16:, 2] = -1.
    elif failure == 'unready':
        ready[:, 16:] = False
    elif failure == 'unsafe':
        finite[16:] = False
    else:
        velocity[:, 16:, 0] = float('nan')
    result = evaluation.evaluate_capabilities(velocity, ready, finite, include_snake=True)
    assert result['snake_success'] == result['validation_passed'] == 0.


def test_five_group_environment_count_rounds_up_with_all_ascent_poses():
    source = SimpleNamespace(snake_teacher_enabled=True, straighten_before_stand=True)
    assert evaluation.UniversalEvaluator(env_cfg=source).num_envs == 160
    assert evaluation.UniversalEvaluator(env_cfg=source, num_envs=161).num_envs == 165
    source.straighten_before_stand = False
    assert evaluation.UniversalEvaluator(env_cfg=source).num_envs == 20
    assert evaluation.UniversalEvaluator(env_cfg=source, num_envs=21).num_envs == 25
    from zbot_rl_mjlab.experiments.universal import universal_env_cfg
    cfg = universal_env_cfg()
    cfg.snake_teacher_enabled = True
    cfg.straighten_before_stand = True
    assert evaluation.evaluation_env_cfg(cfg, num_envs=161).scene.num_envs == 165


def test_snake_reset_is_archived_forward_pose_without_roll(monkeypatch):
    class Scene(dict):
        env_origins = torch.arange(60).reshape(20, 3).float()
    saved = {}
    state = torch.zeros(20, 13)
    state[:, 3] = 1.
    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=state),
                            write_joint_state_to_sim=lambda q, v, env_ids: saved.update(q=q.clone()),
                            write_root_state_to_sim=lambda root, env_ids: saved.update(root=root.clone()))
    env = SimpleNamespace(cfg=SimpleNamespace(snake_teacher_enabled=True), num_envs=20, device='cpu', scene=Scene(robot=robot))
    monkeypatch.setattr(evaluation, 'collision_radial_envelope', lambda pose: ([[-.03, -.03], [.03, -.03], [.03, .03], [-.03, .03]], -.1, .1))
    evaluation.reset_evaluation_pose(env, None)
    torch.testing.assert_close(saved['root'][16:, :3], env.scene.env_origins[16:] + torch.tensor([0., 0., .0555]))
    torch.testing.assert_close(saved['root'][16:, 3:7], torch.tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.]).expand(4, -1))
    assert saved['q'][16:].eq(0).all()
    expected = saved['root'][[16, 19]].clone()
    evaluation.reset_evaluation_pose(env, torch.tensor([16, 19]))
    torch.testing.assert_close(saved['root'], expected)


def test_snake_commands_world_velocity_and_successful_teacher_ids(monkeypatch):
    evaluator = evaluation.UniversalEvaluator(seconds=2., env_cfg=SimpleNamespace(snake_teacher_enabled=True))
    obs = TensorDict({'actor': torch.arange(20).float()[:, None]}, batch_size=[20])
    env = SimpleNamespace(step_dt=1., observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    evaluator.env = cast(Any, env)
    evaluator.wrapper = cast(Any, SimpleNamespace(reset=lambda: None, get_observations=lambda: obs, step=lambda actions: (obs, torch.zeros(20), None, None)))
    targets, velocities = [], []
    command = SimpleNamespace(set_target=lambda target, ids: targets.append((target, ids)), set_velocity=lambda vx, vy, wz, ids: velocities.append((vx, vy, wz, ids)))
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    local_velocity = trace()[0][0]
    # 局部 -Y 在 +Y 航向下对应世界 +X，须还原后才能正确通过蛇爬验收。
    local_velocity[16:, 0] = 0.
    local_velocity[16:, 1] = -.08
    heading = torch.tensor([1., 0.]).expand(20, -1).clone()
    heading[16:] = torch.tensor([0., 1.])
    monkeypatch.setattr(evaluation.mdp, 'heading', lambda env: heading)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: local_velocity)
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(20, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(20, dtype=torch.bool))
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.zeros(20, 6)
    result = evaluator(Actor())
    assert result['validation_passed'] == result['snake_success'] == 1.
    assert [target for target, ids in targets] == ['stand', 'stand', 'snake', 'stand', 'snake']
    assert velocities[4][:3] == (.08, 0., 0.)
    assert evaluator.foundation_samples is not None
    states, _actions, _weights, teacher_ids, _held_out = cast(
        tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        evaluator.foundation_samples,
    )
    assert teacher_ids[states['actor'][:, 0] >= 16].eq(3).all()
    assert teacher_ids[(states['actor'][:, 0] >= 12) & (states['actor'][:, 0] < 16)].eq(2).all()
