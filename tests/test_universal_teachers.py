"""多教师冻结、观测逐列语义、动作适配与路由边界。"""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.transformation.actions import expand_mirror, project_mirror
from zbot_rl_mjlab.tasks.transformation.commanded import BidirectionalEnvCfg
from zbot_rl_mjlab.tasks.transformation.teacher_observations import (
    teacher_reference_observation_groups,
)
from zbot_rl_mjlab.teachers import TeacherEnsemble, position_to_rate


@pytest.fixture
def env():
    class Scene(dict):
        env_origins: torch.Tensor

    n = 6
    cfg = cast(BidirectionalEnvCfg, load_env_cfg("Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional", play=True))
    q = torch.tensor(cfg.standing_joint_pos).expand(n, 6).clone()
    quaternion = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0, 0]).expand(n, 1, 4).clone()
    positions = torch.zeros(n, 1, 3)
    positions[:, :, 2] = 0.264
    data = SimpleNamespace(joint_pos=q, joint_vel=torch.full((n, 6), 0.2), body_link_pos_w=positions, body_link_quat_w=quaternion, body_link_lin_vel_w=torch.full((n, 1, 3), 0.1), body_link_ang_vel_w=torch.full((n, 1, 3), 0.2), site_pos_w=torch.zeros(n, 2, 3), site_quat_w=torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]]).expand(n, 2, 4).clone())
    robot = SimpleNamespace(data=data, find_bodies=lambda name: ([0], [name]), find_sites=lambda names, preserve_order=True: ([0, 1], names))
    scene = Scene(robot=robot, feet_ground_contact=SimpleNamespace(data=SimpleNamespace(found=torch.ones(n, 2))), base_ground_contact=SimpleNamespace(data=SimpleNamespace(found=torch.zeros(n, 1))))
    scene.env_origins = torch.zeros(n, 3)
    limits = torch.tensor([-math.pi, math.pi]).expand(n, 6, 2).clone()
    action = SimpleNamespace(commanded_target=q.clone(), raw_action=torch.full((n, 6), 0.2), rate_cfg=SimpleNamespace(max_target_speed=2 * math.pi), _target_limits=lambda: limits)
    velocities = torch.tensor([[.1, 0, 0], [0, .1, 0], [0, 0, .3], [.1, 0, 0], [0, 0, 0], [0, 0, 0]])
    command = SimpleNamespace(standing_mode=torch.tensor([True, True, True, False, False, True]), transition_mask=torch.tensor([False, False, False, False, True, False]), requested_velocity=velocities.clone(), velocity_command=velocities.clone(), phase=torch.tensor([0, 0, 0, 2, 1, 0]))
    return SimpleNamespace(cfg=cfg, scene=scene, device="cpu", num_envs=n, step_dt=.02, common_step_counter=0, episode_length_buf=torch.zeros(n, dtype=torch.long), action_manager=SimpleNamespace(get_term=lambda name: action), command_manager=SimpleNamespace(get_term=lambda name: command))


@pytest.fixture
def teachers(env):
    return TeacherEnsemble(env)


def test_frozen_deterministic_and_routing(env, teachers):
    before = {key: {name: value.clone() for name, value in model.state_dict().items()} for key, model in teachers.models.items()}
    one, two = teachers.query(), teachers.query()
    assert one.actions.shape == (6, 6)
    assert one.weights.shape == (6,)
    assert one.counts == {"walking": 1, "omni": 1, "transform": 2}
    assert one.weights.tolist() == [1, 1, 0, 0, 1, 1]
    assert one.teacher_ids.tolist() == [0, 1, -1, -1, 2, 2]
    torch.testing.assert_close(one.actions, two.actions)
    assert torch.isfinite(one.actions).all() and one.actions.abs().max() <= 1
    for key, model in teachers.models.items():
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key][name])


def test_missing_teacher_fails_clearly(env, tmp_path):
    with pytest.raises(FileNotFoundError, match="walking.*教师检查点不存在"):
        TeacherEnsemble(env, checkpoints={"walking": str(tmp_path / "missing.pt")})
    with pytest.raises(ValueError, match="未知教师"):
        TeacherEnsemble(env, checkpoints={"other": "ignored"})


def test_walking_columns_match_old_functions_with_isolated_memory(env, teachers):
    for key in ("walking", "omni"):
        observed = teachers.observation(key)
        cfg = teachers.configs[key]
        action = env.action_manager.get_term("joint_pos")
        default = teachers.defaults[key]
        data = env.scene["robot"].data
        proxy_data = SimpleNamespace(**vars(data), default_joint_pos=default.expand(6, 6), default_joint_vel=torch.zeros(6, 6))
        proxy_robot = SimpleNamespace(**{**vars(env.scene["robot"]), "data": proxy_data})
        class Scene(dict):
            env_origins: torch.Tensor
        scene = Scene(env.scene)
        scene["robot"] = proxy_robot
        scene.env_origins = env.scene.env_origins
        proxy = SimpleNamespace(**{**vars(env), "cfg": cfg, "scene": scene, "_zbot_step_frequency": teachers._frequency[key].clone(), "episode_length_buf": torch.ones(6, dtype=torch.long)})
        # 原函数在回合首步会重采样；使用非首步并相应修正相位，比较全部列。
        env.episode_length_buf.fill_(1)
        env.common_step_counter = 1
        proxy._zbot_initial_forward_w = teachers._initial_forward[key].clone()
        proxy.action_manager = SimpleNamespace(action=(action.commanded_target - default) / cfg.actions["joint_pos"].scale)
        proxy.command_manager = SimpleNamespace(get_command=lambda name: env.command_manager.get_term("transform").velocity_command)
        pieces = [term.func(proxy, **term.params) for term in cfg.observations["actor"].terms.values()]
        torch.testing.assert_close(teachers.observation(key), torch.cat(pieces, -1))
        assert observed.shape == (6, 32 if key == "walking" else 33)


def test_transform_columns_match_reference_and_mirror_projection(env, teachers):
    action = env.action_manager.get_term("joint_pos")
    action.commanded_target[:, 0] += .2
    action.raw_action[:, 0] += .3
    cfg = teachers.configs["transform"]
    proxy_action = SimpleNamespace(commanded_target=expand_mirror(project_mirror(action.commanded_target)), raw_action=project_mirror(action.raw_action))
    proxy = SimpleNamespace(**{**vars(env), "cfg": cfg, "action_manager": SimpleNamespace(get_term=lambda name: proxy_action)})
    pieces = []
    for term in teacher_reference_observation_groups()["actor"].terms.values():
        value = term.func(proxy, **term.params)
        if term.clip is not None:
            value = value.clamp(*term.clip)
        if term.scale is not None:
            value = value * term.scale
        pieces.append(value)
    pieces.append(torch.where(env.command_manager.get_term("transform").standing_mode, 1., -1.)[:, None])
    torch.testing.assert_close(teachers.observation("transform"), torch.cat(pieces, -1))
    output = torch.tensor([[.1, -.2, .3]]).expand(6, 3)
    # 从零目标验证三维 Beta 动作转换，不触发教师 joint1 边界。
    action.commanded_target.zero_()
    converted = teachers._convert("transform", output)
    torch.testing.assert_close(converted, expand_mirror(output))


def test_position_to_rate_clips_bounds_and_respects_units():
    target = torch.tensor([[0., .5, -.5]])
    limits = torch.tensor([[[-1., 1.], [-1., 1.], [-1., 1.]]])
    torch.testing.assert_close(position_to_rate(torch.tensor([[.02, 9., -9.]]), target, limits, .02, 2), torch.tensor([[.5, 1., -1.]]))
    torch.testing.assert_close(position_to_rate(torch.tensor([[9., -9., 0.]]), torch.tensor([[1., -1., 0.]]), limits, .02, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="必须为正"):
        position_to_rate(target, target, limits, 0, 2)


def test_transform_rate_recovers_asymmetric_targets_without_exceeding_speed(env, teachers):
    action = env.action_manager.get_term("joint_pos")
    action.commanded_target.zero_()
    action.commanded_target[:, 1] = .6
    initial = action.commanded_target.clone()
    for _ in range(4):
        rate = teachers._convert("transform", torch.zeros(env.num_envs, 3))
        assert rate.abs().max() <= 1
        action.commanded_target.add_(rate * env.step_dt * action.rate_cfg.max_target_speed)
    error = lambda target: (target - expand_mirror(project_mirror(target))).abs().amax()
    assert error(action.commanded_target) < error(initial) * .01


def test_transform_conversion_obeys_teacher_bounds_and_student_limits(env, teachers):
    action = env.action_manager.get_term("joint_pos")
    cfg = teachers.configs["transform"].actions["joint_pos"]
    for i, name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
        if name in cfg.joint_target_bounds:
            low, high = cfg.joint_target_bounds[name]
            action.commanded_target[:, i] = high
    rate = teachers._convert("transform", torch.ones(env.num_envs, 3))
    reached = action.commanded_target + rate * env.step_dt * action.rate_cfg.max_target_speed
    assert rate.abs().max() <= 1
    for i, name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")):
        if name in cfg.joint_target_bounds:
            low, high = cfg.joint_target_bounds[name]
            assert (reached[:, i] <= high + 1e-6).all()
            assert (reached[:, i] >= low - 1e-6).all()


@pytest.mark.parametrize(("key", "dimension", "index", "bad_value"), [("walking", 6, 0, float("nan")), ("transform", 3, 4, float("inf"))])
def test_nonfinite_teacher_output_has_no_teacher_id(env, teachers, key, dimension, index, bad_value):
    class NonfiniteTeacher:
        def __call__(self, inputs, stochastic_output=False):
            return torch.full((inputs["actor"].shape[0], dimension), bad_value)

    teachers.models[key] = NonfiniteTeacher()
    result = teachers.query()
    assert result.weights[index] == 0
    assert result.teacher_ids[index] == -1
    assert result.counts[key] == 0


def test_query_does_not_mutate_student(env, teachers):
    keys = set(vars(env))
    tensors = {name: value.clone() for name, value in vars(env.scene["robot"].data).items()}
    action = env.action_manager.get_term("joint_pos")
    original_target = action.commanded_target.clone()
    original_raw = action.raw_action.clone()
    command = env.command_manager.get_term("transform")
    original_command = {name: value.clone() for name, value in vars(command).items()}
    for key in teachers.models:
        teachers.observation(key)
    teachers.query()
    assert set(vars(env)) == keys
    for name, value in tensors.items():
        torch.testing.assert_close(getattr(env.scene["robot"].data, name), value)
    torch.testing.assert_close(action.commanded_target, original_target)
    torch.testing.assert_close(action.raw_action, original_raw)
    for name, value in original_command.items():
        torch.testing.assert_close(getattr(command, name), value)


def test_lying_body_never_uses_walking_teacher(env, teachers):
    env.scene["robot"].data.body_link_quat_w[:] = torch.tensor([1., 0, 0, 0])
    result = teachers.query()
    assert result.weights[:4].sum() == 0
    assert result.counts["walking"] == result.counts["omni"] == 0


def test_requested_pure_yaw_is_not_supervised_during_command_smoothing(env, teachers):
    command = env.command_manager.get_term("transform")
    command.requested_velocity[0] = torch.tensor([0., 0, .3])
    assert teachers.query().weights[0] == 0


def test_far_asymmetric_target_has_no_transform_teacher(env, teachers):
    action = env.action_manager.get_term("joint_pos")
    action.commanded_target[4, 0] += 2.
    assert teachers.query().weights[4] == 0


def test_teacher_command_reprojects_world_velocity(env, teachers):
    command = env.command_manager.get_term("transform")
    # 当前真实旧行走前方为 -X；人为给新任务 +Y 朝向。
    command.heading = torch.tensor([[0., 1.]]).expand(6, 2).clone()
    command.velocity_command[:] = torch.tensor([.1, .02, .05])
    observed = teachers.observation("omni")[:, -3:]
    torch.testing.assert_close(observed, torch.tensor([[.02, -.1, .05]]).expand(6, 3))


def test_public_gait_and_initial_heading_are_teacher_conditions(env, teachers):
    command = env.command_manager.get_term("transform")
    command.gait_frequency = torch.linspace(.3, .9, 6)
    command.initial_heading = torch.tensor([[0., 1.]]).expand(6, 2).clone()
    command.heading_initialized = torch.ones(6, dtype=torch.bool)
    env.episode_length_buf.fill_(25)
    expected_phase = env.episode_length_buf * env.step_dt * command.gait_frequency * (2 * torch.pi)
    phase_features = torch.stack((expected_phase.sin(), expected_phase.cos()), -1)
    before = {name: value.clone() for name, value in vars(command).items()}
    for key, phase_index in (("walking", 28), ("omni", 27)):
        observed = teachers.observation(key)
        torch.testing.assert_close(observed[:, phase_index:phase_index + 2], phase_features)
        cfg = teachers.configs[key]
        expected_frequency = (command.gait_frequency - cfg.step_frequency_min) / (cfg.step_frequency_max - cfg.step_frequency_min)
        torch.testing.assert_close(observed[:, phase_index + 2], expected_frequency)
        if key == "walking":
            torch.testing.assert_close(observed[:, 9], torch.full((6,), -math.pi / 2))
    for name, value in before.items():
        torch.testing.assert_close(getattr(command, name), value)
    # 私有备援条件任意变化，不得影响具有公开条件的真实任务。
    for value in teachers._frequency.values():
        value.fill_(.25)
    for value in teachers._initial_forward.values():
        value.fill_(0)
    torch.testing.assert_close(teachers.observation("walking")[:, 28:30], phase_features)
    torch.testing.assert_close(teachers.observation("walking")[:, 9], torch.full((6,), -math.pi / 2))



@pytest.mark.parametrize('standing', [True, False])
@pytest.mark.parametrize('velocity', [(.01, 0., 0.), (.015, 0., 0.), (0., -.02, 0.), (0., 0., .025), (.008, .008, 0.)])
def test_small_motion_request_is_not_supervised_as_stationary(env, teachers, standing, velocity):
    """运动奖励已生效的小指令不能再接受静止构型教师监督。"""
    command = env.command_manager.get_term('transform')
    command.standing_mode[5] = standing
    command.transition_mask[5] = False
    command.requested_velocity[5] = torch.tensor(velocity)
    command.velocity_command[5] = torch.tensor(velocity)
    labels = teachers.query()
    assert labels.weights[5] == 0
    assert labels.teacher_ids[5] == -1


@pytest.mark.parametrize('transition', [True, False])
def test_stationary_teacher_remains_available_for_quiet_and_transition(env, teachers, transition):
    command = env.command_manager.get_term('transform')
    command.transition_mask[5] = transition
    command.requested_velocity[5] = torch.tensor([.02 if transition else .009, 0., 0.])
    command.velocity_command[5] = command.requested_velocity[5]
    assert teachers.query().teacher_ids[5] == 2


def _prepare_straightening(env):
    """构造受弯曲影响的低位上升，避免修改旧教师配方。"""
    env.cfg = SimpleNamespace(straighten_before_stand=True)
    command = env.command_manager.get_term("transform")
    command.standing_mode.fill_(True)
    command.transition_mask.fill_(True)
    command.requested_velocity.zero_()
    command.velocity_command.zero_()
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = .07
    env.scene["robot"].data.joint_pos.fill_(.2)
    env.action_manager.get_term("joint_pos").commanded_target.fill_(.2)


def _advance_teacher_step(env):
    env.common_step_counter += 1
    env.episode_length_buf += 1


def test_straightening_labels_are_analytic_and_query_is_idempotent(env, teachers):
    _prepare_straightening(env)
    action = env.action_manager.get_term("joint_pos")
    expected = position_to_rate(torch.zeros_like(action.commanded_target), action.commanded_target, action._target_limits(), env.step_dt, action.rate_cfg.max_target_speed)
    before_q = env.scene["robot"].data.joint_pos.clone()
    before_target = action.commanded_target.clone()
    first = teachers.query()
    second = teachers.query()
    torch.testing.assert_close(first.actions, expected)
    torch.testing.assert_close(second.actions, expected)
    assert first.teacher_ids.tolist() == [2] * 6
    assert first.weights.tolist() == [1.] * 6
    assert first.counts == {"walking": 0, "omni": 0, "transform": 6}
    torch.testing.assert_close(before_q, env.scene["robot"].data.joint_pos)
    torch.testing.assert_close(before_target, action.commanded_target)


def test_straightening_waits_for_both_target_and_actual_joints(env, teachers):
    _prepare_straightening(env)
    action = env.action_manager.get_term("joint_pos")
    teachers.query()
    action.commanded_target.zero_()
    _advance_teacher_step(env)
    torch.testing.assert_close(teachers.query().actions, torch.zeros(6, 6))
    env.scene["robot"].data.joint_pos.zero_()
    _advance_teacher_step(env)
    # 两者归零后必须恢复冻结变形教师，而不是持续提供归零标签。
    assert teachers.query().actions.abs().max() > .01


def test_straightening_releases_once_and_does_not_rearm_while_ascending(env, teachers):
    _prepare_straightening(env)
    action = env.action_manager.get_term("joint_pos")
    teachers.query()
    action.commanded_target.zero_()
    env.scene["robot"].data.joint_pos.zero_()
    _advance_teacher_step(env)
    teachers.query()
    action.commanded_target.fill_(.2)
    env.scene["robot"].data.joint_pos.fill_(.2)
    _advance_teacher_step(env)
    guided = teachers.query().actions
    # 同一上升指令继续弯腿时不得重新归直。
    env.cfg.straighten_before_stand = False
    torch.testing.assert_close(guided, teachers.query().actions)


def test_straightening_cancel_new_target_and_episode_reset(env, teachers):
    _prepare_straightening(env)
    command = env.command_manager.get_term("transform")
    action = env.action_manager.get_term("joint_pos")
    teachers.query()
    command.standing_mode.fill_(False)
    _advance_teacher_step(env)
    canceled = teachers.query().actions
    env.cfg.straighten_before_stand = False
    torch.testing.assert_close(canceled, teachers.query().actions)
    env.cfg.straighten_before_stand = True
    command.standing_mode.fill_(True)
    assert (teachers.query().actions == -1).all()
    action.commanded_target.zero_()
    env.scene["robot"].data.joint_pos.zero_()
    _advance_teacher_step(env)
    teachers.query()
    action.commanded_target.fill_(.2)
    env.scene["robot"].data.joint_pos.fill_(.2)
    env.common_step_counter += 1
    env.episode_length_buf.zero_()
    assert (teachers.query().actions == -1).all()


def test_straightening_does_not_flatten_an_initially_standing_robot(env, teachers):
    _prepare_straightening(env)
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = .264
    guided = teachers.query().actions
    env.cfg.straighten_before_stand = False
    torch.testing.assert_close(guided, teachers.query().actions)
    env.cfg.straighten_before_stand = True
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = .07
    _advance_teacher_step(env)
    guided = teachers.query().actions
    env.cfg.straighten_before_stand = False
    torch.testing.assert_close(guided, teachers.query().actions)


def test_straightening_release_is_per_environment_and_repeated_target_keeps_latch(env, teachers):
    _prepare_straightening(env)
    action = env.action_manager.get_term("joint_pos")
    command = env.command_manager.get_term("transform")
    teachers.query()
    action.commanded_target[0].zero_()
    env.scene["robot"].data.joint_pos[1].zero_()
    action.commanded_target[2].fill_(.08)
    env.scene["robot"].data.joint_pos[2].fill_(.08)
    # 重复同一站立目标，不改变上升边沿或归直进度。
    command.standing_mode.fill_(True)
    _advance_teacher_step(env)
    result = teachers.query()
    assert teachers._straighten_active.tolist() == [True, True, False, True, True, True]
    torch.testing.assert_close(result.actions[0], torch.zeros(6))
    torch.testing.assert_close(result.actions[1], -torch.ones(6))


def test_deployment_software_stage_is_shared_without_privileged_height_routing(env, teachers):
    _prepare_straightening(env)
    command = env.command_manager.get_term("transform")
    command.straightening_mask = torch.tensor([True, False, True, False, True, False])
    # 高度与软件阶段故意冲突，教师必须读取和学生相同的公开阶段。
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = .264
    result = teachers.query()
    torch.testing.assert_close(result.actions[command.straightening_mask], -torch.ones(3, 6))
    env.cfg.straighten_before_stand = False
    baseline = teachers.query()
    torch.testing.assert_close(result.actions[~command.straightening_mask], baseline.actions[~command.straightening_mask])
    env.cfg.straighten_before_stand = True
    command.straightening_mask.zero_()
    # 软件已经释放时，低位也不允许另用教师高度 latch 再次归直。
    env.scene["robot"].data.body_link_pos_w[:, :, 2] = .07
    released = teachers.query().actions
    env.cfg.straighten_before_stand = False
    torch.testing.assert_close(released, teachers.query().actions)
