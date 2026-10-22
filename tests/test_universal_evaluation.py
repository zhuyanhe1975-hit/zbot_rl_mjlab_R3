"""独立能力门控不能被静止或错误方向的运动蒙混通过。"""

import torch

from zbot_rl_mjlab.universal_evaluation import evaluate_capabilities


def successful_trace():
    velocity = torch.zeros(10, 16, 3)
    velocity[:, :4, 0] = .08
    velocity[:, 4:8, 1] = .05
    ready = torch.ones(10, 16, dtype=torch.bool)
    finite = torch.ones(16, dtype=torch.bool)
    return velocity, ready, finite


def test_teacher_like_success_passes_all_four_capabilities():
    result = evaluate_capabilities(*successful_trace())
    assert result['validation_passed'] == 1.
    assert result['forward_success'] == 1.
    assert result['ascent_success'] == 1.


def test_standing_without_walking_does_not_pass():
    velocity, ready, finite = successful_trace()
    velocity.zero_()
    result = evaluate_capabilities(velocity, ready, finite)
    assert result['validation_passed'] == 0.
    assert result['forward_success'] == 0.
    assert result['lateral_success'] == 0.


def test_wrong_direction_or_spinning_does_not_pass():
    velocity, ready, finite = successful_trace()
    velocity[:, :4, 0] = -.08
    velocity[:, 4:8, 2] = .5
    assert evaluate_capabilities(velocity, ready, finite)['validation_passed'] == 0.


def test_one_lucky_ascent_is_not_enough_and_nonfinite_fails():
    velocity, ready, finite = successful_trace()
    ready[:, 13:] = False
    result = evaluate_capabilities(velocity, ready, finite)
    assert result['ascent_success'] == .25
    assert result['validation_passed'] == 0.
    ready[:] = True
    finite[:2] = False
    assert evaluate_capabilities(velocity, ready, finite)['validation_passed'] == 0.


def test_brief_bursts_and_cancelling_yaw_do_not_count_as_sustained_motion():
    velocity, ready, finite = successful_trace()
    velocity.zero_()
    velocity[0, :4, 0] = .8
    velocity[0, 4:8, 1] = .5
    velocity[::2, :8, 2] = 1.
    velocity[1::2, :8, 2] = -1.
    result = evaluate_capabilities(velocity, ready, finite)
    assert result['forward_success'] == 0.
    assert result['lateral_success'] == 0.
    assert result['validation_passed'] == 0.


def test_evaluator_preserves_random_streams_and_actor_mode(monkeypatch):
    import random
    from types import SimpleNamespace

    import numpy as np
    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    obs = TensorDict({'actor': torch.zeros(16, 3)}, batch_size=[16])
    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=lambda *args: None)
    env = SimpleNamespace(step_dt=1., observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    wrapper = SimpleNamespace(reset=lambda: None, get_observations=lambda: obs,
                              step=lambda actions: (obs, torch.zeros(16), None, None))
    evaluator = evaluation.UniversalEvaluator(seconds=2.)

    def build():
        evaluator.env, evaluator.wrapper = env, wrapper
        torch.rand(4)
        np.random.rand(3)
        random.random()

    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.rand(16, 6)

    monkeypatch.setattr(evaluator, '_build', build)
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: torch.zeros(16, 3))
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(16, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))
    actor = Actor().train()
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert evaluator(actor)['validation_passed'] == 0.
    assert actor.training
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_capability_mask_excludes_failed_environment_from_protection_data():
    velocity, ready, finite = successful_trace()
    ready[:, 15] = False
    mask = torch.zeros(16, dtype=torch.bool)
    result = evaluate_capabilities(velocity, ready, finite, mask)
    assert result['validation_passed'] == 1.
    assert mask[:15].all() and not mask[15]


def test_successful_student_trajectory_packet_is_trainable_and_filters_failed_env(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    obs = TensorDict({'actor': torch.arange(16).float()[:, None]}, batch_size=[16])
    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=lambda *args: None)
    env = SimpleNamespace(step_dt=1., observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    wrapper = SimpleNamespace(reset=lambda: None, get_observations=lambda: obs,
                              step=lambda actions: (obs, torch.zeros(16), None, None))
    evaluator = evaluation.UniversalEvaluator(seconds=2.)
    evaluator.env, evaluator.wrapper = env, wrapper
    velocity = successful_trace()[0][0]
    ready = torch.ones(16, dtype=torch.bool)
    ready[15] = False
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: velocity)
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: ready)
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 6)

        def forward(self, obs, stochastic_output=False):
            return self.linear(obs['actor'] / 16).tanh()

    actor = Actor()
    assert evaluator(actor)['validation_passed'] == 1.
    packet = evaluator.foundation_samples
    assert packet is not None
    states, actions, weights, teacher_ids, held_out = packet
    assert len(actions) == 30
    assert not states['actor'].is_inference() and not actions.is_inference()
    assert (states['actor'][:, 0] < 15).all()
    assert set(teacher_ids.tolist()) == {0, 1, 2}
    assert weights.eq(1).all() and held_out.any()
    torch.testing.assert_close(actor(states), actions)
    actor(states).square().mean().backward()
    assert actor.linear.weight.grad is not None


def test_evaluation_config_preserves_sensor_history_and_never_mutates_training_schema():
    from mjlab.managers.event_manager import EventTermCfg

    from zbot_rl_mjlab.experiments.universal import (
        configure_deployment,
        universal_env_cfg,
    )
    from zbot_rl_mjlab.universal_evaluation import evaluation_env_cfg

    source = universal_env_cfg()
    configure_deployment(source)
    source.observations['actor'].enable_corruption = True
    source.events['randomize_test_mass'] = EventTermCfg(func=lambda env: None, mode='startup')
    source.events['base_disturbance'] = EventTermCfg(func=lambda env: None, mode='step')
    original_events = dict(source.events)
    nominal = evaluation_env_cfg(source, num_envs=16, randomized=False)
    robust = evaluation_env_cfg(source, num_envs=16, randomized=True)
    for cfg in (nominal, robust):
        assert cfg.observations['actor'].history_length == 5
        assert cfg.observations['actor'].flatten_history_dim
        assert list(cfg.observations['actor'].terms) == list(source.observations['actor'].terms)
        assert not cfg.observations['actor'].enable_corruption
        assert not cfg.commands['transform'].training
        assert cfg.commands['transform'].mode == 'stand'
        assert not cfg.auto_reset and cfg.seed is None
        assert cfg.scene.num_envs == 16
    assert 'randomize_test_mass' not in nominal.events and 'base_disturbance' not in nominal.events
    assert 'randomize_test_mass' in robust.events and 'base_disturbance' in robust.events
    assert source.scene.num_envs == 8192
    assert source.commands['transform'].training
    assert source.observations['actor'].enable_corruption
    assert source.events == original_events


def test_evaluator_build_uses_passed_schema_instead_of_base_task(monkeypatch):
    from types import SimpleNamespace

    from zbot_rl_mjlab import universal_evaluation as evaluation
    from zbot_rl_mjlab.experiments.universal import (
        configure_deployment,
        universal_env_cfg,
    )

    source = universal_env_cfg()
    configure_deployment(source)
    captured = []
    monkeypatch.setattr(evaluation, 'ManagerBasedRlEnv', lambda cfg, **kwargs: captured.append(cfg) or SimpleNamespace())
    monkeypatch.setattr(evaluation, 'RslRlVecEnvWrapper', lambda env: env)
    evaluator = evaluation.UniversalEvaluator(env_cfg=source, randomized=False)
    evaluator._build()
    assert captured[0].observations['actor'].history_length == 5
    assert list(captured[0].observations['actor'].terms) == list(source.observations['actor'].terms)


def test_robust_evaluation_disturbs_every_environment_without_changing_training_fraction():
    from mjlab.managers.event_manager import EventTermCfg

    from zbot_rl_mjlab.experiments.universal import universal_env_cfg
    from zbot_rl_mjlab.universal_evaluation import evaluation_env_cfg

    source = universal_env_cfg()
    source.events['base_disturbance'] = EventTermCfg(func=lambda env: None, mode='step', params={'disturbed_fraction': .25})
    robust = evaluation_env_cfg(source, randomized=True)
    assert robust.events['base_disturbance'].params['disturbed_fraction'] == 1.
    assert source.events['base_disturbance'].params['disturbed_fraction'] == .25
    assert 'base_disturbance' not in evaluation_env_cfg(source, randomized=False).events


def test_evaluator_rebuilds_initial_command_history_after_reset(monkeypatch):
    from types import SimpleNamespace

    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationManager,
        ObservationTermCfg,
    )
    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    env = SimpleNamespace(num_envs=16, device='cpu', step_dt=1., command=torch.zeros(16, 4))
    env.observation_manager = ObservationManager({'actor': ObservationGroupCfg(
        terms={'command': ObservationTermCfg(func=lambda env: env.command)},
        history_length=5, flatten_history_dim=True,
    )}, env)
    def get_observations():
        return TensorDict(env.observation_manager.compute(), batch_size=[16])
    def reset():
        env.command.zero_()
        env.observation_manager.reset()
        env.observation_manager.compute(update_history=True)
    def target(mode, ids):
        env.command[ids, 0] = 1 if mode == 'stand' else -1
    def velocity(vx, vy, wz, ids):
        env.command[ids, 1:] = torch.tensor([vx, vy, wz])
    command = SimpleNamespace(set_target=target, set_velocity=velocity)
    wrapper = SimpleNamespace(reset=reset, get_observations=get_observations,
                              step=lambda actions: (get_observations(), torch.zeros(16), None, None))
    evaluator = evaluation.UniversalEvaluator(seconds=2.)
    evaluator.env, evaluator.wrapper = env, wrapper
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: successful_trace()[0][0])
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(16, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))
    seen = []
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            seen.append(obs['actor'].clone())
            return torch.zeros(16, 6)
    evaluator(Actor())
    torch.testing.assert_close(seen[0], env.command.repeat(1, 5))
    assert torch.all(seen[0][:4].reshape(4, 5, 4)[:, :, 1] == .08)
    assert torch.all(seen[0][4:8].reshape(4, 5, 4)[:, :, 2] == .05)
    assert torch.all(seen[0][8:12].reshape(4, 5, 4)[:, :, 0] == -1)


def test_deployment_evaluator_expands_only_its_own_capability_environment_count():
    import pytest

    from zbot_rl_mjlab.experiments.universal import universal_env_cfg
    from zbot_rl_mjlab.universal_evaluation import (
        UniversalEvaluator,
        evaluation_env_cfg,
    )

    source = universal_env_cfg()
    source.straighten_before_stand = True
    assert UniversalEvaluator(env_cfg=source).num_envs == 128
    assert UniversalEvaluator(env_cfg=source, num_envs=256).num_envs == 256
    assert evaluation_env_cfg(source, num_envs=16).scene.num_envs == 16
    source.straighten_before_stand = False
    assert UniversalEvaluator(env_cfg=source).num_envs == 16
    source.straighten_before_stand = True
    for invalid in (0, 15, 17, True, 16.):
        with pytest.raises(ValueError):
            UniversalEvaluator(env_cfg=source, num_envs=invalid)


def test_deployment_gate_rejects_concentrated_bent_failures_and_weak_overall_ascent(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    source = SimpleNamespace(straighten_before_stand=True)
    evaluator = evaluation.UniversalEvaluator(num_envs=128, seconds=2., env_cfg=source)
    obs = TensorDict({'actor': torch.arange(128).float()[:, None]}, batch_size=[128])
    env = SimpleNamespace(step_dt=1., observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    evaluator.env = env
    evaluator.wrapper = SimpleNamespace(reset=lambda: None, get_observations=lambda: obs,
                                        step=lambda actions: (obs, torch.zeros(128), None, None))
    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=lambda *args: None)
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    velocity = torch.zeros(128, 3)
    velocity[:32, 0] = .08
    velocity[32:64, 1] = .05
    ready = torch.ones(128, dtype=torch.bool)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: velocity)
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: ready)
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(128, dtype=torch.bool))
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.zeros(128, 6)
    actor = Actor()
    for failed in ([100, 101], [96, 100, 104, 108]):
        ready.fill_(True)
        ready[failed] = False
        result = evaluator(actor)
        assert result['ascent_success'] >= .75
        assert result['validation_passed'] == 0.
        packet = evaluator.foundation_samples
        assert packet is not None
        assert not torch.isin(packet[0]['actor'][:, 0], torch.tensor(failed).float()).any()
    ready.fill_(True)
    ready[[100, 104, 108]] = False
    result = evaluator(actor)
    assert result['validation_passed'] == 1.
    assert result['ascent_pose_1_success'] == .75
    assert abs(result['ascent_bent_success'] - 25 / 28) < 1e-6
    assert evaluator.foundation_samples is not None
    assert not torch.isin(evaluator.foundation_samples[0]['actor'][:, 0], torch.tensor([100., 104., 108.])).any()


def test_deployment_evaluation_reset_uses_fixed_heldout_bends_and_cached_envelopes(monkeypatch):
    from types import SimpleNamespace

    from zbot_rl_mjlab import universal_evaluation as evaluation

    class Scene(dict):
        env_origins = torch.zeros(128, 3)
    joints, roots, calls = [], [], []
    state = torch.zeros(128, 13)
    state[:, 3] = 1.
    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=state),
                            write_joint_state_to_sim=lambda q, v, env_ids: joints.append((env_ids.clone(), q.clone())),
                            write_root_state_to_sim=lambda root, env_ids: roots.append(root.clone()))
    env = SimpleNamespace(cfg=SimpleNamespace(straighten_before_stand=True), num_envs=128, device='cpu', scene=Scene(robot=robot))
    def envelope(pose):
        calls.append(pose)
        return ([[-.03, -.03], [.03, -.03], [.03, .03], [-.03, .03]], -.1, .1)
    monkeypatch.setattr(evaluation, 'collision_radial_envelope', envelope)
    evaluation.reset_evaluation_pose(env, None)
    q = joints[-1][1]
    expected = torch.tensor([(0.,) * 6, (.15, 0, 0, 0, 0, 0), (0, 0, .15, 0, 0, 0), (0, 0, 0, 0, -.15, 0), (.15,) * 6, (.2, 0, -.2, 0, .2, 0), (0, .2, 0, -.2, 0, .2), (.15, -.1, .15, .1, -.15, .1)])
    torch.testing.assert_close(q[96:], expected.repeat_interleave(4, dim=0))
    assert torch.isfinite(roots[-1]).all()
    assert (roots[-1][:, 2] > 0).all()
    number = len(calls)
    evaluation.reset_evaluation_pose(env, torch.tensor([99, 100, 127]))
    assert len(calls) == number
    torch.testing.assert_close(joints[-1][1], q[[99, 100, 127]])
    env.cfg.straighten_before_stand = False
    evaluation.reset_evaluation_pose(env, None)
    assert joints[-1][1][96:].eq(0).all()
    env.cfg.straighten_before_stand = True
    env.num_envs = 64
    evaluation.reset_evaluation_pose(env, None)
    assert joints[-1][1][48:].eq(0).all()


def test_forward_frontier_uses_requested_speed_and_strict_sustained_motion():
    for speed in (.15, .2):
        velocity, ready, finite = successful_trace()
        velocity[:, :4, 0] = speed * .8
        assert evaluate_capabilities(velocity, ready, finite, forward_speed=speed, forward_tolerance=.25, strict_forward=True)['forward_success'] == 1.
        velocity[:, :4, 0] = speed * .7
        assert evaluate_capabilities(velocity, ready, finite, forward_speed=speed, forward_tolerance=.25, strict_forward=True)['forward_success'] == 0.
        velocity[:, :4, 0] = speed * 1.2
        assert evaluate_capabilities(velocity, ready, finite, forward_speed=speed, forward_tolerance=.25, strict_forward=True)['forward_success'] == 1.
        velocity[:, :4, 0] = speed * 1.3
        assert evaluate_capabilities(velocity, ready, finite, forward_speed=speed, forward_tolerance=.25, strict_forward=True)['forward_success'] == 0.
        velocity[:6, :4, 0] = speed * .4
        velocity[6:, :4, 0] = speed * 1.9
        assert evaluate_capabilities(velocity, ready, finite, forward_speed=speed, forward_tolerance=.25)['forward_success'] == 1.
        assert evaluate_capabilities(velocity, ready, finite, forward_speed=speed, forward_tolerance=.25, strict_forward=True)['forward_success'] == 0.


def test_forward_settings_validate_construction_and_each_mutable_call():
    import pytest

    from zbot_rl_mjlab.universal_evaluation import UniversalEvaluator

    for key, values in (('forward_speed', (0., -.1, float('nan'), float('inf'))),
                        ('forward_tolerance', (-.1, 1., float('nan'), float('inf')))):
        for value in values:
            with pytest.raises(ValueError):
                UniversalEvaluator(**{key: value})
            evaluator = UniversalEvaluator()
            setattr(evaluator, key, value)
            with pytest.raises(ValueError):
                evaluator(torch.nn.Identity())
            velocity, ready, finite = successful_trace()
            with pytest.raises(ValueError):
                evaluate_capabilities(velocity, ready, finite, **{key: value})


def test_strict_forward_rejects_an_early_fall_without_rejecting_low_ascent_group(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    evaluator = evaluation.UniversalEvaluator(seconds=2., forward_speed=.15, forward_tolerance=.25, strict_forward=True)
    obs = TensorDict({'actor': torch.zeros(16, 3)}, batch_size=[16])
    env = SimpleNamespace(step_dt=1., step=0, observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    def step(actions):
        env.step += 1
        return obs, torch.zeros(16), None, None
    evaluator.env = env
    evaluator.wrapper = SimpleNamespace(reset=lambda: setattr(env, 'step', 0), get_observations=lambda: obs, step=step)
    requests = []
    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=lambda vx, vy, wz, ids: requests.append(vx))
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    velocity = successful_trace()[0][0]
    velocity[:4, 0] = .15
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: velocity)
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(16, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))
    fallen = False
    def height(env):
        h = torch.full((16,), .264)
        h[12:] = .07
        if fallen and env.step == 1:
            h[0] = .07
        return h
    monkeypatch.setattr(evaluation.mdp, 'base_height', height)
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.zeros(16, 6)
    actor = Actor()
    good = evaluator(actor)
    assert good['forward_success'] == 1. and good['ascent_success'] == 1.
    assert good['forward_no_fall_fraction'] == 1.
    assert requests[0] == .15
    fallen = True
    bad = evaluator(actor)
    assert bad['forward_success'] == .75 and bad['ascent_success'] == 1.
    assert bad['forward_no_fall_fraction'] == .75
    evaluator.forward_speed = .2
    velocity[:4, 0] = .2
    evaluator(actor)
    assert requests[-4] == .2


def test_directional_projection_checks_signed_motion_and_preserves_left_group():
    directions = {'forward': (1., 0.), 'backward': (-1., 0.), 'left': (0., 1.), 'right': (0., -1.)}
    for direction, (dx, dy) in directions.items():
        velocity, ready, finite = successful_trace()
        velocity[:, :4, 0] = .04 * dx
        velocity[:, :4, 1] = .04 * dy
        original = velocity.clone()
        result = evaluate_capabilities(velocity, ready, finite, forward_speed=.04, strict_forward=True, motion_direction=direction)
        assert result['forward_success'] == 1.
        assert result['lateral_success'] == 1.
        assert abs(result['forward_velocity'] - .04) < 1e-6
        assert result['lateral_velocity'] > .049
        torch.testing.assert_close(velocity, original)
        velocity[:, :4, :2] *= -1
        result = evaluate_capabilities(velocity, ready, finite, forward_speed=.04, strict_forward=True, motion_direction=direction)
        assert result['forward_success'] == 0. and result['lateral_success'] == 1.


def test_motion_direction_validates_construction_each_call_and_capability_helper():
    import pytest

    from zbot_rl_mjlab.universal_evaluation import UniversalEvaluator

    for invalid in ('north', '', None, 0, []):
        with pytest.raises(ValueError):
            UniversalEvaluator(motion_direction=invalid)
        evaluator = UniversalEvaluator()
        evaluator.motion_direction = invalid
        with pytest.raises(ValueError):
            evaluator(torch.nn.Identity())
        with pytest.raises(ValueError):
            evaluate_capabilities(*successful_trace(), motion_direction=invalid)


def test_directional_evaluator_sends_dynamic_signed_velocity_without_changing_other_groups(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    evaluator = evaluation.UniversalEvaluator(seconds=2., forward_speed=.04)
    obs = TensorDict({'actor': torch.zeros(16, 3)}, batch_size=[16])
    evaluator.env = SimpleNamespace(step_dt=1., observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    evaluator.wrapper = SimpleNamespace(reset=lambda: None, get_observations=lambda: obs,
                                        step=lambda actions: (obs, torch.zeros(16), None, None))
    requests = []
    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=lambda vx, vy, wz, ids: requests.append((vx, vy, wz)))
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: torch.zeros(16, 3))
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(16, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.zeros(16, 6)
    for direction, expected in (('forward', (.04, 0., 0.)), ('backward', (-.04, 0., 0.)), ('left', (0., .04, 0.)), ('right', (0., -.04, 0.))):
        evaluator.motion_direction = direction
        evaluator(Actor())
        assert requests[-4] == expected
        assert requests[-3:] == [(0., .05, 0.), (0., 0., 0.), (0., 0., 0.)]


def test_keyboard_protocol_staggers_live_commands_without_refilling_history(monkeypatch):
    from types import SimpleNamespace

    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationManager,
        ObservationTermCfg,
    )
    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    env = SimpleNamespace(num_envs=16, device='cpu', step_dt=.1, time=0., command=torch.zeros(16, 4))
    manager = ObservationManager({'actor': ObservationGroupCfg(terms={'command': ObservationTermCfg(func=lambda env: env.command)}, history_length=5, flatten_history_dim=True)}, env)
    env.observation_manager = manager
    resets, requests, seen = [], [], []
    original_reset = manager.reset
    def reset_history(*args, **kwargs):
        resets.append(env.time)
        return original_reset(*args, **kwargs)
    monkeypatch.setattr(manager, 'reset', reset_history)
    def observations():
        return TensorDict(manager.compute(), batch_size=[16])
    def reset():
        env.time = 0.
        env.command.zero_()
        requests.clear()
        manager.reset()
        manager.compute(update_history=True)
    def step(actions):
        env.time += env.step_dt
        manager.compute(update_history=True)
        return observations(), torch.zeros(16), None, None
    def target(value, ids):
        env.command[ids, 0] = 1 if value == 'stand' else -1
    def velocity(vx, vy, wz, ids):
        env.command[ids, 1:] = torch.tensor([vx, vy, wz])
        if (ids < 4).all():
            requests.append((env.time, ids.tolist(), vx, vy))
    command = SimpleNamespace(set_target=target, set_velocity=velocity)
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: env.command[:, 1:])
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: torch.ones(16, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))
    monkeypatch.setattr(evaluation.mdp, 'base_height', lambda env: torch.full((16,), .264))
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            seen.append(obs['actor'].reshape(16, 5, 4).clone())
            return torch.zeros(16, 6)
    evaluator = evaluation.UniversalEvaluator(seconds=2., forward_speed=.04, strict_forward=True, keyboard_protocol=True)
    evaluator.env = env
    evaluator.wrapper = SimpleNamespace(reset=reset, get_observations=observations, step=step)
    for direction, initial, target_speed in (('forward', 0., .04), ('backward', .08, -.04)):
        resets.clear()
        seen.clear()
        evaluator.motion_direction = direction
        result = evaluator(Actor())
        assert len(seen) == 57
        assert len(resets) == 2
        assert requests[0][2] == initial
        target_requests = [request for request in requests if len(request[1]) == 1 and request[2] == target_speed]
        assert len(target_requests) == 4
        for phase, (time, ids, _, _) in enumerate(target_requests):
            assert ids == [phase]
            assert abs(time - (2.2 + phase * .5)) < 1e-6
        if direction == 'backward':
            assert any(abs(time - 2.) < 1e-6 and vx == 0. for time, _, vx, _ in requests)
        assert any((frame[0, :, 1] == 0).any() and (frame[0, :, 1] == target_speed).any() for frame in seen)
        assert result['forward_success'] == 1.
        assert result['keyboard_protocol'] == 1.
    # 预备前进期间跌倒，即使计分窗口已经恢复，也不能通过方向验收。
    monkeypatch.setattr(evaluation.mdp, 'base_height', lambda env: torch.full((16,), .07 if env.time < 1. else .264))
    fallen = evaluator(Actor())
    assert fallen['forward_success'] == 0.
    assert fallen['forward_no_fall_fraction'] == 0.
    assert fallen['ascent_success'] == 1.


def test_partial_success_protects_only_successful_students_without_releasing_global_gate(monkeypatch):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from zbot_rl_mjlab import universal_evaluation as evaluation

    evaluator = evaluation.UniversalEvaluator(seconds=2.)
    obs = TensorDict({'actor': torch.arange(16).float()[:, None]}, batch_size=[16])
    evaluator.env = SimpleNamespace(step_dt=1., observation_manager=SimpleNamespace(reset=lambda: None, compute=lambda **kwargs: None))
    evaluator.wrapper = SimpleNamespace(reset=lambda: None, get_observations=lambda: obs, step=lambda actions: (obs, torch.zeros(16), None, None))
    command = SimpleNamespace(set_target=lambda *args: None, set_velocity=lambda *args: None)
    monkeypatch.setattr(evaluation.mdp, 'command_term', lambda env: command)
    monkeypatch.setattr(evaluation.mdp, 'tracking_velocity', lambda env: successful_trace()[0][0])
    ready = torch.ones(16, dtype=torch.bool)
    ready[12:] = False
    monkeypatch.setattr(evaluation.mdp, 'configuration_ready', lambda env: ready)
    monkeypatch.setattr(evaluation.mdp, 'unsafe', lambda env: torch.zeros(16, dtype=torch.bool))
    class Actor(torch.nn.Module):
        def forward(self, obs, stochastic_output=False):
            return torch.zeros(16, 6)
    assert evaluator(Actor())['validation_passed'] == 0.
    packet = evaluator.foundation_samples
    assert packet is not None
    assert len(packet[1]) == 24
    assert (packet[0]['actor'][:, 0] < 12).all()
    assert packet[3].tolist().count(0) == 8
    ready.fill_(False)
    assert evaluator(Actor())['validation_passed'] == 0.
    assert evaluator.foundation_samples is None
