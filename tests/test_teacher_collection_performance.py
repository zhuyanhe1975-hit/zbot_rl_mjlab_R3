"""Collection optimizations preserve full-batch labels and observation history."""

import pytest
import test_universal_snake_teacher as snake_fixtures
import test_universal_teachers as teacher_fixtures
import torch
from tensordict import TensorDict

env = teacher_fixtures.env
teachers = teacher_fixtures.teachers
snake_env = snake_fixtures.snake_env
snake_teachers = snake_fixtures.snake_teachers


def _full_batch_labels(teachers, ids):
    result = torch.zeros(teachers.env.num_envs, 6)
    for teacher_id, key in enumerate(teachers.teacher_names):
        observation = teachers.observation(key)
        inputs = TensorDict({'actor': observation}, batch_size=[len(ids)])
        output = teachers.models[key](inputs, stochastic_output=False)
        converted = teachers._convert(key, output)
        selected = ids == teacher_id
        result[selected] = converted[selected]
    return result


def test_routed_labels_match_full_batch_and_models_only_see_selected_rows(teachers):
    ids = torch.tensor([0, 1, -1, -1, 2, 2])
    expected = _full_batch_labels(teachers, ids)
    batches = {}
    for key, model in teachers.models.items():
        def record(module, args, key=key):
            batches[key] = args[0]['actor'].shape[0]
        model.register_forward_pre_hook(record)
    result = teachers.query()
    torch.testing.assert_close(result.actions, expected)
    assert result.teacher_ids.tolist() == ids.tolist()
    assert batches == {'walking': 1, 'omni': 1, 'transform': 2}


def test_empty_routes_skip_models(teachers, env):
    command = env.command_manager.get_term('transform')
    command.requested_velocity[:] = torch.tensor([0., 0., .3])
    command.transition_mask.zero_()
    def unexpected(*args, **kwargs):
        pytest.fail('empty teacher route ran inference')
    for model in teachers.models.values():
        model.register_forward_pre_hook(unexpected)
    result = teachers.query()
    assert result.counts == {'walking': 0, 'omni': 0, 'transform': 0}
    assert result.teacher_ids.tolist() == [-1] * env.num_envs


def test_selected_nonfinite_observation_and_output_keep_other_labels(teachers, env):
    # Joint velocity does not affect routing; an invalid observation must only
    # suppress the corresponding label, including after compacting the batch.
    env.scene['robot'].data.joint_vel[4, 0] = float('nan')
    model = teachers.models['transform']
    def nonfinite_output(module, args, output):
        output[-1] = float('inf')
        return output
    model.register_forward_hook(nonfinite_output)
    result = teachers.query()
    assert result.teacher_ids.tolist() == [0, 1, -1, -1, -1, -1]
    assert result.counts == {'walking': 1, 'omni': 1, 'transform': 0}
    assert torch.isfinite(result.actions).all()


def test_snake_observation_built_once_per_query_and_labels_match_full_batch(snake_teachers, snake_env, monkeypatch):
    command = snake_env.command_manager.get_term('transform')
    command.requested_velocity[1:5] = torch.tensor([-.08, 0., 0.])
    command.velocity_command[:] = command.requested_velocity
    ids = torch.tensor([3, -1, -1, -1, -1, 3])
    expected = _full_batch_labels(snake_teachers, ids)
    original = snake_teachers._snake_observation
    calls = []
    def observe():
        calls.append(1)
        return original()
    monkeypatch.setattr(snake_teachers, '_snake_observation', observe)
    batches = []
    snake_teachers.models['snake'].register_forward_pre_hook(lambda module, args: batches.append(args[0]['actor'].shape[0]))
    result = snake_teachers.query()
    torch.testing.assert_close(result.actions, expected)
    assert result.teacher_ids.tolist() == ids.tolist()
    assert len(calls) == 1
    assert batches == [2]
    # Repeated queries at the same stamp must still see current action values.
    snake_env.action_manager.get_term('joint_pos').raw_action.add_(.2)
    updated = snake_teachers.query()
    assert len(calls) == 2
    assert not torch.equal(updated.actions, result.actions)
    torch.testing.assert_close(updated.actions, _full_batch_labels(snake_teachers, ids))


@pytest.mark.parametrize('key', ['walking', 'omni', 'transform'])
def test_subset_conversion_matches_full_batch_with_nonuniform_targets(teachers, env, key):
    action = env.action_manager.get_term('joint_pos')
    action.commanded_target += torch.arange(env.num_envs)[:, None] * .01
    dimension = 3 if key == 'transform' else 6
    output = torch.linspace(-.5, .5, env.num_envs * dimension).reshape(env.num_envs, dimension)
    ids = torch.tensor([1, 4])
    expected = teachers._convert(key, output)[ids]
    torch.testing.assert_close(teachers._convert(key, output[ids], env_ids=ids), expected)


def test_walking_observation_state_is_built_once_per_query(teachers, monkeypatch):
    from zbot_rl_mjlab.tasks.walking import mdp

    calls = {}
    for name in ('base_lin_vel', 'base_ang_vel', 'projected_gravity'):
        original = getattr(mdp, name)
        def record(env, name=name, original=original):
            calls[name] = calls.get(name, 0) + 1
            return original(env)
        monkeypatch.setattr(mdp, name, record)
    original_velocity = teachers._walking_velocity_command
    def velocity():
        calls['velocity_command'] = calls.get('velocity_command', 0) + 1
        return original_velocity()
    monkeypatch.setattr(teachers, '_walking_velocity_command', velocity)
    teachers.query()
    assert calls == {'base_lin_vel': 1, 'base_ang_vel': 1, 'projected_gravity': 1, 'velocity_command': 1}


def test_query_local_walking_observations_match_public_and_refresh_same_stamp(teachers, env):
    captured = {}
    handles = []
    for key in ('walking', 'omni'):
        def record(module, args, key=key):
            captured[key] = args[0]['actor'].clone()
        handles.append(teachers.models[key].register_forward_pre_hook(record))
    teachers.query()
    initial = {key: value.clone() for key, value in captured.items()}
    # No simulation step advances here; shared data must be local to the query.
    data = env.scene['robot'].data
    data.body_link_lin_vel_w.add_(.4)
    data.body_link_ang_vel_w.add_(.3)
    env.action_manager.get_term('joint_pos').raw_action.add_(.1)
    teachers.query()
    for key, index in (('walking', 0), ('omni', 1)):
        expected = teachers.observation(key)[index:index + 1]
        torch.testing.assert_close(captured[key], expected)
        assert not torch.equal(captured[key], initial[key])
    for handle in handles:
        handle.remove()
