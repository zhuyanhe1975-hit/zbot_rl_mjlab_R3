"""Editable cards keep recipe values together without replacing runtime callbacks."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from zbot_rl_mjlab.experiments.cards import apply_agent, apply_environment, load_card


def callback():
    pass


@dataclass
class Reward:
    weight: float = 1.
    params: dict = field(default_factory=lambda: {'std': .1})
    func: object = callback


@dataclass
class Environment:
    episode_length_s: float = 45.
    velocity_limits: tuple[float, ...] = (.1, .2, .3)
    auto_reset: bool = True
    gains: list[float] = field(default_factory=lambda: [1., 2.])
    optional_limit: float | None = 1.
    rewards: dict = field(default_factory=lambda: {'tracking': Reward()})


@dataclass
class Model:
    hidden_dims: tuple[int, ...] = (256, 128)
    obs_normalization: bool = True
    class_name: str = 'runtime:Model'


@dataclass
class Algorithm:
    learning_rate: float = .0003
    optional_rate: float | None = None
    class_name: str = 'runtime:PPO'


@dataclass
class Agent:
    actor: Model = field(default_factory=Model)
    algorithm: Algorithm = field(default_factory=Algorithm)


def write_card(directory, name, **settings):
    (directory / f'{name}.json').write_text(json.dumps({'schema_version': 1} | settings))


def test_inheritance_removal_reloading_and_independent_siblings(tmp_path):
    write_card(tmp_path, '_defaults', env={'episode_length_s': 10., 'optional_limit': 3.}, agent={'actor': {'hidden_dims': [64]}})
    write_card(tmp_path, 'first', inherits='_defaults.json', env={'$remove': ['optional_limit'], 'episode_length_s': 20.})
    write_card(tmp_path, 'second', inherits='_defaults.json', env={'optional_limit': None})
    first = load_card('first.json', tmp_path)
    second = load_card('second', tmp_path)
    assert first['env'] == {'episode_length_s': 20.}
    assert second['env']['optional_limit'] is None
    first['agent']['actor']['hidden_dims'][0] = 999
    assert second['agent']['actor']['hidden_dims'] == [64]
    write_card(tmp_path, '_defaults', env={'episode_length_s': 11.})
    assert load_card('second', tmp_path)['env']['episode_length_s'] == 11.


def test_card_edits_both_environment_and_agent_with_play_overrides(tmp_path):
    write_card(tmp_path, 'experiment', env={'episode_length_s': 20, 'velocity_limits': [.2, .3, .4], 'optional_limit': None, 'gains': [3, 4]},
               train={'env': {'auto_reset': False}, 'rewards': {'tracking': {'weight': 5.}}},
               rewards={'tracking': {'weight': 4, 'params': {'std': .2, 'new_threshold': 3.}}},
               agent={'actor': {'hidden_dims': [128, 64]}, 'algorithm': {'learning_rate': .001}},
               play={'env': {'episode_length_s': 100}, 'rewards': {'tracking': {'weight': 2.}}})
    card = load_card('experiment', tmp_path)
    train, play, agent = Environment(), Environment(), Agent()
    apply_environment(train, card)
    apply_environment(play, card, play=True)
    apply_agent(agent, card)
    assert train.episode_length_s == 20.
    assert type(train.episode_length_s) is float
    assert play.episode_length_s == 100.
    assert train.velocity_limits == (.2, .3, .4)
    assert train.optional_limit is None
    assert train.rewards['tracking'].weight == 5.
    assert train.auto_reset is False
    assert play.auto_reset is True
    assert train.gains == [3., 4.]
    assert type(train.gains) is list
    assert play.rewards['tracking'].weight == 2.
    assert train.rewards['tracking'].params == {'std': .2, 'new_threshold': 3.}
    assert train.rewards['tracking'].func is callback
    assert agent.actor.hidden_dims == (128, 64)
    assert agent.algorithm.learning_rate == .001
    assert agent.actor.class_name == 'runtime:Model'
    assert agent.algorithm.class_name == 'runtime:PPO'


@pytest.mark.parametrize('settings, message', [
    ({'inherits': '../other.json'}, 'filename'),
    ({'inherits': 'missing.json'}, 'cannot read'),
    ({'env': []}, 'env must'),
    ({'play': {'agent': {}}}, 'play may'),
    ({'rewards': {'tracking': {'weight': 'heavy'}}}, 'numeric'),
    ({'rewards': {'tracking': {'weight': True}}}, 'numeric'),
    ({'env': {'episode_length_s': float('nan')}}, 'finite'),
    ({'agent': {'actor': {'class_name': 'other:Model'}}}, 'structural'),
])
def test_invalid_card_shapes_and_values(tmp_path, settings, message):
    write_card(tmp_path, 'invalid', **settings)
    with pytest.raises(ValueError, match=message):
        load_card('invalid', tmp_path)


def test_inheritance_cycle_and_invalid_remove(tmp_path):
    write_card(tmp_path, 'first', inherits='second')
    write_card(tmp_path, 'second', inherits='first.json')
    with pytest.raises(ValueError, match='cycle'):
        load_card('first', tmp_path)
    write_card(tmp_path, 'first', env={'$remove': 'field'})
    with pytest.raises(ValueError, match=r'\$remove'):
        load_card('first', tmp_path)


@pytest.mark.parametrize('env, message', [
    ({'typo': 1}, 'unknown field'),
    ({'episode_length_s': True}, 'expected float'),
    ({'auto_reset': 1}, 'expected bool'),
    ({'episode_length_s': None}, 'null'),
])
def test_environment_rejects_invalid_fields_and_types(env, message):
    with pytest.raises(ValueError, match=message):
        apply_environment(Environment(), {'schema_version': 1, 'env': env})


def test_unknown_reward_and_callback_writes_are_rejected():
    with pytest.raises(ValueError, match='unknown reward'):
        apply_environment(Environment(), {'schema_version': 1, 'rewards': {'missing': {'weight': 1}}})
    with pytest.raises(ValueError, match='structural'):
        apply_environment(Environment(), {'schema_version': 1, 'env': {'rewards': {'tracking': {'func': 'other'}}}})


@pytest.mark.parametrize('patch', [
    {'actor': {'hidden_dims': [64, -1]}},
    {'actor': {'hidden_dims': [True]}},
    {'actor': {'hidden_dims': []}},
    {'algorithm': {'learning_rate': float('inf')}},
    {'algorithm': {'optional_rate': True}},
    {'actor': {'missing': 3}},
])
def test_agent_rejects_invalid_dimensions_numbers_and_fields(patch):
    with pytest.raises(ValueError):
        apply_agent(Agent(), {'schema_version': 1, 'agent': patch})


@pytest.fixture
def registered_cards(tmp_path, monkeypatch):
    """Exercise real experiment construction against isolated editable card files."""
    import shutil

    from zbot_rl_mjlab.experiments import cards

    for source in Path(cards.__file__).parent.glob('*.json'):
        shutil.copy2(source, tmp_path / source.name)
    monkeypatch.setattr(cards, '__file__', str(tmp_path / '__init__.py'))
    return tmp_path


def test_registered_training_and_play_share_one_editable_card(registered_cards, monkeypatch):
    from zbot_rl_mjlab.experiments import base
    from zbot_rl_mjlab.experiments.walking import FlatWalking

    path = registered_cards / 'walking.json'
    card = json.loads(path.read_text())
    card['agent'] = {'actor': {'hidden_dims': [64, 32]}, 'critic': {'hidden_dims': [64, 32]},
                     'algorithm': {'learning_rate': .0007}}
    card.setdefault('rewards', {})['upright'] = {'weight': .73}
    path.write_text(json.dumps(card))
    captured = {}
    monkeypatch.setattr(base, 'register_mjlab_task', lambda **values: captured.update(values))
    FlatWalking().register()
    assert captured['rl_cfg'].actor.hidden_dims == (64, 32)
    assert captured['rl_cfg'].critic.hidden_dims == (64, 32)
    assert captured['rl_cfg'].algorithm.learning_rate == .0007
    assert captured['env_cfg'].rewards['upright'].weight == .73
    assert captured['play_env_cfg'].rewards['upright'].weight == .73
    card['rewards']['upright']['weight'] = .91
    path.write_text(json.dumps(card))
    assert FlatWalking().build_env().rewards['upright'].weight == .91


def test_parent_card_update_reaches_inheriting_experiments(registered_cards):
    from zbot_rl_mjlab.experiments.variants import DisturbedWalking
    from zbot_rl_mjlab.experiments.walking import FlatWalking

    path = registered_cards / 'periodic_stepping.json'
    card = json.loads(path.read_text())
    card['rewards']['upright']['weight'] = .84
    path.write_text(json.dumps(card))
    assert FlatWalking().build_env().rewards['upright'].weight == .84
    assert DisturbedWalking().build_env().rewards['upright'].weight == .84


def test_disturbed_card_pins_archived_differences_from_parent(registered_cards):
    import yaml
    from test_bests import BESTS, _Loose, archived_env

    from zbot_rl_mjlab.experiments.variants import DisturbedWalking

    archive = BESTS / 'zbot_disturbed_walking'
    saved = archived_env(archive)
    agent = yaml.load((archive / 'agent.yaml').read_text(), Loader=_Loose)
    card = json.loads((registered_cards / 'disturbed_walking.json').read_text())
    for name in ('swing_clearance', 'soft_landing'):
        assert card['rewards'][name]['weight'] == saved['rewards'][name]['weight']
    assert card['agent']['max_iterations'] == agent['max_iterations']

    # A later parent edit must not silently replace the archived recipe.
    parent_path = registered_cards / 'walking.json'
    parent = json.loads(parent_path.read_text())
    parent.setdefault('rewards', {})['swing_clearance'] = {'weight': 9.0}
    parent['rewards']['soft_landing'] = {'weight': -9.0}
    parent.setdefault('agent', {})['max_iterations'] = 9999
    parent_path.write_text(json.dumps(parent))
    for play in (False, True):
        cfg = DisturbedWalking().build_env(play=play)
        for name in ('swing_clearance', 'soft_landing'):
            assert cfg.rewards[name].weight == saved['rewards'][name]['weight']
        impulse = cfg.events['base_disturbance'].params
        saved_impulse = saved['events']['base_disturbance']['params']
        for name in ('force_range', 'torque_range', 'duration_s', 'cooldown_s'):
            assert list(impulse[name]) == saved_impulse[name]
        assert impulse['disturbed_fraction'] == saved_impulse['disturbed_fraction']
        friction = cfg.events['randomize_foot_friction'].params
        assert list(friction['ranges']) == saved['events']['randomize_foot_friction']['params']['ranges']
        encoder = cfg.events['randomize_encoder_bias'].params
        assert list(encoder['bias_range']) == saved['events']['randomize_encoder_bias']['params']['bias_range']
        com = cfg.events['randomize_base_com'].params
        assert {axis: list(bounds) for axis, bounds in com['ranges'].items()} == saved['events']['randomize_base_com']['params']['ranges']
    assert DisturbedWalking().build_ppo().max_iterations == agent['max_iterations']


def test_wrong_registered_card_identity_fails_before_build(registered_cards):
    from zbot_rl_mjlab.experiments.walking import FlatWalking

    path = registered_cards / 'walking.json'
    card = json.loads(path.read_text())
    card['task_id'] = 'wrong-task'
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match='task_id'):
        FlatWalking().build_ppo()


def test_frozen_addon_card_values_and_explicit_cli_precedence():
    from zbot_rl_mjlab.experiments.addons import BaseDisturbance
    from zbot_rl_mjlab.experiments.cards import configure_addons

    original = BaseDisturbance()
    card = {'schema_version': 1, 'addons': {'BaseDisturbance': {
        'disturbed_fraction': .5, 'force_range': [-8, 8],
    }}}
    configured = configure_addons((original,), card)[0]
    assert configured is not original
    assert configured.disturbed_fraction == .5
    assert configured.force_range == (-8., 8.)
    assert all(type(value) is float for value in configured.force_range)
    assert original.disturbed_fraction == .25
    assert original.force_range == (-5., 5.)
    cli = BaseDisturbance(disturbed_fraction=1.)
    assert configure_addons([cli], card, overrides=(BaseDisturbance,))[0].disturbed_fraction == 1.


def test_optional_tuple_addons_and_integer_dictionary_keys():
    from zbot_rl_mjlab.experiments.addons import DomainRandomization
    from zbot_rl_mjlab.experiments.cards import configure_addons

    original = DomainRandomization(encoder_bias_range=None)
    card = {'schema_version': 1, 'addons': {'DomainRandomization': {
        'encoder_bias_range': [-.01, .01], 'base_com_ranges': {'0': [-.05, .05]},
    }}}
    configured = configure_addons([original], card)[0]
    assert configured.encoder_bias_range == (-.01, .01)
    assert configured.base_com_ranges[0] == (-.05, .05)
    assert '0' not in configured.base_com_ranges
    assert original.base_com_ranges[0] == (-.025, .025)
    assert original.encoder_bias_range is None
    card['addons']['DomainRandomization']['encoder_bias_range'] = [1., 2., 3.]
    with pytest.raises(ValueError, match='tuple items'):
        configure_addons([original], card)


@pytest.mark.parametrize('settings', [
    {'MissingAddon': {}},
    {'BaseDisturbance': {'missing_field': 1}},
    {'BaseDisturbance': {'force_range': [1., 2., 3.]}},
    {'BaseDisturbance': {'disturbed_fraction': True}},
    {'BaseDisturbance': {'disturbed_fraction': float('inf')}},
    {'BaseDisturbance': {'func': 'other'}},
])
def test_addon_configuration_rejects_unknown_and_invalid_settings(settings):
    from zbot_rl_mjlab.experiments.addons import BaseDisturbance
    from zbot_rl_mjlab.experiments.cards import configure_addons

    with pytest.raises(ValueError):
        configure_addons([BaseDisturbance()], {'schema_version': 1, 'addons': settings})


def test_addon_replacement_runs_dataclass_post_init_validation():
    from zbot_rl_mjlab.experiments.cards import configure_addons
    from zbot_rl_mjlab.experiments.universal import StudentDisturbance

    with pytest.raises(ValueError, match='预热外扰比例'):
        configure_addons([StudentDisturbance()], {'schema_version': 1, 'addons': {
            'StudentDisturbance': {'pretraining_scale': 2.},
        }})


def test_cli_overrides_only_requested_addon_fields(registered_cards, monkeypatch):
    from zbot_rl_mjlab import cli
    from zbot_rl_mjlab.experiments import base
    from zbot_rl_mjlab.experiments.variants import DeployWalking

    path = registered_cards / 'deploy_walking.json'
    card = json.loads(path.read_text())
    card.setdefault('addons', {}).setdefault('BaseDisturbance', {}).update(
        disturbed_fraction=.63, force_range=[-2., 3.])
    path.write_text(json.dumps(card))
    assert DeployWalking().build_env().events['base_disturbance'].params['disturbed_fraction'] == .63
    captured = {}
    monkeypatch.setattr(base, 'register_mjlab_task', lambda **values: captured.update(values))
    monkeypatch.setattr(cli.mjlab_registry, '_REGISTRY', {})
    cli.apply_cli([DeployWalking.task_id, '--disturbance', '--disturbance-fraction=1.0'])
    actual = captured['env_cfg'].events['base_disturbance'].params
    assert actual['disturbed_fraction'] == 1.
    assert actual['force_range'] == (-2., 3.)
    assert captured['play_env_cfg'].events['base_disturbance'].params['force_range'] == (-2., 3.)


def test_bundled_cards_match_editor_schema():
    # Optional development validation; card loading itself uses only the standard library.
    jsonschema = pytest.importorskip('jsonschema')
    from zbot_rl_mjlab.experiments import cards

    directory = Path(cards.__file__).parent
    schema = json.loads((directory / 'schema.json').read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    failures = {path.name: [error.message for error in validator.iter_errors(json.loads(path.read_text()))]
                for path in directory.glob('*.json') if path.name != 'schema.json'}
    assert not {name: errors for name, errors in failures.items() if errors}
