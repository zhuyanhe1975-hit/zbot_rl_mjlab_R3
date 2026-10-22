"""Archived recipe differences must be visible in the owning experiment card."""

import json
from pathlib import Path

import pytest
import yaml
from test_bests import ENTRIES, _Loose, archived_env, task_of

import zbot_rl_mjlab
from zbot_rl_mjlab.experiments.cards import load_card

CARDS = Path(zbot_rl_mjlab.__file__).parent / 'experiments' / 'cards'
EXPERIMENTS = {cls.task_id: cls for cls in zbot_rl_mjlab.EXPERIMENTS}
# Runtime identities/state and callback/model classes are not card parameters.
IGNORED = {
    'class_name', 'func', 'spec_fn', 'obs_groups', 'experiment_name',
    'run_name', 'resume', 'load_run', 'load_checkpoint',
    'reward_hot_reload_path',
}


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in IGNORED:
                yield from leaves(item, (*path, key))
    else:
        yield path, value


def lookup(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


@pytest.mark.parametrize('entry', ENTRIES, ids=lambda path: path.name)
def test_archived_parent_differences_are_explicit_in_own_card(entry):
    experiment = EXPERIMENTS[task_of(entry)]
    card_name = experiment.card_name
    assert card_name is not None
    card = json.loads((CARDS / f'{card_name}.json').read_text())
    parent = load_card(card['inherits'])
    resolved = load_card(card_name)
    assert entry.name in card['description']
    saved_agent = yaml.load((entry / 'agent.yaml').read_text(), Loader=_Loose)
    saved = archived_env(entry)
    missing = []
    for section, archived in (
        ('agent', saved_agent),
        ('env', {key: value for key, value in saved.items() if key in parent.get('env', {})}),
        ('rewards', {
            key: value for key, value in saved['rewards'].items()
            if key in resolved.get('rewards', {})
        }),
    ):
        for path, value in leaves(archived):
            # Scene internals, selectors, sequences and structural fields are
            # audited separately; here check editable scalar overrides.
            if section == 'env' and path[0] in {'scene', 'events', 'actions', 'commands', 'sim', 'viewer'}:
                continue
            if isinstance(value, (list, tuple)):
                continue
            exists, inherited = lookup(parent.get(section, {}), path)
            supported, _ = lookup(resolved.get(section, {}), path)
            if not exists or not supported or value == inherited:
                continue
            explicit, _ = lookup(card.get(section, {}), path)
            if not explicit:
                missing.append(f"{section}.{'.'.join(path)}")
    assert not missing, f'{entry.name}: inherited archive differences {missing}'
