"""Load editable experiment settings without changing runtime implementations."""
# Malformed user configuration consistently raises ValueError at this API boundary.
# ruff: noqa: TRY004

import json
import math
import re
import types
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

_BLOCKED = {'func', 'spec_fn', 'class_name', 'obs_groups'}
_ROOT_KEYS = {'$schema', 'schema_version', 'task_id', 'title', 'description', 'inherits', 'env', 'rewards', 'agent', 'addons', 'train', 'play'}


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    removed = override.get('$remove', [])
    if not isinstance(removed, list) or any(not isinstance(key, str) for key in removed):
        raise ValueError('$remove must be a list of field names')
    for key in removed:
        result.pop(key, None)
    for key, value in override.items():
        if key != '$remove':
            if isinstance(value, dict):
                previous = result.get(key)
                result[key] = _merge(previous if isinstance(previous, dict) else {}, value)
            else:
                result[key] = deepcopy(value)
    return result


def _validate_numbers(value: Any, path: str) -> None:
    if callable(value):
        raise ValueError(f'{path}: callables cannot be configured')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{path}: numbers must be finite')
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _BLOCKED:
                raise ValueError(f'{path}.{key}: structural fields cannot be configured')
            _validate_numbers(item, f'{path}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_numbers(item, f'{path}[{index}]')


def _validate_rewards(rewards: Any, path: str) -> None:
    if not isinstance(rewards, dict):
        raise ValueError(f'{path} must be an object')
    for name, settings in rewards.items():
        if not isinstance(settings, dict) or settings.keys() - {'weight', 'params'}:
            raise ValueError(f'{path}.{name}: only weight and params may be configured')
        if 'weight' in settings and (type(settings['weight']) not in (int, float)):
            raise ValueError(f'{path}.{name}.weight must be numeric')
        if 'params' in settings and not isinstance(settings['params'], dict):
            raise ValueError(f'{path}.{name}.params must be an object')


def _validate_card(card: dict) -> None:
    if card.keys() - _ROOT_KEYS:
        raise ValueError(f'unknown card fields: {sorted(card.keys() - _ROOT_KEYS)}')
    if type(card.get('schema_version')) is not int or card['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    for key in ('task_id', 'title', 'description', '$schema'):
        if key in card and not isinstance(card[key], str):
            raise ValueError(f'{key} must be a string')
    for key in ('env', 'agent', 'addons', 'train', 'play'):
        if key in card and not isinstance(card[key], dict):
            raise ValueError(f'{key} must be an object')
    if any(not isinstance(settings, dict) for settings in card.get('addons', {}).values()):
        raise ValueError('addons entries must be objects')
    _validate_rewards(card.get('rewards', {}), 'rewards')
    for mode in ('train', 'play'):
        overrides = card.get(mode, {})
        if overrides.keys() - {'env', 'rewards'}:
            raise ValueError(f'{mode} may contain only env and rewards')
        if not isinstance(overrides.get('env', {}), dict):
            raise ValueError(f'{mode}.env must be an object')
        _validate_rewards(overrides.get('rewards', {}), f'{mode}.rewards')
    _validate_numbers(card, 'card')


def load_card(name: str, directory: Path | None = None) -> dict:
    """Resolve same-directory inheritance afresh; each caller owns its result."""
    directory = Path(directory) if directory is not None else Path(__file__).parent

    def read(filename: str, stack: tuple[str, ...]) -> dict:
        if not isinstance(filename, str) or not re.fullmatch(r'[A-Za-z0-9_-]+(?:\.json)?', filename):
            raise ValueError(f'invalid card filename: {filename!r}')
        filename = filename if filename.endswith('.json') else filename + '.json'
        if filename in stack:
            raise ValueError(f'card inheritance cycle: {" -> ".join((*stack, filename))}')
        try:
            card = json.loads((directory / filename).read_text(encoding='utf-8'))
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f'cannot read card {filename}: {error}') from error
        if not isinstance(card, dict):
            raise ValueError(f'card {filename} must be an object')
        parent = card.get('inherits')
        if 'inherits' in card and not isinstance(parent, str):
            raise ValueError(f'{filename}: inherits must be a filename')
        base = read(parent, (*stack, filename)) if parent is not None else {}
        return _merge(base, {key: value for key, value in card.items() if key != 'inherits'})

    card = read(name, ())
    _validate_card(card)
    return card


def _nullable(annotation: Any) -> bool:
    return get_origin(annotation) in (Union, types.UnionType) and type(None) in get_args(annotation)


def _patch(current: Any, patch: Any, path: str, annotation: Any = None) -> Any:
    if callable(current):
        raise ValueError(f'{path}: runtime callables cannot be configured')
    if patch is None:
        if current is None or _nullable(annotation):
            return None
        raise ValueError(f'{path}: null is not allowed')
    if _nullable(annotation):
        choices = tuple(item for item in get_args(annotation) if item is not type(None))
        annotation = choices[0] if len(choices) == 1 else annotation
    if current is None and get_origin(annotation) is tuple:
        current = ()
    if isinstance(patch, dict):
        dataclass = is_dataclass(current) and not isinstance(current, type)
        if not dataclass and not isinstance(current, dict):
            if current is None:
                return deepcopy(patch)
            raise ValueError(f'{path}: expected {type(current).__name__}, received object')
        target: Any = current
        try:
            annotations = get_type_hints(type(target)) if dataclass else {}
        except (NameError, TypeError):
            annotations = {field.name: field.type for field in fields(target)} if dataclass else {}
        known = {field.name for field in fields(target)} if dataclass else target.keys()
        for key, value in patch.items():
            if not dataclass and key not in known:
                key = next((existing for existing in known if type(existing) is int and str(existing) == key), key)
            child_path = f'{path}.{key}'
            if key in _BLOCKED:
                raise ValueError(f'{child_path}: structural fields cannot be configured')
            if key not in known:
                if isinstance(current, dict) and path.endswith('.params') and (value is None or type(value) in (str, bool, int, float)):
                    current[key] = deepcopy(value)
                    continue
                raise ValueError(f'{child_path}: unknown field')
            previous = getattr(target, key) if dataclass else target[key]
            item_annotation = annotations.get(key) if dataclass else get_args(annotation)[1] if get_origin(annotation) is dict else None
            updated = _patch(previous, value, child_path, item_annotation)
            if dataclass:
                setattr(target, key, updated)
            else:
                target[key] = updated
        return current
    if isinstance(current, (tuple, list)):
        if not isinstance(patch, (tuple, list)):
            raise ValueError(f'{path}: expected a sequence')
        arguments = get_args(annotation) if get_origin(annotation) is tuple else ()
        if arguments and arguments[-1] is not Ellipsis and len(patch) != len(arguments):
            raise ValueError(f'{path}: expected {len(arguments)} tuple items')
        if path.endswith('.hidden_dims') and (not patch or any(type(item) is not int or item <= 0 for item in patch)):
            raise ValueError(f'{path}: network dimensions must be positive integers')
        items = []
        for index, value in enumerate(patch):
            item_annotation = None
            if arguments:
                item_annotation = arguments[0] if arguments[-1] is Ellipsis else arguments[index]
            previous = current[min(index, len(current) - 1)] if current else None
            items.append(_patch(previous, value, f'{path}[{index}]', item_annotation))
        return tuple(items) if isinstance(current, tuple) else items
    if current is None:
        choices = tuple(item for item in get_args(annotation) if item is not type(None))
        expected = choices[0] if len(choices) == 1 else annotation
        if expected in (float, int, bool, str):
            return _patch(expected(), patch, path)
        return deepcopy(patch)
    if type(current) is float and type(patch) in (int, float):
        return float(patch)
    if type(current) is type(patch) and type(patch) in (str, int, bool):
        return patch
    raise ValueError(f'{path}: expected {type(current).__name__}, received {type(patch).__name__}')


def apply_environment(cfg: Any, card: dict, *, play: bool = False) -> None:
    """Apply common settings, followed by train/play overrides, to an environment."""
    _validate_card(card)
    overrides = card.get('play' if play else 'train', {})
    _patch(cfg, _merge(card.get('env', {}), overrides.get('env', {})), 'env')
    rewards = _merge(card.get('rewards', {}), overrides.get('rewards', {}))
    for name, settings in rewards.items():
        if name not in cfg.rewards:
            raise ValueError(f'rewards.{name}: unknown reward')
        _patch(cfg.rewards[name], settings, f'rewards.{name}')


def apply_agent(cfg: Any, card: dict) -> None:
    """Apply training settings while retaining registered algorithm/model paths."""
    _validate_card(card)
    _patch(cfg, card.get('agent', {}), 'agent')


def configure_addons(addons: list | tuple, card: dict, *, overrides: tuple = ()) -> list:
    """Replace frozen addon settings; explicit CLI replacement types take priority."""
    _validate_card(card)
    settings = card.get('addons', {})
    known = {type(addon).__name__ for addon in addons}
    if settings.keys() - known:
        raise ValueError(f'unknown addons: {sorted(settings.keys() - known)}')
    configured = []
    for addon in addons:
        name = type(addon).__name__
        values = settings.get(name, {})
        known_fields = {item.name for item in fields(addon)}
        if values.keys() - known_fields:
            raise ValueError(f'addons.{name}: unknown fields {sorted(values.keys() - known_fields)}')
        if type(addon) in overrides or not values:
            configured.append(addon)
            continue
        annotations = get_type_hints(type(addon))
        patched = {key: _patch(deepcopy(getattr(addon, key)), value, f'addons.{name}.{key}', annotations.get(key)) for key, value in values.items()}
        configured.append(replace(addon, **patched))
    return configured
