"""Source releases contain publishable working-tree files reproducibly."""

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'package_release.py'


def load_packager():
    spec = importlib.util.spec_from_file_location('package_release', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_tree(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    (root / 'pyproject.toml').write_text('[project]\nname = "zbot-rl-mjlab"\nversion = "0.2.0"\n')
    files = {
        'README.md': b'release',
        'src/cards/untracked.json': b'{"new": true}',
        'assets/robot.stl': b'new asset',
        'bests/run/model.pt': b'checkpoint\x00',
        'bests/run/SHA256SUMS': b'original checksums\n',
        'bests/run/source_snapshot.tar.gz': b'opaque original snapshot',
        'train.sh': b'#!/bin/sh\ntrue\n',
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / 'train.sh').chmod(0o755)
    return root, files


def test_allowlist_manifest_and_preserved_bytes(tmp_path):
    root, files = source_tree(tmp_path)
    unwanted = [
        '.env', '.git/config', '.venv/token', '.omx/state.json', 'private.txt',
        'logs/model.pt', 'outputs/result.txt', 'work/study.txt',
        'src/__pycache__/x.pyc', 'src/x.pyo', 'src/pkg.egg-info/PKG-INFO',
        'tests/.pytest_cache/data', 'src/.ruff_cache/data', 'assets/.git/config',
        'bests/build/temp', 'bests/dist/temp', 'src/logs/temp', 'assets/MUJOCO_LOG.TXT',
        'src/.env', 'src/.env.production', 'docs/.env.local',
        'scripts/id_rsa', 'scripts/id_ed25519', 'assets/.ssh/config',
    ]
    for name in unwanted:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('local secret or generated output')
    output = load_packager().package_release(root)
    assert output == root / 'dist/zbot-rl-mjlab-0.2.0-source.tar.gz'
    with tarfile.open(output) as archive:
        members = {m.name: m for m in archive.getmembers()}
        prefix = 'zbot-rl-mjlab-0.2.0/'
        assert set(members) == {prefix + name for name in files} | {
            prefix + 'pyproject.toml', prefix + 'RELEASE_MANIFEST.json',
        }
        stream = archive.extractfile(prefix + 'RELEASE_MANIFEST.json')
        assert stream is not None
        manifest = json.load(stream)
        assert manifest['version'] == '0.2.0'
        assert set(manifest) == {'version', 'files'}
        assert [entry['path'] for entry in manifest['files']] == sorted([*files, 'pyproject.toml'])
        for entry in manifest['files']:
            expected = (root / entry['path']).read_bytes()
            stream = archive.extractfile(prefix + entry['path'])
            assert stream is not None and stream.read() == expected
            assert entry['size'] == len(expected)
            assert entry['sha256'] == hashlib.sha256(expected).hexdigest()
        for member in members.values():
            assert (member.uid, member.gid, member.uname, member.gname, member.mtime) == (0, 0, '', '', 0)
        assert members[prefix + 'train.sh'].mode == 0o755
        assert members[prefix + 'README.md'].mode == 0o644


def test_builds_are_deterministic_and_skip_output_inside_inputs(tmp_path):
    root, _ = source_tree(tmp_path)
    module = load_packager()
    output = root / 'scripts/share.tar.gz'
    module.package_release(root, output)
    first = output.read_bytes()
    (root / 'README.md').touch()
    module.package_release(root, output)
    assert output.read_bytes() == first
    second = tmp_path / 'second.tar.gz'
    # The prior output is not release source; remove it before changing output paths.
    output.unlink()
    module.package_release(root, second)
    assert second.read_bytes() == first


@pytest.mark.parametrize('selected', ['src/linked', 'assets'])
def test_selected_symlinks_are_rejected_before_writing(tmp_path, selected):
    root, _ = source_tree(tmp_path)
    target = tmp_path / 'outside'
    target.mkdir()
    link = root / selected
    if link.exists():
        for child in link.iterdir():
            child.unlink()
        link.rmdir()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    output = tmp_path / 'release.tar.gz'
    with pytest.raises(ValueError, match='[Ss]ymlink'):
        load_packager().package_release(root, output)
    assert not output.exists()


@pytest.mark.parametrize('name', ['README.md', 'bests/run/model.pt', 'bests/release.tar.gz'])
def test_output_cannot_replace_source_or_bests(tmp_path, name):
    root, files = source_tree(tmp_path)
    with pytest.raises(ValueError, match='[Rr]elease output'):
        load_packager().package_release(root, root / name)
    for relative, expected in files.items():
        assert (root / relative).read_bytes() == expected


def test_version_comes_from_project_section(tmp_path):
    root, _ = source_tree(tmp_path)
    (root / 'pyproject.toml').write_text(
        '[build-system]\nversion = "wrong"\n[project]\nversion = "0.3.1" # release\n'
        '[tool.example]\nversion = "also-wrong"\n',
    )
    output = load_packager().package_release(root)
    assert output.name == 'zbot-rl-mjlab-0.3.1-source.tar.gz'
    with tarfile.open(output) as archive:
        stream = archive.extractfile('zbot-rl-mjlab-0.3.1/RELEASE_MANIFEST.json')
        assert stream is not None
        assert json.load(stream)['version'] == '0.3.1'


def test_output_parent_symlink_cannot_overwrite_bests(tmp_path):
    root, files = source_tree(tmp_path)
    link = tmp_path / 'linked-output'
    link.symlink_to(root / 'bests/run', target_is_directory=True)
    with pytest.raises(ValueError, match='preserved bests'):
        load_packager().package_release(root, link / 'source_snapshot.tar.gz')
    for relative, expected in files.items():
        assert (root / relative).read_bytes() == expected


def test_output_file_symlink_is_rejected(tmp_path):
    root, _ = source_tree(tmp_path)
    output = tmp_path / 'linked-output.tar.gz'
    original = root / 'bests/run/source_snapshot.tar.gz'
    before = original.read_bytes()
    output.symlink_to(original)
    with pytest.raises(ValueError, match='Symlink'):
        load_packager().package_release(root, output)
    assert original.read_bytes() == before
