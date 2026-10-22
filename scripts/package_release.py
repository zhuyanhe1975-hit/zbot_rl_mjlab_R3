#!/usr/bin/env python3
"""Build a reproducible source release, including assets and archived best models.

This packages the current working tree, including untracked publishable files.
The result supports editable installation in an existing MJLab environment; it
is not a self-contained runtime or a wheel.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

ROOT_FILES = (
    'README.md', 'experiments.md', 'pyproject.toml', 'uv.lock', '.gitignore',
    '.gitattributes', 'train.py', 'play.py', 'train.sh', 'run.sh', 'sim2sim.sh',
)
ROOT_DIRECTORIES = ('src', 'assets', 'scripts', 'tests', 'docs', 'bests')
EXCLUDED_NAMES = {
    '__pycache__', '.pytest_cache', '.ruff_cache', '.git', '.omx', '.venv',
    'logs', 'outputs', 'work', 'build', 'dist', 'MUJOCO_LOG.TXT',
    '.ssh', '.env', 'id_rsa', 'id_ed25519',
}


def project_version(root: Path) -> str:
    """Read the simple static project version without requiring Python 3.11 TOML."""
    text = (root / 'pyproject.toml').read_text(encoding='utf-8')
    project = re.search(r'^\[project\]\s*\n(.*?)(?=^\[|\Z)', text, re.MULTILINE | re.DOTALL)
    match = None if project is None else re.search(
        r'''^version\s*=\s*["']([A-Za-z0-9][A-Za-z0-9._+\-]*)["']\s*(?:\#.*)?$''',
        project.group(1), re.MULTILINE,
    )
    if match is None:
        raise ValueError('pyproject.toml must define a static [project] version')
    return match.group(1)


def excluded(path: Path) -> bool:
    return (
        path.name in EXCLUDED_NAMES
        or path.name.startswith('.env.')
        or path.name.endswith(('.pyc', '.pyo', '.egg-info'))
    )


def source_files(root: Path, output: Path) -> list[Path]:
    files: list[Path] = []

    def visit(path: Path) -> None:
        if excluded(path):
            return
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed in release inputs: {path.relative_to(root)}')
        if path == output:
            return
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child)
        elif path.is_file():
            files.append(path)
        elif path.exists():
            raise ValueError(f'Unsupported release input: {path.relative_to(root)}')

    for name in (*ROOT_FILES, *ROOT_DIRECTORIES):
        visit(root / name)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    # TarInfo defaults normalize uid/gid, owner/group names and mtime to zero/empty.
    archive.addfile(info, io.BytesIO(data))


def package_release(root: Path, output: Path | None = None) -> Path:
    root = root.resolve()
    version = project_version(root)
    prefix = f'zbot-rl-mjlab-{version}'
    output = root / 'dist' / f'{prefix}-source.tar.gz' if output is None else Path(os.path.abspath(output))
    if output.is_symlink():
        raise ValueError('Symlink is not allowed as release output')
    output = output.parent.resolve() / output.name
    if output in (root / name for name in ROOT_FILES):
        raise ValueError('Release output cannot overwrite a root source file')
    if root / 'bests' in output.parents:
        raise ValueError('Release output cannot be placed inside preserved bests')
    if not output.name.endswith('.tar.gz'):
        raise ValueError('Release output must have a .tar.gz suffix')
    files = source_files(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the destination and replace only after the complete archive exists.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as raw:
            temporary = Path(raw.name)
            with (
                gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed,
                tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive,
            ):
                entries = []
                for path in files:
                    relative = path.relative_to(root).as_posix()
                    data = path.read_bytes()
                    mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
                    add_bytes(archive, f'{prefix}/{relative}', data, mode)
                    entries.append({
                        'path': relative, 'size': len(data),
                        'sha256': hashlib.sha256(data).hexdigest(),
                    })
                manifest = json.dumps(
                    {'version': version, 'files': entries}, indent=2, sort_keys=True,
                ).encode('utf-8') + b'\n'
                add_bytes(archive, f'{prefix}/RELEASE_MANIFEST.json', manifest)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Destination .tar.gz (default: dist/<name>-source.tar.gz)')
    args = parser.parse_args()
    try:
        output = package_release(args.root, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Cannot package release: {error}\n')
    print(output)


if __name__ == '__main__':
    main()
