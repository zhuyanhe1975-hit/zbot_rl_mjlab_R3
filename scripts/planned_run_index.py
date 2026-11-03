"""Track the most recently started planned or full-polish execution."""
import json
import os
import tempfile
from pathlib import Path

INDEX = Path('outputs/reconfiguration/latest_execution.json')
ROOTS = (Path('outputs/reconfiguration/planned_runs'), Path('outputs/reconfiguration/full_polish'))


def record_execution(directory, kind):
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    record = {'directory': str(Path(directory).resolve()), 'kind': kind}
    fd, name = tempfile.mkstemp(prefix='.latest-execution-', dir=INDEX.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream)
            stream.write('\n')
        os.replace(name, INDEX)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def latest_execution():
    if INDEX.exists():
        record = json.loads(INDEX.read_text())
        return Path(record['directory'])
    # Older runs predate the index and use sortable timestamp directory names.
    directories = [p for root in ROOTS for p in root.glob('*') if p.is_dir()]
    if not directories:
        raise FileNotFoundError('no run found; run ./run_planned_mjbatch.sh or ./polish_mjbatch.sh first')
    return max(directories, key=lambda p: p.name)
