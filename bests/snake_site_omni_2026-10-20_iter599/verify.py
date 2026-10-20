"""Re-run strict site-snake qualification without changing archived evidence."""
import subprocess
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent
root = folder.parents[1]
output = root / 'outputs' / 'archive_verification' / (folder.name + '.json')
if __name__ == '__main__':
    raise SystemExit(subprocess.call([
        sys.executable, str(root / 'scripts' / 'evaluate_site_snake.py'),
        '--checkpoint', str(folder / 'model_599.pt'), '--output', str(output),
        *sys.argv[1:],
    ], cwd=root))
