#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_root/scripts/python_mjbatch.sh" "$project_root/train_mjbatch.py" --num-envs 256 --iterations 200 "$@"
