#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ZBOT_PYTHON="${ZBOT_MJBATCH_PYTHON:-/home/yhzhu/dev-envs/robotics-mjbatch-py313/bin/python}"
exec "$project_root/scripts/python.sh" "$@"
