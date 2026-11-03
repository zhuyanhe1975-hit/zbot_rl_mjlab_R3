#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MJLAB_ROOT="${MJLAB_ROOT:-$HOME/AI/mjlab}"
PYTHON_BIN="${ZBOT_PYTHON:-$MJLAB_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -n "${ZBOT_PYTHON:-}" ]]; then
    echo "ZBOT_PYTHON is not executable: $ZBOT_PYTHON" >&2
    exit 1
  fi
  PYTHON_BIN="${ROBOTICS_ENV:-$HOME/dev-envs/robotics-mjlab-py313}/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "No MJLab environment found. See docs/development-environment.md" >&2
    exit 1
  fi
fi
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$@"
