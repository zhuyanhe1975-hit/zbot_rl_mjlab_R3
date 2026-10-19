#!/usr/bin/env bash
# 左侧原生 MuJoCo，右侧 MJLab；其余参数传给对比工具。
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/scripts/python.sh" "$PROJECT_DIR/scripts/viewer_compare.py" \
  --task Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking \
  --target-speed 0.4 --force-scale 0 --torque-scale 0 "$@"
