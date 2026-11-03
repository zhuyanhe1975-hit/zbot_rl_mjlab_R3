#!/usr/bin/env bash
# Source this file to activate the existing runtime; never creates an environment.
_zbot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_zbot_python="$("$_zbot_root/scripts/python.sh" -c 'import sys; print(sys.executable)')" || return 1
source "$(dirname -- "$_zbot_python")/activate" || return 1
export PYTHONPATH="$_zbot_root/src:$_zbot_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$_zbot_root" || return 1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
unset _zbot_root _zbot_python
