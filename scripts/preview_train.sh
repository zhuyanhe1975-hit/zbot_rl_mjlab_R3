#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ZBOT_LOG_ROOT:-$PROJECT_DIR/logs/rsl_rl}"
TASK_ID="${ZBOT_TASK_ID:-Mjlab-Zbot-6dof-Periodic-Stepping}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
NUM_ENVS="${ZBOT_NUM_ENVS:-1}"
VIEWER="${VIEWER:-native}"
if [[ "${1:-}" == "--task" ]]; then
  case "${2:-}" in
    stepping) TASK_ID="Mjlab-Zbot-6dof-Periodic-Stepping" ;;
    walking) TASK_ID="Mjlab-Zbot-6dof-Walking" ;;
    "Mjlab-Zbot-6dof-Periodic-Stepping"|"Mjlab-Zbot-6dof-Walking") TASK_ID="$2" ;;
    *) echo "未知任务 '$2'（请选择 stepping 或 walking）" >&2; exit 2 ;;
  esac
  shift 2
fi
if [[ "$TASK_ID" == "Mjlab-Zbot-6dof-Walking" ]]; then
  EXPERIMENT="zbot_walking"
else
  EXPERIMENT="zbot_periodic_stepping"
fi
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  NUM_ENVS="$1"
  shift
elif [[ "${1:-}" == "--num-envs" ]]; then
  NUM_ENVS="${2:-}"
  shift 2
fi
if [[ ! "$NUM_ENVS" =~ ^[1-9][0-9]*$ ]]; then
  echo "环境数必须是正整数" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
用法：./scripts/preview_train.sh [--task stepping|walking] [NUM_ENVS]
       ./scripts/preview_train.sh [--task TASK] --num-envs NUM_ENVS

打开最新模型的查看器，并在训练保存新模型后刷新。
每个环境独立采样迈步频率。
HELP
  exit 0
fi
mkdir -p "$PROJECT_DIR/outputs"
last_checkpoint=""
viewer_pid=""
seen_training=0
training_running() {
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill -0 "$TRAIN_PID" 2>/dev/null
  else
    pgrep -f 'mjlab\.scripts\.train|zbot_rl_mjlab.*train' >/dev/null 2>&1
  fi
}
latest_checkpoint() {
  find "$LOG_ROOT/$EXPERIMENT" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n1 | cut -d' ' -f2- || true
}
launch_viewer() {
  local checkpoint="$1"
  if [[ -n "$viewer_pid" ]] && kill -0 "$viewer_pid" 2>/dev/null; then
    echo "正在刷新查看器： $checkpoint"
    kill "$viewer_pid" 2>/dev/null || true
    wait "$viewer_pid" 2>/dev/null || true
  else
    echo "正在打开查看器： $checkpoint"
  fi
  ZBOT_TASK_ID="$TASK_ID" ZBOT_LOG_ROOT="$LOG_ROOT" \
    "$PROJECT_DIR/scripts/python.sh" "$PROJECT_DIR/play.py" "$TASK_ID" --env.scene.num-envs "$NUM_ENVS" --checkpoint-file "$checkpoint" --viewer "$VIEWER" \
    >"$PROJECT_DIR/outputs/preview-viewer.log" 2>&1 &
  viewer_pid=$!
}
echo "正在监测 $LOG_ROOT/$EXPERIMENT 中的 $TASK_ID 模型"
while true; do
  running=0
  if training_running; then
    seen_training=1
    running=1
  fi
  checkpoint="$(latest_checkpoint)"
  if [[ -n "$checkpoint" && "$checkpoint" != "$last_checkpoint" && -s "$checkpoint" ]]; then
    before="$(stat -c '%s' "$checkpoint")"
    sleep 1
    after="$(stat -c '%s' "$checkpoint")"
    if [[ "$before" == "$after" ]]; then
      launch_viewer "$checkpoint"
      last_checkpoint="$checkpoint"
    fi
  fi
  if (( seen_training )) && (( ! running )) && [[ -z "$checkpoint" || "$checkpoint" == "$last_checkpoint" ]]; then
    echo "训练结束，已将最新模型交给查看器。"
    if [[ -n "$viewer_pid" ]] && kill -0 "$viewer_pid" 2>/dev/null; then
      wait "$viewer_pid" 2>/dev/null || true
    fi
    exit 0
  fi
  sleep "$POLL_INTERVAL"
done
