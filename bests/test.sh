#!/usr/bin/env bash
# 在 MJLab 查看器中播放已归档模型。
#   ./bests/test.sh                      列出已归档模型
#   ./bests/test.sh <dir-or-index> ...   播放指定模型，其余参数传给 play.py
# 示例： ./bests/test.sh 1 --target-speed 0.3
set -euo pipefail
BESTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$BESTS_DIR")"

task_of() {
  case "$1" in
    universal_student_disturbed_*) echo "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student-Disturbed" ;;
    universal_student_clean_*) echo "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Student" ;;
    universal_three_teacher_gaussian_mirror_*) echo "Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher-Guided" ;;
    velocity_frequency_tracking_*) echo "Mjlab-Zbot-6dof-Velocity-Frequency-Tracking-Walking" ;;
    periodic_stepping_*) echo "Mjlab-Zbot-6dof-Periodic-Stepping" ;;
    snake_site_omni_*) echo "Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher" ;;
    snake_world_x_*) echo "Mjlab-Zbot-6dof-Snake-Omni-Teacher" ;;
    transform_snake_biped_bidirectional_*) echo "Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional" ;;
    transform_snake_to_biped_teacher_*) echo "Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher" ;;
    transform_snake_to_biped_beta_*) echo "Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta" ;;
    transform_biped_to_snake_*) echo "Mjlab-Zbot-6dof-Transform-Biped-To-Snake" ;;
    deploy_walking_*) echo "Mjlab-Zbot-6dof-Deploy-Walking" ;;
    flat_foot_velocity_energy_*) echo "Mjlab-Zbot-6dof-Flat-Foot-Velocity-Energy-Walking" ;;
    omni_velocity_energy_*) echo "Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking" ;;
    velocity_energy_disturbed_*) echo "Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking" ;;
    velocity_energy_tracking_*) echo "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking" ;;
    zbot_disturbed_walking|zbot_disturbed_walking_*) echo "Mjlab-Zbot-6dof-Disturbed-Walking" ;;
    zbot_walking|zbot_walking_*) echo "Mjlab-Zbot-6dof-Walking" ;;
    *) echo "无法识别 '$1' 的任务，请在 $0 的 task_of 中补充映射" >&2; return 1 ;;
  esac
}

mapfile -t entries < <(cd "$BESTS_DIR" && ls -d */ | tr -d /)
if [[ $# -eq 0 ]]; then
  for i in "${!entries[@]}"; do echo "$((i + 1)) ${entries[$i]}"; done
  exit 0
fi

pick="${1%/}"
shift
if [[ "$pick" =~ ^[0-9]+$ ]]; then
  pick="${entries[$((pick - 1))]:?索引不存在}"
fi
checkpoint="$(ls "$BESTS_DIR/$pick"/model_*.pt | head -n1)"

playback_args=()
if [[ -f "$BESTS_DIR/$pick/playback_args.json" ]]; then
  mapfile -t playback_args < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))))' "$BESTS_DIR/$pick/playback_args.json")
fi

exec "$PROJECT_DIR/scripts/python.sh" "$PROJECT_DIR/play.py" "$(task_of "$pick")" \
  --checkpoint-file "$checkpoint" "${playback_args[@]}" "$@"
