#!/usr/bin/env bash
# Helper to launch an Eval VLA recording run using the same global SO101_*
# environment variables as the demo. Adjust the env vars before running if
# your ports or cameras differ.

set -euo pipefail

# Robot + camera defaults (override via env).
SO101_PORT=${SO101_PORT:-/dev/ttyACM0}               # follower on port 0
SO101_ROBOT_ID=${SO101_ROBOT_ID:-my_awesome_follower_arm}
SO101_CAMERA_SOURCES=${SO101_CAMERA_SOURCES:-/dev/video6,/dev/video2}
SO101_CAMERA_NAMES=${SO101_CAMERA_NAMES:-image,image2}
SO101_CAMERA_WIDTH=${SO101_CAMERA_WIDTH:-640}
SO101_CAMERA_HEIGHT=${SO101_CAMERA_HEIGHT:-480}
SO101_CAMERA_FPS=${SO101_CAMERA_FPS:-30}

# Eval parameters (override via env to change task/policy/episodes, etc).
EVAL_TASK=${EVAL_TASK:-"Stack cups"}
EVAL_REPO_ID=${EVAL_REPO_ID:-"devsheroubi/eval_xvlastack_$(date +%Y%m%d_%H%M%S)"}
EVAL_EPISODE_TIME_S=${EVAL_EPISODE_TIME_S:-50}
EVAL_NUM_EPISODES=${EVAL_NUM_EPISODES:-5}
EVAL_POLICY_PATH=${EVAL_POLICY_PATH:-"Gowshigan/stackcupsv5"}
EVAL_DATASET_ROOT=${EVAL_DATASET_ROOT:-}
EVAL_RENAME_MAP=${EVAL_RENAME_MAP:-'{"observation.images.mount": "observation.images.image", "observation.images.front": "observation.images.image2"}'}

IFS=',' read -ra CAMERA_SRCS <<<"$SO101_CAMERA_SOURCES"
IFS=',' read -ra CAMERA_NAMES <<<"$SO101_CAMERA_NAMES"

if [ ${#CAMERA_SRCS[@]} -eq 0 ]; then
  echo "No cameras configured. Set SO101_CAMERA_SOURCES." >&2
  exit 1
fi

# Build the --robot.cameras flag from the env-provided sources/names.
CAMERA_ENTRIES=()
for idx in "${!CAMERA_SRCS[@]}"; do
  src="${CAMERA_SRCS[$idx]}"
  name="${CAMERA_NAMES[$idx]:-camera_$idx}"
  if [[ $src =~ ^[0-9]+$ ]]; then
    src_val=$src
  else
    src_val="\"$src\""
  fi
  CAMERA_ENTRIES+=("${name}: {type: opencv, index_or_path: ${src_val}, width: ${SO101_CAMERA_WIDTH}, height: ${SO101_CAMERA_HEIGHT}, fps: ${SO101_CAMERA_FPS}}")
done
CAMERA_FLAG="{ $(IFS=', '; echo "${CAMERA_ENTRIES[*]}") }"

cmd=(
  lerobot-record
  --robot.type=so101_follower
  "--robot.port=${SO101_PORT}"
  "--robot.id=${SO101_ROBOT_ID}"
  "--robot.cameras=${CAMERA_FLAG}"
  "--dataset.single_task=${EVAL_TASK}"
  "--dataset.repo_id=${EVAL_REPO_ID}"
  "--dataset.episode_time_s=${EVAL_EPISODE_TIME_S}"
  "--dataset.num_episodes=${EVAL_NUM_EPISODES}"
  "--policy.path=${EVAL_POLICY_PATH}"
  "--dataset.rename_map=${EVAL_RENAME_MAP}"
)

if [ -n "$EVAL_DATASET_ROOT" ]; then
  cmd+=("--dataset.root=${EVAL_DATASET_ROOT}")
fi

echo "Launching Eval VLA with:"
printf '  %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
