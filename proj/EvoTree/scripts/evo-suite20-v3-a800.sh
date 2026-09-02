#!/usr/bin/env bash
set -euo pipefail
umask 077

BASE=/data/songweishuai/evotree_m2_baseline20_v3_20260903
OUT=/data/songweishuai/evotree_m2_20tasks_v3_20260903
REPO=/data/songweishuai/projs/rh/EvoTree/proj/EvoTree
RUNNER=/data/songweishuai/projs/rh/OSWorld-mcp-run
TASK="$REPO/experiments/m2_small_eval/tasks_20.json"
PYTHON=/data/songweishuai/envs/evo-osworld-20260829/bin/python
CONFIG=/data/songweishuai/projs/rh/cofig/model.md
SOCKET=unix:///data/songweishuai/evo-podman.sock
LABEL=20tasks-v3-20260903
BASE_CONTAINER=evo-m2-baseline20-v3-20260903
ONLINE_CONTAINER=evo-m2-20tasks-v3-20260903-online
FROZEN_CONTAINER=evo-m2-20tasks-v3-20260903-frozen

stage() {
  printf '%s\n' "$1" > "$OUT.stage"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$1"
}

require_exact_container_absent() {
  local name="$1"
  local inspect_output
  if inspect_output=$(podman --remote --url "$SOCKET" inspect "$name" --format '{{.State.Status}}' 2>&1); then
    printf 'refusing to start while exact container exists: %s status=%s\n' "$name" "$inspect_output" >&2
    return 1
  fi
  case "$inspect_output" in
    *"no such object"*|*"no container with name or ID"*) return 0 ;;
    *) printf 'container inspect failed for %s: %s\n' "$name" "$inspect_output" >&2; return 1 ;;
  esac
}

check_exact_container_gone() {
  local name="$1"
  local inspect_output status
  if ! inspect_output=$(podman --remote --url "$SOCKET" inspect "$name" --format '{{.State.Status}}' 2>&1); then
    case "$inspect_output" in
      *"no such object"*|*"no container with name or ID"*) return 0 ;;
      *) printf 'container inspect failed for %s: %s\n' "$name" "$inspect_output" >&2; return 1 ;;
    esac
  fi
  status=$(printf '%s' "$inspect_output" | tr -d '\r\n')
  if [ "$status" != exited ] && [ "$status" != stopped ]; then
    printf 'exact experiment container remains active: %s status=%s\n' "$name" "$status" >&2
    return 1
  fi
  podman --remote --url "$SOCKET" rm "$name" >/dev/null
  require_exact_container_absent "$name"
}

if [ -e "$BASE" ] || [ -e "$OUT" ] || [ -e "$OUT.stage" ]; then
  printf 'refusing to overwrite an existing v3 experiment output\n' >&2
  exit 2
fi
for name in "$BASE_CONTAINER" "$ONLINE_CONTAINER" "$FROZEN_CONTAINER"; do
  require_exact_container_absent "$name"
done
if [ ! -x "$PYTHON" ] || [ ! -f "$TASK" ]; then
  printf 'runtime input missing\n' >&2
  exit 3
fi

mkdir -p "$BASE" "$OUT/empty-seed"
ln -s "$BASE" "$OUT/baseline"

export DOCKER_HOST="$SOCKET"
export HF_ENDPOINT=https://hf-mirror.com
export OSWORLD_MCP_DEP_TIMEOUT=900
export OSWORLD_MCP_WHEEL_URL=http://20.20.20.1:18765/
export OSWORLD_PODMAN=1
export OSWORLD_MCP_INJECT=1
export PYTHONPATH=/data/songweishuai/envs/osworld/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
export OSWORLD_MODEL=${OSWORLD_MODEL:-gpt-5.6-terra}
export OSWORLD_API_URL=${OSWORLD_API_URL:-http://llmapi.evolventapi.com/v1}
if [ -z "${OSWORLD_API_KEY:-}" ]; then
  OSWORLD_API_KEY=$(awk 'NR == 1 { sub(/^[^:=]*[:=][[:space:]]*/, ""); sub(/\r$/, ""); print; exit }' "$CONFIG")
  export OSWORLD_API_KEY
fi
if [ -z "${OSWORLD_API_KEY:-}" ]; then
  stage api-key-missing
  exit 4
fi

cd "$RUNNER"
stage baseline-running
export EVO_CONTAINER_NAME="$BASE_CONTAINER"
set +e
timeout 21600s "$PYTHON" "$RUNNER/run_multienv_e2e.py" \
  --headless --action_space mcp --observation_type screenshot \
  --model "$OSWORLD_MODEL" --api_url "$OSWORLD_API_URL" \
  --max_steps 8 --max_trajectory_length 8 \
  --test_all_meta_path "$TASK" \
  --test_config_base_dir "$RUNNER/evaluation_examples" \
  --result_dir "$BASE/results" --num_envs 1 --trial-id 0 --domain os \
  >"$BASE/runner.log" 2>&1
BASE_RC=$?
set -e
printf '%s\n' "$BASE_RC" > "$BASE/returncode"
if [ "$BASE_RC" -ne 0 ]; then
  stage "baseline-failed-$BASE_RC"
  exit "$BASE_RC"
fi
if ! check_exact_container_gone "$BASE_CONTAINER"; then
  stage baseline-container-cleanup-failed
  exit 11
fi

cd "$REPO"
stage online-running
set +e
timeout 21600s "$PYTHON" -m experiments.m2_small_eval.run_arm \
  --arm online --runner-root "$RUNNER" --output-root "$OUT" \
  --seed-memory "$OUT/empty-seed" --task-manifest "$TASK" \
  --evotree-root "$REPO" --python "$PYTHON" \
  --model "$OSWORLD_MODEL" --api-url "$OSWORLD_API_URL" \
  --max-steps 8 --label "$LABEL" \
  >"$OUT/online-runner.log" 2>&1
ONLINE_RC=$?
set -e
printf '%s\n' "$ONLINE_RC" > "$OUT/online.returncode"
if [ "$ONLINE_RC" -ne 0 ]; then
  stage "online-failed-$ONLINE_RC"
  exit "$ONLINE_RC"
fi
if ! check_exact_container_gone "$ONLINE_CONTAINER"; then
  stage online-container-cleanup-failed
  exit 21
fi

stage frozen-running
set +e
timeout 21600s "$PYTHON" -m experiments.m2_small_eval.run_arm \
  --arm frozen --runner-root "$RUNNER" --output-root "$OUT" \
  --seed-memory "$OUT/online/memory/env-0" --task-manifest "$TASK" \
  --evotree-root "$REPO" --python "$PYTHON" \
  --model "$OSWORLD_MODEL" --api-url "$OSWORLD_API_URL" \
  --max-steps 8 --label "$LABEL" \
  >"$OUT/frozen-runner.log" 2>&1
FROZEN_RC=$?
set -e
printf '%s\n' "$FROZEN_RC" > "$OUT/frozen.returncode"
if [ "$FROZEN_RC" -ne 0 ]; then
  stage "frozen-failed-$FROZEN_RC"
  exit "$FROZEN_RC"
fi
if ! check_exact_container_gone "$FROZEN_CONTAINER"; then
  stage frozen-container-cleanup-failed
  exit 31
fi

stage complete
