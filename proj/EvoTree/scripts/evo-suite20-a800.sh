#!/usr/bin/env bash
set -euo pipefail
umask 077

BASE=/data/songweishuai/evotree_m2_baseline20_20260902
BASE_PID=892087
OUT=/data/songweishuai/evotree_m2_20tasks_20260902
REPO=/data/songweishuai/projs/rh/EvoTree/proj/EvoTree
RUNNER=/data/songweishuai/projs/rh/OSWorld-mcp-run
TASK="$REPO/experiments/m2_small_eval/tasks_20.json"
PYTHON=/data/songweishuai/envs/evo-osworld-20260829/bin/python
CONFIG=/data/songweishuai/projs/rh/cofig/model.md
SOCKET=unix:///data/songweishuai/evo-podman.sock
LABEL=20tasks-20260902

stage() {
  printf '%s\n' "$1" > "$OUT.stage"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$1"
}

ensure_exact_container_absent() {
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
    printf 'refusing to remove active container %s status=%s\n' "$name" "$status" >&2
    return 1
  fi
  podman --remote --url "$SOCKET" rm "$name"
  if inspect_output=$(podman --remote --url "$SOCKET" inspect "$name" --format '{{.State.Status}}' 2>&1); then
    printf 'container still exists after exact removal: %s status=%s\n' "$name" "$inspect_output" >&2
    return 1
  fi
  case "$inspect_output" in
    *"no such object"*|*"no container with name or ID"*) return 0 ;;
    *) printf 'post-removal inspect failed for %s: %s\n' "$name" "$inspect_output" >&2; return 1 ;;
  esac
}

stage waiting-baseline
while kill -0 "$BASE_PID" 2>/dev/null; do
  sleep 30
done
if [ ! -f "$BASE/returncode" ]; then
  stage baseline-missing-returncode
  exit 10
fi
BASE_RC=$(tr -d '\r\n' < "$BASE/returncode")
if [ "$BASE_RC" != 0 ]; then
  stage "baseline-failed-$BASE_RC"
  exit "$BASE_RC"
fi
if ! ensure_exact_container_absent evo-m2-baseline20-20260902; then
  stage baseline-container-cleanup-failed
  exit 11
fi

if [ -e "$OUT" ]; then
  stage output-already-exists
  exit 12
fi
mkdir -p "$OUT/empty-seed"
ln -s "$BASE" "$OUT/baseline"

export DOCKER_HOST="$SOCKET"
export HF_ENDPOINT=https://hf-mirror.com
export OSWORLD_MCP_DEP_TIMEOUT=1800
export OSWORLD_PODMAN=1
export OSWORLD_MCP_INJECT=1
export PYTHONPATH=/data/songweishuai/envs/osworld/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
export OSWORLD_MODEL=${OSWORLD_MODEL:-gpt-5.6-terra}
export OSWORLD_API_URL=${OSWORLD_API_URL:-http://llmapi.evolventapi.com/v1}
if [ -z "${OSWORLD_API_KEY:-}" ]; then
  OSWORLD_API_KEY=$(awk 'NR == 1 { sub(/^[^:=]*[:=][[:space:]]*/, ""); sub(/\r$/, ""); print; exit }' "$CONFIG")
  export OSWORLD_API_KEY
fi
if [ -z "$OSWORLD_API_KEY" ] || [ ! -x "$PYTHON" ] || [ ! -f "$TASK" ]; then
  stage runtime-input-missing
  exit 13
fi

cd "$REPO"
stage online-running
set +e
timeout 14400s "$PYTHON" -m experiments.m2_small_eval.run_arm \
  --arm online \
  --runner-root "$RUNNER" \
  --output-root "$OUT" \
  --seed-memory "$OUT/empty-seed" \
  --task-manifest "$TASK" \
  --evotree-root "$REPO" \
  --python "$PYTHON" \
  --model "$OSWORLD_MODEL" \
  --api-url "$OSWORLD_API_URL" \
  --max-steps 8 \
  --label "$LABEL"
ONLINE_RC=$?
set -e
printf '%s\n' "$ONLINE_RC" > "$OUT/online.returncode"
if [ "$ONLINE_RC" -ne 0 ]; then
  stage "online-failed-$ONLINE_RC"
  exit "$ONLINE_RC"
fi
if ! ensure_exact_container_absent evo-m2-20tasks-20260902-online; then
  stage online-container-cleanup-failed
  exit 21
fi

stage frozen-running
set +e
timeout 14400s "$PYTHON" -m experiments.m2_small_eval.run_arm \
  --arm frozen \
  --runner-root "$RUNNER" \
  --output-root "$OUT" \
  --seed-memory "$OUT/online/memory/env-0" \
  --task-manifest "$TASK" \
  --evotree-root "$REPO" \
  --python "$PYTHON" \
  --model "$OSWORLD_MODEL" \
  --api-url "$OSWORLD_API_URL" \
  --max-steps 8 \
  --label "$LABEL"
FROZEN_RC=$?
set -e
printf '%s\n' "$FROZEN_RC" > "$OUT/frozen.returncode"
if [ "$FROZEN_RC" -ne 0 ]; then
  stage "frozen-failed-$FROZEN_RC"
  exit "$FROZEN_RC"
fi
if ! ensure_exact_container_absent evo-m2-20tasks-20260902-frozen; then
  stage frozen-container-cleanup-failed
  exit 31
fi

stage complete
