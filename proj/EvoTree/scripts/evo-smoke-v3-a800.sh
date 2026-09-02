#!/usr/bin/env bash
set -u
umask 077

OUT=/data/songweishuai/evotree_ownership_smoke_v3_20260903
RUNNER=/data/songweishuai/projs/rh/OSWorld-mcp-run
TASK=/data/songweishuai/projs/rh/EvoTree/proj/EvoTree/experiments/m2_small_eval/tasks_smoke.json
PYTHON=/data/songweishuai/envs/evo-osworld-20260829/bin/python
CONFIG=/data/songweishuai/projs/rh/cofig/model.md

if [ -e "$OUT" ]; then
  printf 'refusing to overwrite %s\n' "$OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
export DOCKER_HOST=unix:///data/songweishuai/evo-podman.sock
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
export EVO_CONTAINER_NAME=evo-ownership-smoke-v3-20260903
cd "$RUNNER"
set +e
timeout 3600s "$PYTHON" "$RUNNER/run_multienv_e2e.py" \
  --headless --action_space mcp --observation_type screenshot \
  --model "$OSWORLD_MODEL" --api_url "$OSWORLD_API_URL" \
  --max_steps 1 --max_trajectory_length 1 \
  --test_all_meta_path "$TASK" \
  --test_config_base_dir "$RUNNER/evaluation_examples" \
  --result_dir "$OUT/results" --num_envs 1 --trial-id 0 --domain os \
  >"$OUT/runner.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/returncode"
exit "$RC"
