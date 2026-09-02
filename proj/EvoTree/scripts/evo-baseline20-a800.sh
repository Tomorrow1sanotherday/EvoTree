#!/usr/bin/env bash
set -u
umask 077
OUT=/data/songweishuai/evotree_m2_baseline20_20260902
RUNNER=/data/songweishuai/projs/rh/OSWorld-mcp-run
TASK=/data/songweishuai/projs/rh/EvoTree/proj/EvoTree/experiments/m2_small_eval/tasks_20.json
PYTHON=/data/songweishuai/envs/evo-osworld-20260829/bin/python
MODEL=${OSWORLD_MODEL:-gpt-5.6-terra}
API_URL=${OSWORLD_API_URL:-http://llmapi.evolventapi.com/v1}
CONFIG=/data/songweishuai/projs/rh/cofig/model.md
if [ -z "${OSWORLD_API_KEY:-}" ]; then
  OSWORLD_API_KEY=$(awk 'NR == 1 { sub(/^[^:=]*[:=][[:space:]]*/, ""); sub(/\r$/, ""); print; exit }' "$CONFIG")
  export OSWORLD_API_KEY
fi
if [ -z "${OSWORLD_API_KEY:-}" ] || [ ! -x "$PYTHON" ] || [ ! -f "$TASK" ]; then
  echo "runtime input missing" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing output" >&2
  exit 2
fi
mkdir -p "$OUT"
export DOCKER_HOST=${DOCKER_HOST:-unix:///data/songweishuai/evo-podman.sock}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export OSWORLD_MCP_DEP_TIMEOUT=${OSWORLD_MCP_DEP_TIMEOUT:-1800}
export OSWORLD_PODMAN=1
export OSWORLD_MCP_INJECT=1
export PYTHONPATH=/data/songweishuai/envs/osworld/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
export EVO_CONTAINER_NAME=evo-m2-baseline20-20260902
cd "$RUNNER"
echo "START baseline20 container=$EVO_CONTAINER_NAME"
set +e
timeout 10800s "$PYTHON" "$RUNNER/run_multienv_e2e.py" --headless --action_space mcp --observation_type screenshot --model "$MODEL" --api_url "$API_URL" --max_steps 8 --max_trajectory_length 8 --test_all_meta_path "$TASK" --test_config_base_dir "$RUNNER/evaluation_examples" --result_dir "$OUT/results" --num_envs 1 --trial-id 0 --domain os >"$OUT/runner.log" 2>&1
RC=$?
set -u
printf "%s\n" "$RC" >"$OUT/returncode"
ROWS=$(podman --remote --url "$DOCKER_HOST" ps -a --filter name=evo-m2-baseline20-20260902 --format "{{.Names}}" 2>/dev/null)
QUERY_RC=$?
if [ "$QUERY_RC" -ne 0 ]; then echo "container query failed" >&2; exit 90; fi
if [ -n "$ROWS" ]; then echo "exact container remains: $ROWS" >&2; exit 91; fi
echo "DONE baseline20 returncode=$RC"
exit "$RC"
