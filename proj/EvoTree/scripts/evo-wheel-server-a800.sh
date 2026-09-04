#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE=/data/songweishuai/projs/rh/EvoTree/runs/evotree_mcp_wheelhouse_py310_20260902
PORT=18765

exec python3 -m http.server "$PORT" \
  --bind 0.0.0.0 \
  --directory "$WHEELHOUSE"
