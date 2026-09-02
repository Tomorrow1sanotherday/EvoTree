#!/usr/bin/env bash
set -euo pipefail

VERIFY_DIR=$(mktemp -d /tmp/evo-wheel-verify-XXXXXX)
case "$VERIFY_DIR" in
  /tmp/evo-wheel-verify-*) ;;
  *) printf 'unsafe temporary directory: %s\n' "$VERIFY_DIR" >&2; exit 2 ;;
esac
trap 'rm -rf -- "$VERIFY_DIR"' EXIT

python3 -m venv "$VERIFY_DIR"
"$VERIFY_DIR/bin/pip" install --no-index \
  --find-links /data/songweishuai/evotree_mcp_wheelhouse_py310_20260902 \
  fastmcp==2.13.3 playwright
"$VERIFY_DIR/bin/python" -c 'import importlib.metadata; import playwright; from fastmcp.tools.tool import Tool; assert importlib.metadata.version("fastmcp") == "2.13.3"; print("OFFLINE_VERIFY_OK")'
