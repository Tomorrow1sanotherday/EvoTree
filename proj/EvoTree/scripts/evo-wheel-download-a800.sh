#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE=/data/songweishuai/evotree_mcp_wheelhouse_py310_20260902
LOG=/data/songweishuai/evotree-wheel-download.log
RC=/data/songweishuai/evotree-wheel-download.rc

mkdir -p "$WHEELHOUSE"
PYTHONUNBUFFERED=1 python3 -m pip download \
  --dest "$WHEELHOUSE" \
  --only-binary=:all: \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  fastmcp==2.13.3 playwright \
  >"$LOG" 2>&1
printf '%s\n' "$?" > "$RC"
