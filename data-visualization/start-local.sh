#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cleanup() {
  kill "$PIPELINE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"$ROOT_DIR/worker/.venv/bin/python" "$ROOT_DIR/worker/pipeline_api.py" &
PIPELINE_PID=$!
cd "$ROOT_DIR/site"
npm run dev
