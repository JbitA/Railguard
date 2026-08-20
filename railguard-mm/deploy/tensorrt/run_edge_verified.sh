#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <engine.plan> [railguard_edge args...]" >&2
  exit 2
fi
ENGINE="$1"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${ENGINE}.manifest.json"
read -r MODEL_VERSION SEQ_LEN STEP_MS < <(python "$ROOT/scripts/verify_model_manifest.py" "$ENGINE" --manifest "$MANIFEST" --format tsv)
EDGE_BIN="${RAILGUARD_EDGE_BIN:-$ROOT/build/edge-jetson/railguard_edge}"
for arg in "$@"; do
  case "$arg" in
    --model-version|--seq-len|--model-step-ms)
      echo "runtime model contract arguments are manifest-controlled; do not override $arg" >&2; exit 4;;
  esac
done
exec "$EDGE_BIN" --engine "$ENGINE" --model-version "$MODEL_VERSION" --seq-len "$SEQ_LEN" --model-step-ms "$STEP_MS" "$@"
