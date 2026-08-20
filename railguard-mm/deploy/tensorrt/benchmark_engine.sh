#!/usr/bin/env bash
set -euo pipefail
ENGINE=${1:-models/fusion_transformer_fp16.engine}
SEQ=${SEQ:-32}
command -v trtexec >/dev/null || { echo "trtexec not found; benchmark on JetPack/TensorRT." >&2; exit 2; }
trtexec --loadEngine="$ENGINE" --shapes=frames:1x${SEQ}x3x96x96,sensors:1x${SEQ}x9 --warmUp=3000 --duration=15 --useCudaGraph --noDataTransfers
