#!/usr/bin/env bash
set -euo pipefail
ONNX=${1:-models/fusion_transformer.onnx}
ENGINE=${2:-models/fusion_transformer_fp16.engine}
SEQ_MIN=${SEQ_MIN:-4}; SEQ_MAX=${SEQ_MAX:-128}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_MANIFEST="${ONNX}.manifest.json"
command -v trtexec >/dev/null || { echo "trtexec not found; run this on JetPack/TensorRT." >&2; exit 2; }
[[ -f "$SOURCE_MANIFEST" ]] || { echo "missing checkpoint-derived ONNX manifest: $SOURCE_MANIFEST" >&2; exit 3; }
read -r MODEL_VERSION MANIFEST_SEQ MANIFEST_STEP < <(python "$ROOT/scripts/verify_model_manifest.py" "$ONNX" --manifest "$SOURCE_MANIFEST" --kind onnx --format tsv)
SEQ_OPT=${SEQ_OPT:-$MANIFEST_SEQ}
(( SEQ_MIN <= SEQ_OPT && SEQ_OPT <= SEQ_MAX )) || { echo "sequence profile must contain manifest sequence_length=$MANIFEST_SEQ" >&2; exit 4; }
trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 \
  --minShapes=frames:1x${SEQ_MIN}x3x96x96,sensors:1x${SEQ_MIN}x9 \
  --optShapes=frames:1x${SEQ_OPT}x3x96x96,sensors:1x${SEQ_OPT}x9 \
  --maxShapes=frames:1x${SEQ_MAX}x3x96x96,sensors:1x${SEQ_MAX}x9 \
  --memPoolSize=workspace:2048 --profilingVerbosity=detailed
python - "$SOURCE_MANIFEST" "$ENGINE" "$SEQ_MIN" "$SEQ_OPT" "$SEQ_MAX" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
source_manifest, engine = map(Path, sys.argv[1:3])
seq_min, seq_opt, seq_max = map(int, sys.argv[3:6])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
data=json.loads(source_manifest.read_text())
engine_sha=sha(engine)
data.update({
    'engine':str(engine), 'engine_sha256':engine_sha, 'precision':'fp16',
    'sequence_profile':{'min':seq_min,'opt':seq_opt,'max':seq_max},
    'built_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
})
Path(str(engine)+'.manifest.json').write_text(json.dumps(data,indent=2)+'\n')
print(f"model_version={data['model_version']} engine_sha256={engine_sha}")
PY
printf 'built %s\n' "$ENGINE"
