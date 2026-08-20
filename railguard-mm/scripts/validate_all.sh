#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m compileall -q edge ml cloud scripts benchmarks
python scripts/check_test_dependencies.py
python scripts/sync_evidence.py --check
pytest -q
cmake -S edge/cpp -B build/edge-cpp -DCMAKE_BUILD_TYPE=Release -DRAILGUARD_WARNINGS_AS_ERRORS=ON
cmake --build build/edge-cpp -j2
ctest --test-dir build/edge-cpp --output-on-failure
make firmware-host-test
./build/edge-cpp/railguard_edge
