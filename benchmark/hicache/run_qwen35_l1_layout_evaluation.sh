#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=/home/sukwoo24/.venv_sglang/bin/python
matrix="$repo_root/benchmark/hicache/run_qwen35_l1_layout_matrix.py"
artifact_root="$repo_root/artifacts/qwen35_l1_layout_743cae2"

common=(
  --pages 1 8 32
  --max-total-tokens 120000
  --max-running-requests 8
  --mem-fraction-static 0.27
  --artifact-root "$artifact_root"
)

"$python_bin" "$matrix" parity "${common[@]}" --repetition 1

for repetition in 1 2 3; do
  "$python_bin" "$matrix" resident "${common[@]}" \
    --repetition "$repetition"
  "$python_bin" "$matrix" pressure "${common[@]}" \
    --repetition "$repetition"
  "$python_bin" "$matrix" profile "${common[@]}" \
    --repetition "$repetition"
done
