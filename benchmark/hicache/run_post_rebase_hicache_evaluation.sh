#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=/home/sukwoo24/.venv_sglang/bin/python
matrix="$repo_root/benchmark/hicache/run_qwen35_hicache_matrix.py"
artifact_root="$repo_root/artifacts/qwen35_unified_hicache_post_rebase_743cae2"

common=(
  --model-size 0.8b
  --mem-fraction-static 0.27
  --artifact-root "$artifact_root"
)
post_variants=(post-s0 post-s1 post-u0 post-u3)
default_variants=(post-s0-default post-s1-default)

"$python_bin" "$matrix" parity \
  "${common[@]}" --pages 1 8 32 --variants "${post_variants[@]}"
"$python_bin" "$matrix" graph-parity \
  "${common[@]}" --pages 8 --variants "${post_variants[@]}"

for repetition in 1 2 3; do
  "$python_bin" "$matrix" clean \
    "${common[@]}" --pages 1 8 32 --repetition "$repetition" \
    --variants "${post_variants[@]}"
  "$python_bin" "$matrix" graph \
    "${common[@]}" --pages 8 --repetition "$repetition" \
    --variants "${post_variants[@]}"
  "$python_bin" "$matrix" profile \
    "${common[@]}" --pages 1 8 32 --repetition "$repetition" \
    --variants post-s1 post-u3
done

"$python_bin" "$matrix" parity \
  "${common[@]}" --pages 1 8 32 --variants "${default_variants[@]}"
for repetition in 1 2 3; do
  "$python_bin" "$matrix" clean \
    "${common[@]}" --pages 1 8 32 --repetition "$repetition" \
    --variants "${default_variants[@]}"
done
