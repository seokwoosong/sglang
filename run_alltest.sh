#!/usr/bin/env bash
# Run the Unified Memory + HiCache regression suite in one command.
#
# Default scope:
#   - CPU allocator/dispatch/admission tests touched by the feature
#   - CUDA Full-KV and Mamba transfer/race tests
#   - Qwen3.5-4B L1/L2/L3 serving integrity at TP=1,2,4
#     (write-through + write-back, CUDA Graph off + on)
#   - Qwen3.5-4B concurrent tier-churn serving stress at TP=1,2,4
#
# The broader HiCache tests use unrelated models/configurations. Enable the
# model-independent variants explicitly with --include-general-hicache. Other
# model-specific upstream accuracy tests are intentionally outside this runner.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${SGLANG_UNIFIED_HICACHE_TEST_MODEL:-Qwen/Qwen3.5-4B}"
OUTPUT_ROOT="${SGLANG_ALLTEST_OUTPUT_DIR:-/tmp/unified-hicache-alltests}"
CACHE_ROOT="${SGLANG_ALLTEST_CACHE_DIR:-/tmp/sglang-alltest-cache}"
TP_SIZES=(1 2 4)

INCLUDE_GENERAL_HICACHE=0
SKIP_SERVING=0
LIST_ONLY=0

usage() {
    cat <<'EOF'
Usage: ./run_alltest.sh [options]

Options:
  --model PATH_OR_ID          Qwen3.5-4B checkpoint for serving tests.
  --output-dir DIR            Logs and structured artifacts directory.
  --python PATH               Python executable (default: python3).
  --include-general-hicache   Also run model-independent HiCache variants.
  --skip-serving              Run only unit and CUDA kernel tests.
  --list                      Print the suite/configuration without running.
  -h, --help                  Show this help.

Useful environment overrides:
  SGLANG_UNIFIED_HICACHE_INTEGRITY_MEM_FRACTION_STATIC
  SGLANG_UNIFIED_HICACHE_STRESS_MEM_FRACTION_STATIC
  SGLANG_UNIFIED_HICACHE_SIZE
  SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS
  SGLANG_UNIFIED_HICACHE_PROMPT_COUNT
  SGLANG_UNIFIED_HICACHE_CHURN_ROUNDS
  SGLANG_UNIFIED_HICACHE_CONCURRENT_WORKERS
  SGLANG_UNIFIED_HICACHE_MAX_RUNNING_REQUESTS
  SGLANG_UNIFIED_HICACHE_ENABLE_CUDA_GRAPH
  SGLANG_ALLTEST_UNIT_TIMEOUT
  SGLANG_ALLTEST_KERNEL_TIMEOUT
  SGLANG_ALLTEST_INTEGRITY_TIMEOUT
  SGLANG_ALLTEST_STRESS_TIMEOUT
  SGLANG_ALLTEST_CACHE_DIR

Examples:
  ./run_alltest.sh --model /group-volume/Qwen3.5-4B
  SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS=120 ./run_alltest.sh \
    --model /group-volume/Qwen3.5-4B
EOF
}

while (($#)); do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
            MODEL=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
            OUTPUT_ROOT=$2
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "--python requires a value" >&2; exit 2; }
            PYTHON_BIN=$2
            shift 2
            ;;
        --include-general-hicache)
            INCLUDE_GENERAL_HICACHE=1
            shift
            ;;
        --skip-serving)
            SKIP_SERVING=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$REPO_ROOT" || {
    echo "Cannot enter repository: $REPO_ROOT" >&2
    exit 2
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
}
command -v timeout >/dev/null 2>&1 || {
    echo "GNU timeout is required." >&2
    exit 2
}

PYTHONPATH_VALUE="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

GPU_COUNT=0
GPU_NAME="none"
GPU_MEMORY_MIB=0
GPU_INFO=$(
    env "PYTHONPATH=$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
import torch

count = torch.cuda.device_count()
if count:
    props = torch.cuda.get_device_properties(0)
    print(f"{count}|{props.name}|{props.total_memory // (1024 * 1024)}")
else:
    print("0|none|0")
' 2>/dev/null
) || GPU_INFO="0|none|0"
if [[ "$GPU_INFO" =~ ^[0-9]+\|[^|]+\|[0-9]+$ ]]; then
    IFS='|' read -r GPU_COUNT GPU_NAME GPU_MEMORY_MIB <<<"$GPU_INFO"
fi

# Defaults reproduce the configurations qualified on this branch. A 4B
# checkpoint on an >=70 GiB GPU uses a lower static fraction and more pressure
# requests; every value can be overridden independently.
MODEL_LOWER=${MODEL,,}
if [[ "$MODEL_LOWER" != *"qwen3.5"* || "$MODEL_LOWER" != *"4b"* ]]; then
    echo "ERROR: serving qualification requires a Qwen3.5-4B checkpoint: $MODEL" >&2
    echo "Use --model /path/to/Qwen3.5-4B (or its model ID)." >&2
    exit 2
fi

DEFAULT_PRESSURE_REQUESTS=80
DEFAULT_INTEGRITY_MEM_FRACTION=0.40
DEFAULT_STRESS_MEM_FRACTION=0.40
DEFAULT_STRESS_PROMPTS=60
if ((GPU_MEMORY_MIB >= 70000)); then
    DEFAULT_PRESSURE_REQUESTS=100
    DEFAULT_INTEGRITY_MEM_FRACTION=0.20
    DEFAULT_STRESS_MEM_FRACTION=0.20
    DEFAULT_STRESS_PROMPTS=100
fi

COMMON_MEM_OVERRIDE=${SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC:-}
INTEGRITY_MEM_FRACTION=${SGLANG_UNIFIED_HICACHE_INTEGRITY_MEM_FRACTION_STATIC:-${COMMON_MEM_OVERRIDE:-$DEFAULT_INTEGRITY_MEM_FRACTION}}
STRESS_MEM_FRACTION=${SGLANG_UNIFIED_HICACHE_STRESS_MEM_FRACTION_STATIC:-${COMMON_MEM_OVERRIDE:-$DEFAULT_STRESS_MEM_FRACTION}}
PRESSURE_REQUESTS=${SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS:-$DEFAULT_PRESSURE_REQUESTS}
STRESS_PROMPTS=${SGLANG_UNIFIED_HICACHE_PROMPT_COUNT:-$DEFAULT_STRESS_PROMPTS}
# Unified KV/Mamba typed chunks interpret this as one shared total budget. 2 GB
# matches the former 4B qualification's effective KV 1 GB + Mamba 1 GB capacity.
HICACHE_SIZE=${SGLANG_UNIFIED_HICACHE_SIZE:-2}

UNIT_TIMEOUT=${SGLANG_ALLTEST_UNIT_TIMEOUT:-600}
KERNEL_TIMEOUT=${SGLANG_ALLTEST_KERNEL_TIMEOUT:-900}
INTEGRITY_TIMEOUT=${SGLANG_ALLTEST_INTEGRITY_TIMEOUT:-1800}
STRESS_TIMEOUT=${SGLANG_ALLTEST_STRESS_TIMEOUT:-2400}
GENERAL_TIMEOUT=${SGLANG_ALLTEST_GENERAL_TIMEOUT:-5400}

# A complete serving qualification means that all three requested TP modes ran.
# Do not silently turn an incomplete machine into an apparent all-pass result.
if ((!LIST_ONLY && !SKIP_SERVING && GPU_COUNT < 4)); then
    echo "ERROR: the TP=1,2,4 serving sweep requires at least four visible GPUs; found $GPU_COUNT." >&2
    echo "Use --skip-serving only when intentionally running the non-serving subset." >&2
    exit 2
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_ID"
LOG_DIR="$OUTPUT_DIR/logs"
ARTIFACT_DIR="$OUTPUT_DIR/artifacts"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"
mkdir -p \
    "$CACHE_ROOT/flashinfer" \
    "$CACHE_ROOT/torch-extensions" \
    "$CACHE_ROOT/triton"

declare -a TEST_NAMES TEST_GROUPS TEST_RESULTS TEST_DURATIONS TEST_EXIT_CODES TEST_LOGS
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

record_result() {
    local name=$1 group=$2 result=$3 duration=$4 exit_code=$5 log=$6
    TEST_NAMES+=("$name")
    TEST_GROUPS+=("$group")
    TEST_RESULTS+=("$result")
    TEST_DURATIONS+=("$duration")
    TEST_EXIT_CODES+=("$exit_code")
    TEST_LOGS+=("$log")
}

run_pytest() {
    local name=$1 group=$2 timeout_seconds=$3 file=$4
    shift 4
    local -a extra_env=("$@")

    TOTAL=$((TOTAL + 1))
    local log="$LOG_DIR/$(printf '%02d' "$TOTAL")_${name}.log"
    echo
    echo "=== [$TOTAL] $group / $name ==="
    echo "file: $file"

    if [[ ! -f "$file" ]]; then
        echo "FAIL: required test file not found"
        FAILED=$((FAILED + 1))
        record_result "$name" "$group" "FAIL" 0 2 "$log"
        print_summary
        exit 2
    fi

    if ((LIST_ONLY)); then
        echo "LIST ONLY"
        record_result "$name" "$group" "LIST" 0 0 "$log"
        return 0
    fi

    local start test_rc duration result
    start=$(date +%s)
    env \
        "PYTHONPATH=$PYTHONPATH_VALUE" \
        "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT/flashinfer}" \
        "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch-extensions}" \
        "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}" \
        "SGLANG_UNIFIED_HICACHE_TEST_ARTIFACT_DIR=$ARTIFACT_DIR" \
        "${extra_env[@]}" \
        timeout "$timeout_seconds" \
        "$PYTHON_BIN" -m pytest "$file" -v -s --tb=short 2>&1 | tee "$log"
    local -a pipeline_status=("${PIPESTATUS[@]}")
    test_rc=${pipeline_status[0]}
    duration=$(($(date +%s) - start))

    if ((test_rc == 0)); then
        result=PASS
        PASSED=$((PASSED + 1))
    elif ((test_rc == 124)); then
        result=TIMEOUT
        FAILED=$((FAILED + 1))
    elif grep -qE \
        "CUDA error|torch\.AcceleratorError|illegal memory access|device-side assert|invalid device ordinal|NCCL.*(error|failed)" \
        "$log" 2>/dev/null; then
        # This category includes software ordering/index bugs as well as hardware
        # faults. Do not label every CUDA exception as a hardware failure.
        result=CUDA_ERROR
        FAILED=$((FAILED + 1))
    elif grep -qE \
        "Read-only file system|Permission denied|ModuleNotFoundError|No module named" \
        "$log" 2>/dev/null; then
        result=ENV_ERROR
        FAILED=$((FAILED + 1))
    else
        result=FAIL
        FAILED=$((FAILED + 1))
    fi

    echo "$result: ${duration}s (exit=$test_rc, log=$log)"
    record_result "$name" "$group" "$result" "$duration" "$test_rc" "$log"

    if ((test_rc != 0)); then
        echo "Stopping immediately: every required invocation must pass."
        print_summary
        exit "$test_rc"
    fi
    return 0
}

record_skip() {
    local name=$1 group=$2 reason=$3
    TOTAL=$((TOTAL + 1))
    SKIPPED=$((SKIPPED + 1))
    echo
    echo "=== [$TOTAL] $group / $name ==="
    echo "SKIP: $reason"
    record_result "$name" "$group" "SKIP" 0 0 "-"
}

print_summary() {
    echo
    echo "============================================================================"
    echo "SUMMARY (one row is one pytest file invocation, not one collected test case)"
    echo "============================================================================"
    printf "%-34s %-10s %-12s %-8s %-6s\n" "Invocation" "Group" "Result" "Seconds" "Exit"
    printf '%s\n' "----------------------------------------------------------------------------"
    local i
    for i in "${!TEST_NAMES[@]}"; do
        printf "%-34s %-10s %-12s %-8s %-6s\n" \
            "${TEST_NAMES[$i]}" \
            "${TEST_GROUPS[$i]}" \
            "${TEST_RESULTS[$i]}" \
            "${TEST_DURATIONS[$i]}" \
            "${TEST_EXIT_CODES[$i]}"
    done
    printf '%s\n' "----------------------------------------------------------------------------"
    echo "Invocations: $TOTAL | Pass: $PASSED | Fail: $FAILED | Skip: $SKIPPED"
    echo "Output: $OUTPUT_DIR"
    echo "============================================================================"
}

echo "============================================================================"
echo "Unified Memory + HiCache regression runner"
echo "Repository:             $REPO_ROOT"
echo "Python:                 $PYTHON_BIN"
echo "Model:                  $MODEL"
echo "Serving TP sweep:       ${TP_SIZES[*]}"
echo "GPU:                    $GPU_COUNT x $GPU_NAME (${GPU_MEMORY_MIB} MiB each)"
echo "Integrity pressure:     $PRESSURE_REQUESTS requests"
echo "Integrity mem fraction: $INTEGRITY_MEM_FRACTION"
echo "Stress prompts:         $STRESS_PROMPTS"
echo "Stress mem fraction:    $STRESS_MEM_FRACTION"
echo "Shared HiCache size:     $HICACHE_SIZE GB total"
echo "Output:                 $OUTPUT_DIR"
echo "Writable cache:         $CACHE_ROOT"
echo "============================================================================"

# CPU and mocked unit coverage. These are intentionally model-independent.
run_pytest unified_hicache_artifacts unit "$UNIT_TIMEOUT" \
    test/registered/unit/mem_cache/test_unified_hicache_artifacts.py
run_pytest typed_chunk_host unit "$UNIT_TIMEOUT" \
    test/registered/unit/mem_cache/test_typed_chunk_host.py
run_pytest unified_memory_pool unit "$UNIT_TIMEOUT" \
    test/registered/unit/mem_cache/test_unified_memory_pool.py
run_pytest multi_ended_allocator unit "$UNIT_TIMEOUT" \
    test/registered/unit/mem_cache/test_multi_ended_allocator.py
run_pytest unified_radix_hicache_dispatch unit "$UNIT_TIMEOUT" \
    test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py
run_pytest hicache_staged_write_back unit "$UNIT_TIMEOUT" \
    test/registered/unit/mem_cache/test_hicache_staged_write_back_dispatch.py
run_pytest prefill_adder unit "$UNIT_TIMEOUT" \
    test/registered/unit/managers/test_prefill_adder.py

if ((GPU_COUNT >= 1 || LIST_ONLY)); then
    run_pytest unified_chunk_hicache_transfer kernel "$KERNEL_TIMEOUT" \
        test/registered/kernels/ops/kvcache/test_unified_chunk_hicache_transfer.py
    run_pytest unified_hicache_transfer kernel "$KERNEL_TIMEOUT" \
        test/registered/kernels/ops/kvcache/test_unified_hicache_transfer.py
    run_pytest transfer_mamba kernel "$KERNEL_TIMEOUT" \
        test/registered/kernels/ops/mamba/test_transfer_mamba.py
else
    record_skip unified_chunk_hicache_transfer kernel "CUDA GPU not found"
    record_skip unified_hicache_transfer kernel "CUDA GPU not found"
    record_skip transfer_mamba kernel "CUDA GPU not found"
fi

if ((SKIP_SERVING)); then
    for tp_size in "${TP_SIZES[@]}"; do
        record_skip "unified_hicache_integrity_tp${tp_size}" serving "--skip-serving"
        record_skip "unified_hicache_graph_integrity_tp${tp_size}" serving "--skip-serving"
        record_skip "unified_hicache_overlap_stress_tp${tp_size}" serving "--skip-serving"
    done
else
    for tp_size in "${TP_SIZES[@]}"; do
        run_pytest "unified_hicache_integrity_tp${tp_size}" serving "$INTEGRITY_TIMEOUT" \
            test/registered/hicache/test_unified_memory_hicache_integrity.py \
            "SGLANG_UNIFIED_HICACHE_TEST_MODEL=$MODEL" \
            "SGLANG_UNIFIED_HICACHE_TP_SIZE=$tp_size" \
            "SGLANG_UNIFIED_HICACHE_SIZE=$HICACHE_SIZE" \
            "SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS=$PRESSURE_REQUESTS" \
            "SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC=$INTEGRITY_MEM_FRACTION"
        run_pytest "unified_hicache_graph_integrity_tp${tp_size}" serving "$INTEGRITY_TIMEOUT" \
            test/registered/hicache/test_unified_memory_hicache_integrity.py \
            "SGLANG_UNIFIED_HICACHE_TEST_MODEL=$MODEL" \
            "SGLANG_UNIFIED_HICACHE_TP_SIZE=$tp_size" \
            "SGLANG_UNIFIED_HICACHE_SIZE=$HICACHE_SIZE" \
            "SGLANG_UNIFIED_HICACHE_PRESSURE_REQUESTS=$PRESSURE_REQUESTS" \
            "SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC=$INTEGRITY_MEM_FRACTION" \
            "SGLANG_UNIFIED_HICACHE_ENABLE_CUDA_GRAPH=1"
        run_pytest "unified_hicache_overlap_stress_tp${tp_size}" serving "$STRESS_TIMEOUT" \
            test/registered/hicache/test_unified_memory_hicache_overlap_stress.py \
            "SGLANG_UNIFIED_HICACHE_TEST_MODEL=$MODEL" \
            "SGLANG_UNIFIED_HICACHE_TP_SIZE=$tp_size" \
            "SGLANG_UNIFIED_HICACHE_SIZE=$HICACHE_SIZE" \
            "SGLANG_UNIFIED_HICACHE_PROMPT_COUNT=$STRESS_PROMPTS" \
            "SGLANG_UNIFIED_HICACHE_MEM_FRACTION_STATIC=$STRESS_MEM_FRACTION"
    done
fi

if ((INCLUDE_GENERAL_HICACHE)); then
    if ((GPU_COUNT >= 1 || LIST_ONLY)); then
        run_pytest hicache_variants general "$GENERAL_TIMEOUT" \
            test/registered/hicache/test_hicache_variants.py
    else
        record_skip hicache_variants general "CUDA GPU not found"
    fi
fi

print_summary

if ((LIST_ONLY)); then
    exit 0
fi
if ((FAILED == 0)); then
    echo "ALL EXECUTED TEST INVOCATIONS PASSED"
    exit 0
fi
echo "ONE OR MORE TEST INVOCATIONS FAILED"
exit 1
