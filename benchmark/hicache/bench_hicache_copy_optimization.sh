#!/bin/bash
# =============================================================================
# HiCache Copy Optimization Benchmark Script
#
# Compares three versions of HiCache D2H/H2D copy:
#   A) Static-fraction (--enable-unified-memory off)
#   B) Unified-memory (current — staging buffer relayout)
#   C) Unified-memory (optimized — direct JIT kernel, no staging)
#
# Usage:
#   ./bench_hicache_copy_optimization.sh [MODEL] [GPU]
#
# Example:
#   ./bench_hicache_copy_optimization.sh Qwen/Qwen3.5-32B cuda:0
#
# Prerequisites:
#   - SGLang installed (pip install -e .)
#   - Model downloaded
#   - CUDA GPU available
# =============================================================================

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.5-32B}"
GPU="${2:-cuda:0}"
PORT=30000
NUM_ITERS=100
OUTPUT_DIR="benchmark/hicache/results"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "HiCache Copy Optimization Benchmark"
echo "============================================"
echo "Model: $MODEL"
echo "GPU:   $GPU"
echo "Iters: $NUM_ITERS"
echo "Output: $OUTPUT_DIR"
echo "============================================"
echo ""

# =============================================================================
# Experiment 1: Microbenchmark (per-transfer timing)
# =============================================================================
echo "=== Experiment 1: Microbenchmark ==="
echo ""

# Run microbenchmark for each version
for VERSION in static unified_old unified_new; do
    echo "--- Running $VERSION microbenchmark ---"
    python benchmark/hicache/bench_hicache_microbench.py \
        --model "$MODEL" \
        --gpu "$GPU" \
        --version "$VERSION" \
        --num-iters "$NUM_ITERS" \
        --output "$OUTPUT_DIR/microbench_${VERSION}_${TIMESTAMP}.json" \
        2>&1 | tee "$OUTPUT_DIR/microbench_${VERSION}_${TIMESTAMP}.log"
    echo ""
done

# =============================================================================
# Experiment 2: End-to-end serving performance
# =============================================================================
echo "=== Experiment 2: End-to-end Serving ==="
echo ""

for VERSION in static unified_old unified_new; do
    echo "--- Running $VERSION serving benchmark ---"

    # Set server args based on version
    case $VERSION in
        static)
            SERVER_ARGS="--hicache-ratio 0.5"
            ;;
        unified_old|unified_new)
            SERVER_ARGS="--enable-unified-memory --hicache-ratio 0.5"
            ;;
    esac

    # Start server
    echo "Starting server ($VERSION)..."
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port $PORT \
        $SERVER_ARGS \
        --trust-remote-code \
        > "$OUTPUT_DIR/server_${VERSION}_${TIMESTAMP}.log" 2>&1 &
    SERVER_PID=$!

    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is ready."
            break
        fi
        sleep 2
    done

    # Run benchmark scenarios
    for SCENARIO in short_prefix long_prefix mixed; do
        echo "--- Scenario: $SCENARIO ---"
        python benchmark/hicache/bench_hicache_serving.py \
            --port $PORT \
            --scenario "$SCENARIO" \
            --output "$OUTPUT_DIR/serving_${VERSION}_${SCENARIO}_${TIMESTAMP}.json" \
            2>&1 | tee "$OUTPUT_DIR/serving_${VERSION}_${SCENARIO}_${TIMESTAMP}.log"
        echo ""
    done

    # Stop server
    echo "Stopping server ($VERSION)..."
    kill $SERVER_PID || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
    echo ""
done

# =============================================================================
# Experiment 3: nsys profiling
# =============================================================================
echo "=== Experiment 3: nsys Profiling ==="
echo ""

for VERSION in static unified_old unified_new; do
    echo "--- nsys profile: $VERSION ---"

    case $VERSION in
        static)
            SERVER_ARGS="--hicache-ratio 0.5"
            ;;
        unified_old|unified_new)
            SERVER_ARGS="--enable-unified-memory --hicache-ratio 0.5"
            ;;
    esac

    # Start server with nsys
    nsys profile \
        --output "$OUTPUT_DIR/nsys_${VERSION}_${TIMESTAMP}" \
        --force-overwrite true \
        python -m sglang.launch_server \
            --model-path "$MODEL" \
            --port $PORT \
            $SERVER_ARGS \
            --trust-remote-code \
        > "$OUTPUT_DIR/nsys_server_${VERSION}_${TIMESTAMP}.log" 2>&1 &
    SERVER_PID=$!

    # Wait for server
    for i in $(seq 1 120); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done

    # Run a short workload to trigger HiCache
    python benchmark/hicache/bench_hicache_serving.py \
        --port $PORT \
        --scenario long_prefix \
        --num-requests 50 \
        --output "$OUTPUT_DIR/nsys_workload_${VERSION}_${TIMESTAMP}.json" \
        2>&1 | tee "$OUTPUT_DIR/nsys_workload_${VERSION}_${TIMESTAMP}.log"

    # Stop server
    kill $SERVER_PID || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo "============================================"
echo "Benchmark Complete!"
echo "============================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To generate summary report:"
echo "  python benchmark/hicache/generate_report.py --input $OUTPUT_DIR --timestamp $TIMESTAMP"
echo ""
echo "To view nsys profiles:"
echo "  nsys-ui $OUTPUT_DIR/nsys_static_${TIMESTAMP}.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/nsys_unified_old_${TIMESTAMP}.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/nsys_unified_new_${TIMESTAMP}.nsys-rep"
