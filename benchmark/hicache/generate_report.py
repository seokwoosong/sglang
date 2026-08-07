"""Generate a summary report from HiCache benchmark results.

Reads microbenchmark and serving benchmark JSON files and produces
a comparison table.

Usage:
  python generate_report.py --input benchmark/hicache/results --timestamp 20260806_180000
"""

import argparse
import glob
import json
import os


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_microbench_report(input_dir, timestamp):
    """Generate microbenchmark comparison table."""
    print("=" * 80)
    print("Microbenchmark Results (D2H / H2D copy time)")
    print("=" * 80)
    print()

    versions = ["static", "unified_old", "unified_new"]
    data = {}
    for v in versions:
        path = os.path.join(input_dir, f"microbench_{v}_{timestamp}.json")
        if os.path.exists(path):
            data[v] = load_json(path)

    if not data:
        print("  No microbenchmark data found.")
        return

    # Get token counts from first available version
    token_counts = []
    for v in versions:
        if v in data and data[v].get("d2h"):
            token_counts = [e["tokens"] for e in data[v]["d2h"]]
            break

    # D2H table
    print("D2H (GPU → Host) Copy Time (μs)")
    print("-" * 60)
    header = f"{'Tokens':>8}"
    for v in versions:
        label = {"static": "A(static)", "unified_old": "B(old)", "unified_new": "C(new)"}[v]
        header += f" | {label:>12}"
    print(header)
    print("-" * 60)

    for tc in token_counts:
        row = f"{tc:>8}"
        for v in versions:
            if v in data:
                entry = next((e for e in data[v]["d2h"] if e["tokens"] == tc), None)
                if entry:
                    row += f" | {entry['time_us']:>12.1f}"
                else:
                    row += f" | {'N/A':>12}"
            else:
                row += f" | {'N/A':>12}"
        print(row)
    print()

    # H2D table
    print("H2D (Host → GPU) Copy Time (μs)")
    print("-" * 60)
    print(header)
    print("-" * 60)

    for tc in token_counts:
        row = f"{tc:>8}"
        for v in versions:
            if v in data:
                entry = next((e for e in data[v]["h2d"] if e["tokens"] == tc), None)
                if entry:
                    row += f" | {entry['time_us']:>12.1f}"
                else:
                    row += f" | {'N/A':>12}"
            else:
                row += f" | {'N/A':>12}"
        print(row)
    print()

    # Speedup analysis
    if "unified_old" in data and "unified_new" in data:
        print("Speedup: B(old) → C(new)")
        print("-" * 40)
        for tc in token_counts:
            old_entry = next((e for e in data["unified_old"]["d2h"] if e["tokens"] == tc), None)
            new_entry = next((e for e in data["unified_new"]["d2h"] if e["tokens"] == tc), None)
            if old_entry and new_entry and new_entry["time_us"] > 0:
                speedup = old_entry["time_us"] / new_entry["time_us"]
                print(f"  Tokens {tc:>5}: {speedup:.2f}x")
        print()


def generate_serving_report(input_dir, timestamp):
    """Generate serving benchmark comparison table."""
    print("=" * 80)
    print("Serving Benchmark Results")
    print("=" * 80)
    print()

    versions = ["static", "unified_old", "unified_new"]
    scenarios = ["short_prefix", "long_prefix", "mixed"]

    for scenario in scenarios:
        print(f"Scenario: {scenario}")
        print("-" * 80)
        header = f"{'Version':>15} | {'Throughput':>12} | {'Mean lat':>10} | {'P50':>10} | {'P99':>10}"
        print(header)
        print("-" * 80)

        for v in versions:
            path = os.path.join(input_dir, f"serving_{v}_{scenario}_{timestamp}.json")
            if os.path.exists(path):
                data = load_json(path)
                print(f"{v:>15} | {data['throughput_tok_s']:>10.1f} t/s | "
                      f"{data['mean_latency_ms']:>8.1f} ms | "
                      f"{data['p50_latency_ms']:>8.1f} ms | "
                      f"{data['p99_latency_ms']:>8.1f} ms")
            else:
                print(f"{v:>15} | {'N/A':>12} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate HiCache benchmark report")
    parser.add_argument("--input", default="benchmark/hicache/results", help="Results directory")
    parser.add_argument("--timestamp", required=True, help="Timestamp from benchmark run")
    args = parser.parse_args()

    generate_microbench_report(args.input, args.timestamp)
    generate_serving_report(args.input, args.timestamp)

    print("=" * 80)
    print("Report complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
