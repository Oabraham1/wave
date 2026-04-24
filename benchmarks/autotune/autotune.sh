#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Auto-tuning orchestrator for WAVE GEMM kernels on Apple Silicon.
#
# Enumerates the tuning parameter space, generates WAVE assembly for each
# candidate configuration, assembles and compiles through the wave-asm and
# wave-metal toolchain, benchmarks on the local GPU via a lightweight Swift
# harness, and records results to a JSON cache. Supports running a small
# sanity-check subset or the full search. Results are saved per GPU model
# for future lookup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="/tmp/wave_autotune"
CACHE_DIR="$HOME/.wave"
BENCH_SIZE=4096

WAVE_ASM="cargo run --release --manifest-path=$REPO_ROOT/wave-asm/Cargo.toml --"
WAVE_METAL="cargo run --release --manifest-path=$REPO_ROOT/wave-metal/Cargo.toml --"
GENERATOR="$SCRIPT_DIR/generate_kernel.py"
BENCH_SWIFT="$SCRIPT_DIR/tune_benchmark.swift"
BENCH_BIN="/tmp/wave_tune_bench"

usage() {
    echo "Usage: $0 [--sanity | --full | --config <json>] [--variant <variant>] [--bench-size <N>]"
    echo ""
    echo "Modes:"
    echo "  --sanity    Run 5 representative configs (default)"
    echo "  --full      Run all valid configs (~168)"
    echo "  --config    Run a single config from JSON string"
    echo ""
    echo "Options:"
    echo "  --variant   Kernel variant: f32 (default), f16, bias_relu, bias_gelu, bias_relu_f16"
    echo "  --bench-size  Matrix size for benchmark (default: 4096)"
    exit 1
}

MODE="sanity"
VARIANT="f32"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sanity) MODE="sanity"; shift ;;
        --full) MODE="full"; shift ;;
        --config) MODE="single"; SINGLE_CONFIG="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --bench-size) BENCH_SIZE="$2"; shift 2 ;;
        --help|-h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$WORK_DIR" "$CACHE_DIR"

echo "=== WAVE Auto-Tuning Framework ==="
echo "Variant: $VARIANT"
echo "Bench size: ${BENCH_SIZE}x${BENCH_SIZE}"
echo "Work dir: $WORK_DIR"
echo ""

echo "Building toolchain..."
cargo build --release --manifest-path="$REPO_ROOT/wave-asm/Cargo.toml" 2>/dev/null
cargo build --release --manifest-path="$REPO_ROOT/wave-metal/Cargo.toml" 2>/dev/null

WAVE_ASM_BIN="$REPO_ROOT/target/release/wave-asm"
WAVE_METAL_BIN="$REPO_ROOT/target/release/wave-metal"

echo "Compiling benchmark harness..."
swiftc -O "$BENCH_SWIFT" -framework Metal -framework MetalPerformanceShaders -o "$BENCH_BIN" 2>/dev/null

KERNEL_NAMES_F32="gemm_register_blocked_8x8"
KERNEL_NAMES_F16="gemm_register_blocked_8x8_f16"
KERNEL_NAMES_RELU="gemm_bias_relu_fused"
KERNEL_NAMES_GELU="gemm_bias_gelu_fused"
KERNEL_NAMES_RELU_F16="gemm_bias_relu_f16_fused"

get_entry() {
    case "$VARIANT" in
        f32) echo "$KERNEL_NAMES_F32" ;;
        f16) echo "$KERNEL_NAMES_F16" ;;
        bias_relu) echo "$KERNEL_NAMES_RELU" ;;
        bias_gelu) echo "$KERNEL_NAMES_GELU" ;;
        bias_relu_f16) echo "$KERNEL_NAMES_RELU_F16" ;;
    esac
}

get_has_bias() {
    case "$VARIANT" in
        f32|f16) echo "0" ;;
        *) echo "1" ;;
    esac
}

run_config() {
    local tm=$1 tn=$2 tk=$3 bm=$4 bn=$5 pf=$6
    local tag="t${tm}x${tn}_k${tk}_b${bm}x${bn}"
    if [ "$pf" = "true" ]; then tag="${tag}_p"; else tag="${tag}_np"; fi

    local wave_file="$WORK_DIR/${tag}.wave"
    local wbin_file="$WORK_DIR/${tag}.wbin"
    local metal_file="$WORK_DIR/${tag}.metal"

    local wg_x=$((tn / bn))
    local wg_y=$((tm / bm))
    local entry
    entry=$(get_entry)
    local has_bias
    has_bias=$(get_has_bias)

    local pf_flag=""
    if [ "$pf" = "true" ]; then pf_flag="--prefetch"; fi

    if ! python3 "$GENERATOR" generate \
        --tile-m "$tm" --tile-n "$tn" --tile-k "$tk" \
        --block-m "$bm" --block-n "$bn" $pf_flag \
        --variant "$VARIANT" -o "$wave_file" 2>/dev/null; then
        echo "  $tag: GENERATE FAILED"
        echo "{\"tag\": \"$tag\", \"error\": \"generate\"}"
        return
    fi

    if ! "$WAVE_ASM_BIN" "$wave_file" -o "$wbin_file" 2>/dev/null; then
        echo "  $tag: ASSEMBLE FAILED"
        echo "{\"tag\": \"$tag\", \"error\": \"assemble\"}"
        return
    fi

    if ! "$WAVE_METAL_BIN" "$wbin_file" -o "$metal_file" 2>/dev/null; then
        echo "  $tag: METAL CODEGEN FAILED"
        echo "{\"tag\": \"$tag\", \"error\": \"codegen\"}"
        return
    fi

    local result
    result=$("$BENCH_BIN" "$metal_file" "$entry" "$tm" "$tn" "$wg_x" "$wg_y" "$has_bias" "$BENCH_SIZE" 2>/dev/null) || true

    if [ -z "$result" ]; then
        echo "  $tag: BENCH FAILED"
        echo "{\"tag\": \"$tag\", \"error\": \"bench\"}"
        return
    fi

    local gflops correct
    gflops=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('gflops',0))" 2>/dev/null) || gflops=0
    correct=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin).get('correct',False))" 2>/dev/null) || correct="False"

    if [ "$correct" = "True" ]; then
        printf "  %-40s  %8.1f GFLOPS  PASS\n" "$tag" "$gflops"
    else
        printf "  %-40s  %8s         FAIL\n" "$tag" "---"
    fi

    echo "{\"tag\": \"$tag\", \"tile_m\": $tm, \"tile_n\": $tn, \"tile_k\": $tk, \"block_m\": $bm, \"block_n\": $bn, \"prefetch\": $pf, $result}"
}

SANITY_CONFIGS=(
    "64 64 4 4 4 false"
    "128 128 8 8 8 true"
    "128 128 4 8 8 true"
    "256 256 4 8 8 true"
    "64 128 4 4 8 true"
)

RESULTS_FILE="$WORK_DIR/results_${VARIANT}.jsonl"
> "$RESULTS_FILE"

run_search() {
    local configs=("$@")
    local total=${#configs[@]}
    local idx=0
    local best_gflops=0
    local best_tag=""

    for cfg in "${configs[@]}"; do
        idx=$((idx + 1))
        echo "[$idx/$total]"
        local line
        line=$(run_config $cfg)
        echo "$line" | tail -1 >> "$RESULTS_FILE"

        local gf
        gf=$(echo "$line" | tail -1 | python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    if d.get('correct', False) or 'gflops' in d:
        g = d.get('gflops', 0)
        if isinstance(g, (int, float)):
            print(g)
        else:
            print(0)
    else:
        print(0)
except:
    print(0)
" 2>/dev/null) || gf=0

        if python3 -c "exit(0 if float('$gf') > float('$best_gflops') else 1)" 2>/dev/null; then
            best_gflops="$gf"
            best_tag=$(echo "$cfg" | awk '{
                tm=$1; tn=$2; tk=$3; bm=$4; bn=$5; pf=$6
                tag="t"tm"x"tn"_k"tk"_b"bm"x"bn
                if (pf == "true") tag=tag"_p"; else tag=tag"_np"
                print tag
            }')
        fi
    done

    echo ""
    echo "=== Results ==="
    echo "Total configs tested: $total"
    echo "Best: $best_tag  $best_gflops GFLOPS"
    echo ""
    echo "Top 5 configurations:"
    python3 -c "
import json, sys
results = []
for line in open('$RESULTS_FILE'):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        if d.get('correct', False) and 'gflops' in d:
            results.append(d)
    except:
        pass
results.sort(key=lambda x: x.get('gflops', 0), reverse=True)
for i, r in enumerate(results[:5]):
    print(f'  {i+1}. {r[\"tag\"]:40s}  {r[\"gflops\"]:8.1f} GFLOPS')
if not results:
    print('  No valid results.')
" 2>/dev/null || echo "  (could not parse results)"

    GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | sed 's/.*: //' || echo "Unknown GPU")
    CACHE_FILE="$CACHE_DIR/tuning_cache.json"

    python3 -c "
import json, os, sys

results = []
for line in open('$RESULTS_FILE'):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        if d.get('correct', False) and 'gflops' in d:
            results.append(d)
    except:
        pass

if not results:
    sys.exit(0)

results.sort(key=lambda x: x.get('gflops', 0), reverse=True)
best = results[0]

cache = {}
cf = '$CACHE_FILE'
if os.path.exists(cf):
    try:
        with open(cf) as f:
            cache = json.load(f)
    except:
        pass

gpu = '$GPU_NAME'
if gpu not in cache:
    cache[gpu] = {}

key = 'gemm_$VARIANT'
cache[gpu][key] = {
    'best': {
        'tile_m': best.get('tile_m', 0),
        'tile_n': best.get('tile_n', 0),
        'tile_k': best.get('tile_k', 0),
        'block_m': best.get('block_m', 0),
        'block_n': best.get('block_n', 0),
        'prefetch': best.get('prefetch', True),
        'gflops': best.get('gflops', 0),
        'tag': best.get('tag', ''),
    },
    'all_results': [
        {
            'tag': r.get('tag', ''),
            'gflops': r.get('gflops', 0),
            'correct': r.get('correct', False),
        }
        for r in results
    ],
}

with open(cf, 'w') as f:
    json.dump(cache, f, indent=2)
print(f'Saved to {cf}')
" 2>/dev/null || echo "(could not save cache)"

    REPO_RESULTS="$REPO_ROOT/benchmarks/tuning_results_${VARIANT}.json"
    if [ -f "$CACHE_FILE" ]; then
        cp "$CACHE_FILE" "$REPO_RESULTS" 2>/dev/null && echo "Copied to $REPO_RESULTS" || true
    fi
}

if [ "$MODE" = "sanity" ]; then
    echo "Running sanity check (5 configs)..."
    echo ""
    run_search "${SANITY_CONFIGS[@]}"

elif [ "$MODE" = "full" ]; then
    echo "Enumerating all valid configurations..."
    FULL_CONFIGS=()
    while IFS= read -r line; do
        tm=$(echo "$line" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['tile_m'])")
        tn=$(echo "$line" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['tile_n'])")
        tk=$(echo "$line" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['tile_k'])")
        bm=$(echo "$line" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['block_m'])")
        bn=$(echo "$line" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['block_n'])")
        pf=$(echo "$line" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(str(c['prefetch']).lower())")
        FULL_CONFIGS+=("$tm $tn $tk $bm $bn $pf")
    done < <(python3 "$GENERATOR" list --json | python3 -c "
import sys, json
for c in json.load(sys.stdin):
    print(json.dumps(c))
")
    echo "Total valid configs: ${#FULL_CONFIGS[@]}"
    echo ""
    run_search "${FULL_CONFIGS[@]}"

elif [ "$MODE" = "single" ]; then
    tm=$(echo "$SINGLE_CONFIG" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['tile_m'])")
    tn=$(echo "$SINGLE_CONFIG" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['tile_n'])")
    tk=$(echo "$SINGLE_CONFIG" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['tile_k'])")
    bm=$(echo "$SINGLE_CONFIG" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['block_m'])")
    bn=$(echo "$SINGLE_CONFIG" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(c['block_n'])")
    pf=$(echo "$SINGLE_CONFIG" | python3 -c "import sys,json; c=json.loads(sys.stdin.read()); print(str(c['prefetch']).lower())")
    run_search "$tm $tn $tk $bm $bn $pf"
fi
