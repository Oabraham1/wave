# Metal measurements

The harnesses compile WAVE assembly to Metal and measure GPU command-buffer duration. Run them on macOS 15 or later with the Metal SDK and cached Cargo dependencies.

Run the fixed GEMM workloads and an MPS matrix multiplication reference from the repository root:

```sh
python3 benchmarks/autotune/run_metal.py --work-dir /path/to/new-artifacts
```

The output directory must not exist. It contains generated sources, input assembly, binaries, diagnostics, a source-hash manifest, and JSONL results. Use `--prepare-only` to build without GPU execution, then run `--run-prepared /path/to/artifacts`. Prepared artifacts are checked against their recorded hashes. Existing result files are not overwritten.

Select matrix sizes and sampling parameters explicitly:

```sh
python3 benchmarks/autotune/run_metal.py \
    --sizes 128,256,512,1024,2048,4096 \
    --warmup 5 --iterations 20 --seed 42 \
    --work-dir /path/to/new-artifacts
```

Each JSONL record retains the GPU timing samples in acquisition order, the median, matrix shape, dispatch dimensions, numerical types, compiler settings, source hash, and output validation. GPU timestamps exclude source compilation, allocation, transfers performed before dispatch, and reference computation. GFLOPS uses `2*M*N*K`; fused epilogue operations are excluded from that count.

Every measured size is checked in full against Accelerate SGEMM. Inputs use a deterministic SplitMix64 generator over `[-1, 1)`. Binary16 workloads quantize the input before both GPU execution and the reference computation. Both paths accumulate in binary32, apply the epilogue, and round the output to binary16. The sigmoid GELU approximation uses `x / (1 + exp2(-c*x))`, where `c` is the binary32 value with bits `0x401D1463` and `x` includes a bias of `0.5`. This is an approximation to GELU.

An element passes when `abs(actual-reference) <= atol + rtol*abs(reference)`. Binary32 uses `atol=0.001` and `rtol=0.0001`. Binary16 uses `atol=0.002` and `rtol=0.001`. These tolerances accommodate rounding and reduction-order differences. Results report maximum absolute error, error divided by the tolerance bound, failed elements, and nonfinite elements. Unwritten output starts as NaN. Nonfinite output, GPU errors, invalid timestamps, missing kernels, and validation failures produce a nonzero exit status.

The fixed kernels require square matrices whose dimensions are multiples of 128. Measurements from one process describe that session. Estimate uncertainty across independent runs before making comparative performance claims. MPS implements plain binary32 GEMM; it is a library reference for that workload, not an equivalent fused or binary16 workload.

## Tuning

```sh
benchmarks/autotune/autotune.sh --sanity --variant f32 \
    --bench-size 4096 --warmup 5 --iterations 20 \
    --work-dir /path/to/new-tuning-artifacts
```

`--full` enumerates supported configurations. `--config` accepts one JSON object with `tile_m`, `tile_n`, `tile_k`, `block_m`, `block_n`, and `prefetch`. Candidate order is shuffled with `--order-seed`. The manifest records that order. `selection.json` selects the largest median-sample GFLOPS among configurations with complete output validation. All candidate records and raw samples remain in `results.jsonl`. Any candidate failure makes the overall command fail.

## Validation helpers

```sh
python3 -m unittest discover -s benchmarks/autotune -p 'test_*.py'
swiftc -O -Xcc -DACCELERATE_NEW_LAPACK \
    benchmarks/autotune/BenchmarkSupport.swift \
    benchmarks/autotune/test_benchmark_support.swift \
    -framework Metal -framework MetalPerformanceShaders -framework Accelerate \
    -o /tmp/wave-benchmark-tests
/tmp/wave-benchmark-tests
```
