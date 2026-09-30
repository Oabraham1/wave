---
title: "Hardware verification"
description: "Hardware execution, independent references, and measurement scope."
---

WAVE hardware verification includes Apple M4 Pro, NVIDIA T4, and AMD MI300X. The matched benchmark study described here measures Apple M4 Pro and NVIDIA T4 execution against independent references and native kernels.

## Hardware experiments

The experiments execute on an Apple M4 Pro and an NVIDIA Tesla T4. The [benchmark runners](https://github.com/Oabraham1/wave/tree/main/benchmarks/gpu) use the same WAVE binaries, input allocations, and references across both GPUs and the emulator. Matched native Metal and CUDA kernels use the same algorithms and dispatch geometry.

Both devices execute vector addition, divergent arithmetic, gather, uniform and skewed histograms, wave reductions, sparse matrix-vector multiplication and a five-point stencil. Validation checks the complete allocation, including inputs and guard regions. Floating-point inputs use bounded binary fractions with exact reference results. Non-finite outputs fail validation.

A comparison runs WAVE, native Metal and Slang in five fresh Apple processes with randomized variant order. The paired reduction experiment compares shuffle and local-memory communication with the same arithmetic and output. Its largest case gives a local-memory-to-shuffle time ratio of 1.855, with a 95% interval of 1.813 to 1.919 across process estimates.

The [NVIDIA comparison](https://github.com/Oabraham1/wave/tree/main/benchmarks/gpu/nvidia) uses five separate processes on one T4 allocation and measures execution with CUDA events. WAVE-to-native ratios range from 0.934 to 1.019 across the nine workloads at the largest size. Every process checks an input that distinguishes separate arithmetic from fused multiply-add. Both execution paths produce the exact expected bits, and disassembly confirms matching arithmetic in the measured workloads. The runner uses a standalone Driver API harness.

The separate [GEMM suite](https://github.com/Oabraham1/wave/tree/main/benchmarks/autotune) checks every output element against Accelerate SGEMM using the values stored in the input format. It reports the absolute and relative tolerance, errors, GPU timing samples, and workload configuration.

## Source-to-device path

The runtime example compiles a source kernel, translates its WAVE binary, dispatches it through Metal, and checks all outputs and unchanged inputs. Its sizes include incomplete workgroups and multiple workgroups.

```sh
cargo run --release --manifest-path wave-runtime/Cargo.toml --example metal_validation -- 3
```

The Metal backend emits Metal Shading Language. The host compiles it through the Metal API. Runtime measurements include the host compiler and temporary-file path; device-only benchmarks use a resident harness with compilation outside the timed region.

## Scope

These measurements apply to the evaluated kernels, inputs, software, and devices. Full ISA conformance and performance across other workloads require separate checks. The matched benchmark study does not include AMD or Intel timings. See [Supported GPUs](/getting-started/supported-gpus/) for hardware coverage.
