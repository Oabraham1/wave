# wave-hip

AMD HIP backend for the WAVE ISA. Translates `.wbin` binaries to HIP C++ (`.hip`) for AMD GPUs via ROCm.

## Build

```
cd wave-hip && cargo build --release
```

## Usage

```
wave-hip program.wbin -o program.hip
```

## Pipeline

```bash
wave-asm vector_add.wave -o vector_add.wbin
wave-hip vector_add.wbin -o vector_add.hip
hipcc examples/run_on_amd.hip -o run_on_amd
./run_on_amd
```

Requires ROCm and `hipcc`.

## Instruction Mapping

| WAVE | HIP |
|------|-----|
| Registers | `uint32_t` local variables |
| Float bitcast | `__uint_as_float()` / `__float_as_uint()` via `rf()` / `ri()` helpers |
| Local memory | `extern __shared__ uint8_t local_mem[]` (dynamic shared memory) |
| Device memory | `uint8_t*` pointer arithmetic |
| Barrier | `__syncthreads()` |
| Fence | `__threadfence_block()` / `__threadfence()` / `__threadfence_system()` |
| Shuffle | `__shfl()` / `__shfl_up()` / `__shfl_down()` / `__shfl_xor()` |
| Reduce | Butterfly tree via `__shfl_down()` with `for` loop over `warpSize` |
| Prefix sum | `__shfl_up()` with predicated accumulation over `warpSize` |
| Ballot | `__ballot()` |
| Vote | `__any()` / `__all()` |
| Atomics | `atomicAdd()` / `atomicSub()` / `atomicCAS()` etc. |
| Bit ops | `__popc()` / `__ffs()` / `__brev()` |
| Control flow | Native C++ `if`/`else`/`while` |
| Predication | Wrapped in `if (pN) { ... }` guards |
| Special registers | `threadIdx.x`, `blockIdx.x`, `blockDim.x`, `__lane_id()`, `warpSize` |

## Wavefront Width

Generated code uses `warpSize` everywhere instead of hardcoding 32. This makes the output correct on both architectures:

| Architecture | GPUs | Wavefront |
|-------------|------|-----------|
| RDNA | RX 7000, RX 6000 | 32 |
| CDNA | MI250, MI300X | 64 |

Reduction trees and prefix sums use `for` loops bounded by `warpSize` rather than unrolled steps, so the same generated code adapts to either width at runtime.

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](../LICENSE) for terms.
