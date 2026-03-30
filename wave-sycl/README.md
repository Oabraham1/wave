# wave-sycl

Intel SYCL backend for the WAVE ISA. Translates `.wbin` binaries to SYCL C++ (`.cpp`) for Intel GPUs via oneAPI.

## Build

```
cd wave-sycl && cargo build --release
```

## Usage

```
wave-sycl program.wbin -o program.cpp
```

## Pipeline

```bash
wave-asm vector_add.wave -o vector_add.wbin
wave-sycl vector_add.wbin -o vector_add.cpp
icpx -fsycl -O3 examples/run_on_intel.cpp -o run_on_intel
./run_on_intel
```

Requires Intel oneAPI and `icpx -fsycl`.

## Instruction Mapping

| WAVE | SYCL |
|------|------|
| Registers | `uint32_t` local variables |
| Float bitcast | `sycl::bit_cast<float>()` / `sycl::bit_cast<uint32_t>()` via `rf()` / `ri()` helpers |
| Local memory | `sycl::local_accessor<uint8_t, 1>` → raw `uint8_t*` pointer |
| Device memory | USM pointer via `sycl::malloc_device` |
| Barrier | `sycl::group_barrier(it.get_group())` |
| Fence | `sycl::atomic_fence(order, scope)` with separate order and scope |
| Shuffle | `sycl::select_from_group(sg, val, lane)` |
| Shuffle up/down | `sycl::shift_group_right(sg, val, delta)` / `shift_group_left()` |
| Shuffle XOR | `sycl::permute_group_by_xor(sg, val, mask)` |
| Broadcast | `sycl::group_broadcast(sg, val, lane)` |
| Reduce | `sycl::reduce_over_group(sg, val, plus<>())` / `minimum<>()` / `maximum<>()` |
| Prefix sum | `sycl::exclusive_scan_over_group(sg, val, plus<>())` |
| Ballot | `sycl::reduce_over_group(sg, bit, bit_or<>())` (emulated) |
| Vote | `sycl::any_of_group(sg, pred)` / `sycl::all_of_group(sg, pred)` |
| Atomics | `sycl::atomic_ref<T, order, scope, address_space>` |
| Math | `sycl::sqrt()`, `sycl::sin()`, `sycl::cos()`, `sycl::exp2()`, etc. |
| Control flow | Native C++ `if`/`else`/`while` |
| Predication | Wrapped in `if (pN) { ... }` guards |
| Special registers | `it.get_local_id(0)`, `it.get_group(0)`, `sg.get_local_id()[0]`, etc. |

## Sub-group Operations

SYCL 2020 provides first-class group algorithms that replace the multi-step shuffle trees needed by PTX and HIP:

| Operation | PTX / HIP | SYCL |
|-----------|-----------|------|
| Reduce add | 5-step butterfly shfl tree | `reduce_over_group(sg, val, plus<>())` |
| Reduce min | 5-step butterfly shfl tree | `reduce_over_group(sg, val, minimum<>())` |
| Prefix sum | 5-step shfl_up + shift | `exclusive_scan_over_group(sg, val, plus<>())` |

One function call instead of 10+ instructions.

## Sub-group Width

Intel GPUs use sub-group widths of 8 or 16, narrower than NVIDIA/AMD/Apple (32 or 64). The generated code queries width at runtime via `sg.get_max_local_range()[0]` and never hardcodes a specific width. Programs that use `sr_wave_width` get the correct value on any hardware.

| Platform | Sub-group Width |
|----------|----------------|
| Intel Arc (RDNA-like) | 8, 16, or 32 |
| Intel Data Center GPU Max | 16 or 32 |
| Apple Silicon (via Metal) | 32 |
| NVIDIA (via PTX) | 32 |
| AMD RDNA (via HIP) | 32 |
| AMD CDNA (via HIP) | 64 |

## Kernel Structure

Unlike Metal/HIP (standalone kernel functions) or PTX (assembly), SYCL uses a lambda-based dispatch model. The generated code is a launch function containing nested lambdas:

```
function → queue::submit → handler::parallel_for → nd_item lambda → kernel body
```

The kernel body lives at indent level 4 inside the lambda.

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.

Apache License, Version 2.0. See [LICENSE](../LICENSE) for terms.
