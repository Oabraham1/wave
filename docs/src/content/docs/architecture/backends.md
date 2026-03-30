---
title: Backends
description: How the four WAVE backends translate .wbin binaries to Metal, PTX, HIP, and SYCL.
---

Each WAVE backend reads a `.wbin` binary and translates it into vendor-native GPU code that can be compiled and dispatched by the vendor's own toolchain.

## Backend Summary

| | wave-metal | wave-ptx | wave-hip | wave-sycl |
|---|---|---|---|---|
| **Output format** | Metal Shading Language (MSL) | PTX assembly | HIP C++ | SYCL C++ |
| **Target hardware** | Apple M1-M4, A-series | NVIDIA Turing+ (SM 75+) | AMD RDNA / CDNA | Intel Xe / Arc |
| **Vendor toolchain** | `metallib` / Xcode | `ptxas` / CUDA toolkit | `hipcc` / ROCm | `dpcpp` / oneAPI |
| **Wave width** | 32 (fixed) | 32 (warpSize) | 64 (RDNA: 32/64) | varies (subgroup) |
| **F64 support** | No | Yes | Yes | Yes |

## wave-metal (Apple)

The Metal backend generates MSL compute kernel functions.

### Register Mapping

WAVE's 32 general-purpose registers map to local `uint32_t` variables:

```cpp
uint32_t r0 = 0, r1 = 0, r2 = 0, /* ... */ r31 = 0;
```

Floating-point operations use `as_type<float>()` for bitwise reinterpretation between `uint32_t` and `float`, preserving WAVE's untyped register model.

### Memory Mapping

- **Local (shared) memory** maps to `threadgroup uint8_t shared_mem[N]`, where `N` is the local memory size declared in the `.wbin` metadata. Loads and stores use pointer casts at the access site.
- **Device (global) memory** maps to `device uint8_t* buf [[buffer(0)]]` passed as a kernel argument. The backend generates byte-addressed loads and stores with explicit casts to the target type.

### Wave Operations

WAVE wave-level operations map to Metal's SIMD-group functions:

| WAVE instruction | MSL function |
|---|---|
| `wave_shuffle` | `simd_shuffle()` |
| `wave_reduce_add` | `simd_sum()` |
| `wave_ballot` | `simd_ballot()` |
| `wave_broadcast` | `simd_broadcast()` |

### Synchronization

Barriers translate to `threadgroup_barrier(mem_flags::mem_threadgroup)` for local memory scope and `threadgroup_barrier(mem_flags::mem_device)` for device memory scope.

### Limitations

- **No F64 support.** Apple GPUs do not support double-precision floating-point arithmetic. Kernels using `f64` types will fail at the Metal backend with a diagnostic error.
- **Fixed wave width of 32.** Metal SIMD-groups are always 32 lanes wide. The backend hardcodes this value rather than querying it at runtime.
- **Buffer binding model.** Device memory is passed via Metal buffer bindings (`[[buffer(N)]]`), which limits the number of distinct device memory arguments.

## wave-ptx (NVIDIA)

The PTX backend generates PTX assembly text targeting SM 75 (Turing) or later.

### Register Mapping

WAVE registers map to PTX virtual registers with type-specific prefixes:

| WAVE type | PTX register | PTX type |
|---|---|---|
| `i32` / `u32` | `%r0` - `%r31` | `.b32` |
| `f32` | `%f0` - `%f31` | `.f32` |
| `f64` | `%rd0` - `%rd31` | `.b64` |

Float bitcasting uses `mov.b32` to reinterpret between integer and float register types without conversion:

```
mov.b32 %f0, %r0;   // reinterpret uint32 as float
```

### Memory Mapping

- **Local (shared) memory** maps to `.shared .b8 shared_mem[N];` in the PTX `.shared` state space.
- **Device (global) memory** uses 64-bit addressing with `ld.global` and `st.global` instructions:

```
ld.global.b32 %r0, [%rd0];       // load 32 bits from global address in %rd0
st.global.b32 [%rd0], %r1;       // store 32 bits to global address
```

### Atomic Operations

PTX atomics use the `atom.global` and `atom.shared` instructions with explicit scope:

```
atom.global.add.u32 %r0, [%rd0], %r1;   // global atomic add
atom.shared.cas.b32 %r0, [%r2], %r3, %r4;  // shared CAS
```

PTX does not provide `atom.sub`. The backend emits `neg` followed by `atom.add` as an equivalent sequence.

### Limitations

- **SM 75+ required.** The backend uses features introduced in Turing (e.g., uniform registers, independent thread scheduling). Older GPUs are not supported.
- **No atomic subtract.** Implemented as negate-then-add, which is functionally equivalent but uses two instructions.

## wave-hip (AMD)

The HIP backend generates HIP C++ kernel functions compilable with `hipcc`.

### Register Mapping

WAVE registers map to local `uint32_t` variables, identical to the Metal approach:

```cpp
uint32_t r0 = 0, r1 = 0, /* ... */ r31 = 0;
```

Float bitcasting uses AMD-specific intrinsics:

```cpp
float f = __uint_as_float(r0);    // uint32 → float
uint32_t u = __float_as_uint(f);  // float → uint32
```

### Memory Mapping

- **Local (shared) memory** maps to dynamically-sized shared memory declared with `extern __shared__ uint8_t shared_mem[];`. The actual size is set at kernel launch.
- **Device (global) memory** is passed as a raw pointer parameter (`uint8_t* __restrict__ buf`).

### Wave Operations

WAVE wave-level operations map to HIP intrinsics:

| WAVE instruction | HIP function |
|---|---|
| `wave_shuffle` | `__shfl()` |
| `wave_ballot` | `__ballot()` |
| `wave_reduce_add` | Manual shuffle-tree reduction |
| `wave_broadcast` | `__shfl(val, src_lane)` |

The backend uses `warpSize` rather than hardcoding the wave width, allowing correct execution on both RDNA (wave32/wave64 configurable) and CDNA (wave64) hardware.

### Synchronization

Barriers map to `__syncthreads()` for workgroup scope. The backend does not currently emit wave-level barriers as AMD hardware provides implicit wave-level synchrony.

### Limitations

- **Wave reduce is emulated.** HIP does not expose a single-instruction wave reduce. The backend generates a butterfly reduction tree using `__shfl_xor()`.
- **RDNA vs. CDNA wave width.** While the backend uses `warpSize` for correctness, performance tuning may differ between wave32 (RDNA) and wave64 (CDNA) modes.

## wave-sycl (Intel)

The SYCL backend generates SYCL C++ kernel functions targeting the oneAPI DPC++ compiler.

### Register Mapping

WAVE registers map to local `uint32_t` variables:

```cpp
uint32_t r0 = 0, r1 = 0, /* ... */ r31 = 0;
```

Float bitcasting uses the SYCL standard library:

```cpp
float f = sycl::bit_cast<float>(r0);
uint32_t u = sycl::bit_cast<uint32_t>(f);
```

### Memory Mapping

- **Local (shared) memory** maps to `sycl::local_accessor<uint8_t, 1>` bound during command group submission. The accessor is passed to the kernel as a parameter.
- **Device (global) memory** uses SYCL Unified Shared Memory (USM) pointers (`uint8_t*`), allocated with `sycl::malloc_device()` and passed directly to the kernel.

### Wave Operations

WAVE wave-level operations map to SYCL sub-group functions:

| WAVE instruction | SYCL function |
|---|---|
| `wave_shuffle` | `sycl::select_from_group(sg, val, lane)` |
| `wave_reduce_add` | `sycl::reduce_over_group(sg, val, sycl::plus<>())` |
| `wave_ballot` | `sycl::group_ballot(sg, pred)` |
| `wave_broadcast` | `sycl::group_broadcast(sg, val, lane)` |

### Atomic Operations

SYCL atomics use `sycl::atomic_ref` with explicit memory order and scope:

```cpp
sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space> ref(buf[offset]);
ref.fetch_add(val);
```

Memory fences use `sycl::atomic_fence()` with configurable scope matching WAVE's four-level scoping model (wave, workgroup, device, system).

### Limitations

- **Sub-group size variability.** Intel GPUs support multiple sub-group sizes (8, 16, 32). The backend requests a specific size via `[[intel::reqd_sub_group_size(N)]]` but the actual size depends on the target hardware.
- **Local memory model.** SYCL's accessor-based local memory model differs from the raw pointer model used by Metal, PTX, and HIP. The backend must manage accessor creation during command group submission, which adds complexity to the host-side dispatch code.

## Cross-Backend Comparison

### Float Bitcasting

Every backend must reinterpret `uint32_t` register values as `float` without numeric conversion. Each uses a different mechanism:

| Backend | Mechanism |
|---|---|
| wave-metal | `as_type<float>(r)` |
| wave-ptx | `mov.b32 %f, %r` |
| wave-hip | `__uint_as_float(r)` |
| wave-sycl | `sycl::bit_cast<float>(r)` |

### Shared Memory Declaration

| Backend | Declaration |
|---|---|
| wave-metal | `threadgroup uint8_t shared_mem[N]` |
| wave-ptx | `.shared .b8 shared_mem[N]` |
| wave-hip | `extern __shared__ uint8_t shared_mem[]` |
| wave-sycl | `sycl::local_accessor<uint8_t, 1>` |

### Barrier Instructions

| Backend | Instruction |
|---|---|
| wave-metal | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| wave-ptx | `bar.sync 0` |
| wave-hip | `__syncthreads()` |
| wave-sycl | `sycl::group_barrier(wg)` |

**Next:** [Emulator](/architecture/emulator/) for how `wave-emu` executes `.wbin` binaries without a GPU.
