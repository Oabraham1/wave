---
title: "Backends"
description: "Translate WAVE binaries to Metal, PTX, HIP, and SYCL."
---

Each backend reads `.wbin` and emits input for a vendor compiler. The vendor toolchain produces executable GPU code.

## Backend summary

| Backend | Output | Target | Toolchain |
|---|---|---|---|
| `wave-metal` | Metal Shading Language | Apple GPUs | Metal compiler and runtime |
| `wave-ptx` | PTX assembly | NVIDIA GPUs | CUDA `ptxas` |
| `wave-hip` | HIP C++ | AMD RDNA and CDNA GPUs | ROCm `hipcc` |
| `wave-sycl` | SYCL C++ | Intel Xe and Arc GPUs | oneAPI DPC++ |

## wave-metal: Apple

The Metal backend emits MSL compute functions.

### Register mapping

Kernel metadata determines the number of local `uint` registers. Floating-point operations reinterpret their bits with `as_type<float>`.

### Memory mapping

Local memory uses a `threadgroup` byte array sized from kernel metadata. Device memory uses one `device` byte-buffer parameter. Instructions access that allocation through byte offsets.

### Wave operations

| WAVE instruction | MSL operation |
|---|---|
| `wave_shuffle` | `simd_shuffle` |
| `wave_shuffle_xor` | `simd_shuffle_xor` |
| `wave_reduce_add` | `simd_sum` |
| `wave_ballot` | SIMD-group ballot |
| `wave_broadcast` | `simd_broadcast` |

### Synchronization

A barrier emits `threadgroup_barrier` with both threadgroup and device memory flags. Fences emit `atomic_thread_fence` at SIMD-group, threadgroup, or device scope with sequentially consistent ordering. System-scope fences are rejected.

### Limitations

The backend uses 32-lane wave behavior. Other widths require compatible lowering. Metal does not provide the double-precision arithmetic needed by WAVE `f64` operations. Calls requiring function extraction are rejected.

## wave-ptx: NVIDIA

The PTX backend emits assembly for the selected shader model. The measurement suite targets SM 75.

### Register mapping

WAVE register bits use `.b32` registers. Floating-point operations move those bits to `.f32` registers. Wide values combine consecutive WAVE registers into 64-bit temporaries.

```text
mov.b32 %f0, %r0;
```

Separate floating-point addition, subtraction, and multiplication use explicit rounding modifiers to preserve their rounding boundaries. Fused multiply-add remains an explicit operation.

### Memory mapping

Local memory uses a PTX `.shared` allocation. Device accesses add WAVE byte offsets to the 64-bit device-memory base pointer and use `ld.global` or `st.global`.

### Atomic operations

Local and device atomics use `atom.shared` and `atom.global`. Subtraction negates the operand before atomic addition. Compare-and-swap supplies separate expected and desired operands.

### Limitations

The selected PTX version and shader model constrain the available instructions. The vendor assembler checks whether the generated program is valid for that target.

## wave-hip: AMD

The HIP backend emits C++ kernels compiled with `hipcc`.

### Register mapping

WAVE registers become local `uint32_t` values. `__uint_as_float` and `__float_as_uint` reinterpret binary32 values.

### Memory mapping

Local memory uses a dynamically sized shared allocation whose size is supplied at launch. Device memory uses a byte-pointer argument.

### Wave operations

| WAVE instruction | HIP operation |
|---|---|
| `wave_shuffle` | `__shfl` |
| `wave_shuffle_xor` | `__shfl_xor` |
| `wave_ballot` | `__ballot` |
| `wave_reduce_add` | Shuffle-tree reduction |
| `wave_broadcast` | Shuffle from a shared source lane |

The backend queries `warpSize`. A kernel's collective algorithm must still support the resulting 32- or 64-lane grouping.

### Synchronization

Workgroup barriers use `__syncthreads`. Fences use `__threadfence_block`, `__threadfence`, or `__threadfence_system` according to scope.

### Limitations

Reduction lowering uses a sequence of shuffle and arithmetic operations. Register allocation and scheduling depend on the target compiler. RDNA and CDNA wave widths can change the communication pattern and resource cost.

## wave-sycl: Intel

The SYCL backend emits a launch function that submits a kernel through a SYCL queue.

### Register mapping

Local `uint32_t` values hold WAVE registers. `sycl::bit_cast` reinterprets floating-point bits.

### Memory mapping

A `local_accessor<uint8_t, 1>` supplies workgroup-local memory. A USM byte pointer supplies device memory. The generated launch function constructs the command group and dispatch dimensions.

### Wave operations

Sub-group operations implement lane selection, shifts, XOR permutation, and reduction. The kernel obtains its sub-group from `nd_item::get_sub_group`.

### Atomic operations

The backend uses `atomic_ref` for memory updates and `atomic_fence` for scoped ordering.

### Limitations

The generated kernel does not request a fixed subgroup size. The target chooses it, so kernels must use compatible collective algorithms. Local-memory accessors also require host-side command-group construction.

## Cross-backend comparison

### Float bitcasting

| Backend | Mechanism |
|---|---|
| Metal | `as_type<float>` |
| PTX | `mov.b32` |
| HIP | `__uint_as_float` |
| SYCL | `bit_cast<float>` |

### Shared memory declaration

| Backend | Storage |
|---|---|
| Metal | `threadgroup` byte array |
| PTX | `.shared` byte allocation |
| HIP | Dynamic `__shared__` byte allocation |
| SYCL | `local_accessor` |

### Barrier instructions

| Backend | Operation |
|---|---|
| Metal | `threadgroup_barrier` |
| PTX | `bar.sync 0` |
| HIP | `__syncthreads` |
| SYCL | `group_barrier` |

See [Emulator](/architecture/emulator/) for CPU execution of the same binary format.
