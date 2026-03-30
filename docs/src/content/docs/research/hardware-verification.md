---
title: Hardware Verification
description: Verification results and methodology for WAVE across GPU vendors and architectures.
---

WAVE's correctness claim rests on end-to-end hardware verification: a single WAVE binary is compiled to each vendor's native GPU instruction format and executed on real hardware, and the output is compared against a known-correct reference.

## Verification Results

| Vendor | GPU | Architecture | Native Target | Status | Output |
|--------|-----|--------------|---------------|--------|--------|
| Apple | M4 Pro | Apple GPU (Unified) | Metal IR | **Verified** | `c[i] = 3i` |
| NVIDIA | T4 | Turing | PTX | **Verified** | `c[i] = 3i` |
| AMD | MI300X | CDNA 3 | GCN ISA | **Verified** | `c[i] = 3i` |
| Intel | *(pending)* | Xe | *(pending)* | **Pending** | - |

All verified GPUs produce identical output for the same input binary.

## Test Kernel: `vector_add`

The verification kernel is a straightforward vector addition:

```
// Pseudocode
kernel vector_add(a: &[f32], b: &[f32], c: &mut [f32]) {
    let i = global_thread_id();
    c[i] = a[i] + b[i];
}
```

The input buffers are initialized so that `a[i] = i` and `b[i] = 2i` for every element. The expected output is therefore `c[i] = 3i` for all `i`.

This kernel exercises several of WAVE's hardware-invariant primitive categories:

- **Device (Global) Memory** - loads from `a` and `b`, store to `c`.
- **Floating-Point Arithmetic** - the addition itself.
- **Thread Identification** - mapping each thread to its global index.

## Verification Methodology

Verification proceeds in four stages:

### 1. Compile

The WAVE binary for `vector_add` is compiled ahead of time by `wave-compiler` into each vendor's native instruction format:

- **Apple:** Metal IR, loaded via the Metal framework.
- **NVIDIA:** PTX assembly, loaded via the CUDA Driver API.
- **AMD:** GCN ISA binary, loaded via the ROCm/HIP runtime.

### 2. Dispatch

The host-side runtime (`wave-runtime`) allocates input and output buffers on the target GPU, uploads the input data (`a[i] = i`, `b[i] = 2i`), and dispatches the compiled kernel.

### 3. Readback

After the kernel completes, the runtime copies the output buffer `c` back to host memory.

### 4. Validate

Every element of the output buffer is compared against the reference value `c[i] = 3i`. Verification passes if and only if every element matches exactly. Floating-point equality is exact here because the inputs and expected outputs are all representable without rounding error in IEEE 754 single precision.

## What "Verified" Means

A GPU is marked **Verified** when:

1. The `vector_add` WAVE binary compiles without error to the vendor's native format.
2. The compiled kernel executes on real hardware (not an emulator or simulator).
3. The output buffer matches the reference output element-for-element.

Verification demonstrates that WAVE's portable encoding, the compiler's code generation for that target, and the runtime's dispatch and memory management all function correctly end to end. It does not by itself constitute a performance benchmark or a guarantee of correctness for all possible kernels - it is a necessary baseline that confirms the toolchain works on real silicon.

## Pending Verification

**Intel Xe** - WAVE includes a code generation path for Intel Xe GPUs, but hardware verification has not yet been performed. Contributions of verification results on Intel hardware are welcome. See the [Contributing](/research/contributing/) page for details.
