---
title: Introduction
description: WAVE execution model and toolchain.
---

WAVE is a universal GPU instruction set architecture. Kernels written in Python, Rust, C++, or TypeScript compile to a common `.wbin` binary. Backends translate that binary for Apple, NVIDIA, AMD, and Intel GPUs.

## Why WAVE

CUDA, Metal, HIP, and SYCL use different toolchains. WAVE defines shared operations and an execution model that each backend maps to its target. Hardware parameters, such as wave width, remain explicit.

## Shared primitives

The cross-vendor analysis identifies 11 categories of shared operations:

| Category | Operations |
|---|---|
| Integer arithmetic | Addition, subtraction, multiplication, division, and remainder |
| Floating-point arithmetic | Addition, multiplication, division, and fused multiply-add |
| Bitwise operations | Boolean operations, shifts, and bit counting |
| Comparison and selection | Comparisons, predicates, and conditional selection |
| Local memory | Access to workgroup scratchpad memory |
| Device memory | Access to device-wide memory |
| Atomic operations | Indivisible read-modify-write operations |
| Wave operations | Shuffles, reductions, and ballots |
| Control flow | Structured branches, loops, and calls |
| Synchronization | Barriers and scoped fences |
| Type conversion | Conversions between numeric types |

Special registers expose thread and workgroup identifiers, dimensions, and other execution parameters. See the [instruction reference](/reference/instruction-set/) for mnemonics and operands.

## Compilation and execution

```text
Source kernel
    |
wave-compiler
    |
WAVE binary
    |
Backend translator
    |
Vendor compiler and runtime
    |
GPU
```

The compiler produces WAVE binaries. The assembler `wave-asm` also produces binaries from assembly, and `wave-dis` disassembles them. The CPU emulator `wave-emu` executes binaries for functional testing.

| Backend | Target | Generated code |
|---|---|---|
| `wave-metal` | Apple GPUs | Metal Shading Language |
| `wave-ptx` | NVIDIA GPUs | PTX |
| `wave-hip` | AMD GPUs | HIP C++ |
| `wave-sycl` | Intel GPUs | SYCL C++ |

GPU dispatch depends on the SDK and installed vendor runtime. The [SDK guides](/guides/rust-sdk/) describe the available execution paths.

## Binary format

Instructions occupy one or two 32-bit words. The encoding records the operation, operands, and modifiers. The [specification](/reference/spec/) defines the instruction fields, register model, and binary layout.

## Memory model

Memory operations and fences use four scopes: wave, workgroup, device, and system. Scope determines which participants an ordering operation covers. The [memory model guide](/guides/memory-model/) explains how to combine memory access, barriers, and fences.

## Research and hardware

The [preprint](/research/papers/) presents the cross-vendor analysis behind WAVE. Hardware verification includes Apple M4 Pro, NVIDIA T4, and AMD MI300X. The [hardware verification page](/research/hardware-verification/) describes the matched Apple and NVIDIA benchmark study and its measurement scope.

Continue to [Installation](/getting-started/installation/) to build the toolchain.
