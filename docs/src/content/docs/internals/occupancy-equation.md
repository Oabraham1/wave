---
title: Occupancy Equation
description: The mathematical relationship between register usage, wave width, and GPU occupancy across vendor architectures.
---

Occupancy --- the number of concurrent waves a compute unit can sustain --- is governed by a single equation that applies uniformly across all four target GPU architectures.

## The Equation

```
O = floor(F / (R * W * w))
```

Where:

| Symbol | Meaning | Unit |
|--------|---------|------|
| **O** | Occupancy (concurrent waves per compute unit) | waves |
| **F** | Register file size per compute unit | bytes |
| **R** | Registers used per thread | registers |
| **W** | Wave width (threads per wave) | threads |
| **w** | Register width | bytes |

The floor function reflects the fact that partial waves cannot be scheduled: you either have enough registers for an entire wave or you do not.

## Why This Equation Matters

Occupancy directly determines a GPU's ability to hide memory latency. When one wave stalls on a memory access, the hardware switches to another wave. Higher occupancy means more waves available for latency hiding, which generally means higher throughput for memory-bound kernels.

The equation reveals a fundamental tension: **more registers per thread means fewer concurrent waves**. A kernel that uses 64 registers achieves half the occupancy of one that uses 32 registers, all else being equal. The WAVE compiler's register allocator must balance register pressure against occupancy.

## Vendor Calculations

### NVIDIA (Ampere SM)

```
F = 256 KB = 262,144 bytes
R = 255 (maximum registers per thread)
W = 32 (warp width)
w = 4 bytes (32-bit registers)

O = floor(262,144 / (255 * 32 * 4))
  = floor(262,144 / 32,640)
  = floor(8.03)
  = 8 waves/SM (theoretical max)
```

With typical register usage of 32 registers:

```
O = floor(262,144 / (32 * 32 * 4)) = floor(64) = 64 warps/SM
```

In practice, NVIDIA Ampere SMs support up to 64 warps, so register pressure is rarely the only bottleneck. At maximum register usage (255 registers), occupancy drops to approximately 7 warps.

### AMD (RDNA 3 CU)

```
F = 512 KB = 524,288 bytes
R = 256 (maximum VGPRs per thread)
W = 64 (wavefront width, Wave64 mode)
w = 4 bytes

O = floor(524,288 / (256 * 64 * 4))
  = floor(524,288 / 65,536)
  = floor(8.0)
  = 8 wavefronts/CU (at maximum register usage)
```

AMD's larger register file compensates for the wider wave width. In Wave32 mode (W=32), occupancy doubles.

### Intel (Xe-HPG EU)

```
F = 128 KB = 131,072 bytes
R = 128 (maximum GRF registers per thread)
W = 16 (sub-group width, SIMD16)
w = 4 bytes

O = floor(131,072 / (128 * 16 * 4))
  = floor(131,072 / 8,192)
  = floor(16.0)
  = 16 threads/EU (at maximum register usage)
```

Intel's narrower sub-groups and smaller register file per EU yield high occupancy per EU, though each EU is smaller than an NVIDIA SM or AMD CU.

### Apple (M1 GPU Core)

```
F = 208 KB = 212,992 bytes
R = 128 (maximum registers per thread)
W = 32 (SIMD-group width)
w = 4 bytes

O = floor(212,992 / (128 * 32 * 4))
  = floor(212,992 / 16,384)
  = floor(13.0)
  = 13 SIMD-groups (at maximum register usage)
```

Apple's GPU achieves relatively high occupancy even at maximum register pressure, reflecting a design that favors thread-level parallelism.

## Implications for WAVE

### Register allocation strategy

The WAVE compiler can target a specific occupancy by capping register usage. If a backend specifies that a kernel should achieve at least O=4 waves, the compiler can compute the maximum register budget:

```
R_max = floor(F / (O_target * W * w))
```

Registers beyond this budget are spilled to local memory.

### Runtime queries

Because W, F, and w vary across (and sometimes within) architectures, WAVE does not hardcode these values. The runtime provides query functions that return the target's parameters, and the compiler uses them to make occupancy-aware decisions. This is an instance of the [Thin Abstraction](/internals/thin-abstraction/) principle: define what to optimize for, not what the hardware parameters are.

### Cross-vendor portability

The same kernel binary, with the same register count, will achieve different occupancy on different GPUs. This is expected and correct --- the equation shows that occupancy is a function of hardware parameters that WAVE cannot and should not try to normalize. Instead, WAVE ensures that the register count is recorded in the WBIN metadata (see [Binary Encoding](/internals/binary-encoding/)), allowing the runtime to compute occupancy and make dispatch decisions accordingly.
