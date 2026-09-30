---
title: "Occupancy equation"
description: "The mathematical relationship between register usage, wave width, and GPU occupancy across vendor architectures."
---

Register capacity bounds the number of waves that can reside on a compute unit. Local memory, workgroup placement, allocation granularity, and hardware limits can lower actual occupancy.

## The equation

```
O = floor(F / (R * W * w))
```

Where:

| Symbol | Meaning | Unit |
|--------|---------|------|
| O | Register-capacity bound on concurrent waves per compute unit | waves |
| F | Register file size per compute unit | bytes |
| R | Machine registers used per lane | registers |
| W | Wave width; threads per wave | threads |
| w | Register width | bytes |

The floor accounts for allocating register storage to whole waves.

## Why this equation matters

Resident waves can provide work while other waves wait for memory. More occupancy can help hide latency, but does not guarantee higher throughput.

Increasing register demand reduces this capacity bound. Doubling demand from 32 to 64 registers approximately halves the bound when register storage is the limiting resource. Other limits can prevent actual occupancy from changing.

## Vendor calculations

The calculations below use the stated parameter assumptions. They illustrate the formula, not measured occupancy or a complete resource model for each architecture.

### NVIDIA: Ampere SM

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

The examples give register-capacity bounds of 8 and 64 waves. Allocation granularity and other target limits must be applied separately.

### AMD: RDNA 3 CU

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

With these assumptions, changing `W` from 64 to 32 doubles the register-capacity bound. It does not establish a doubling of measured occupancy.

### Intel: Xe-HPG EU

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

The units must refer to the same allocation domain. A SIMD-wide register must be converted to per-lane storage before using this scalar-register formula.

### Apple: M1 GPU core

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

These assumed values give a bound of 13 SIMD-groups. They do not establish the occupancy of a compiled Metal kernel.

## Implications for WAVE

### Register allocation strategy

For a target capacity of `O_target` waves, the same model gives a register budget:

```
R_max = floor(F / (O_target * W * w))
```

This is an analytical budget, not an automatic WAVE launch policy. The WAVE allocator rejects demand above its configured register limit; it does not insert spills. The vendor compiler determines machine-register allocation.

### Runtime queries

Use target-specific resource information when estimating occupancy. Wave width, register storage, and allocation granularity can vary within a vendor's product family. See [Thin abstraction](/internals/thin-abstraction/).

### Cross-vendor portability

The same WAVE binary can produce different machine-register usage and occupancy on different GPUs. WBIN metadata records its WAVE register count, not the vendor compiler's allocation. Inspect the compiled target program when applying the model. See [Binary encoding](/internals/binary-encoding/).
