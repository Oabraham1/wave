---
title: "Shuffle primitive"
description: "Exchange register values across lanes in a wave."
---

Shuffle reads a register value from another lane in the same wave. It supports reductions, scans, broadcasts, and sorting without a local-memory round trip.

```asm
wave_shuffle r0, r1, r2
```

Each active lane reads `r1` from the lane selected by its own `r2`, then writes the result to its own `r0`.

## Variants

| Instruction | Source lane |
|-------------|-------------|
| `wave_shuffle` | Lane index in the operand register |
| `wave_shuffle_up` | Current lane minus the offset |
| `wave_shuffle_down` | Current lane plus the offset |
| `wave_shuffle_xor` | Current lane XOR the mask |
| `wave_broadcast` | Shared source lane |

The [specification](/reference/spec-v04/) defines active-lane and source-lane requirements. Shuffle is a mandatory wave operation.

## Backend mapping

| Backend | Operations |
|---------|------------|
| Metal | `simd_shuffle`, `simd_shuffle_up`, `simd_shuffle_down`, `simd_shuffle_xor` |
| PTX | `shfl.sync` variants |
| HIP | `__shfl`, `__shfl_up`, `__shfl_down`, `__shfl_xor` |
| SYCL | `select_from_group`, group shifts, `permute_group_by_xor` |

The emulator gathers source values before writing destinations. This preserves cross-lane reads when source and destination registers overlap. A 64-element stack buffer holds the results for the supported wave widths.
