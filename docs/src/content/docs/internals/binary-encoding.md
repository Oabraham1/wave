---
title: Binary Encoding
description: Deep dive into the WAVE 32-bit and 64-bit instruction encoding formats and the WBIN container layout.
---

The WAVE binary encoding packs every instruction into a fixed 32-bit word, with an optional second word for extended operations, balancing decode simplicity against expressiveness.

## 32-Bit Base Format

Every WAVE instruction fits into a single 32-bit word with the following bit layout:

| Bits | Field | Width | Purpose |
|------|-------|-------|---------|
| 31:26 | `opcode` | 6 bits | Operation class (up to 64 opcodes) |
| 25:21 | `RD` | 5 bits | Destination register (0--31) |
| 20:16 | `RS1` | 5 bits | Source register 1 (0--31) |
| 15:11 | `RS2` | 5 bits | Source register 2 (0--31) |
| 10:7 | `modifier` | 4 bits | Operation variant within opcode class (0--15) |
| 6:5 | `scope` | 2 bits | Memory scope (wave, workgroup, device, system) |
| 4:3 | `predicate` | 2 bits | Predicate register selector |
| 2 | `pred_neg` | 1 bit | Negate predicate condition |
| 1:0 | `flags` | 2 bits | Instruction-specific flags |

```
 31  26 25  21 20  16 15  11 10   7 6  5 4  3  2  1  0
┌──────┬──────┬──────┬──────┬──────┬────┬────┬───┬────┐
│opcode│  RD  │ RS1  │ RS2  │ mod  │scop│pred│neg│flag│
│ 6b   │ 5b   │ 5b   │ 5b   │ 4b   │ 2b │ 2b │1b │ 2b│
└──────┴──────┴──────┴──────┴──────┴────┴────┴───┴────┘
```

### Why 6-bit opcodes?

Six bits provide 64 opcode slots. WAVE currently uses fewer than 40, leaving room for future extensions without changing the encoding width. Wider opcodes would steal bits from register fields or modifiers; narrower opcodes would constrain the instruction set too early.

### Why 5-bit register fields?

Five bits address 32 registers per thread. This is a deliberate trade-off: 32 registers are sufficient for the vast majority of GPU kernels (validated against shader compiler statistics from all four target vendors), while keeping the instruction word at 32 bits. The v0.1 spec text incorrectly referenced 256 registers despite using 5-bit fields; this was identified and corrected as a spec defect (see [Spec Defects](/internals/spec-defects/)).

### Why a 4-bit modifier?

The modifier field disambiguates variants within an opcode class. For example, the `FUnaryOp` opcode class uses the modifier to select between `fsqrt` (0), `frsqrt` (1), ... `fsin` (8), `fcos` (9), `fexp2` (10), `flog2` (11). Four bits encode values 0--15, covering all current variants with room to grow. The earlier 3-bit modifier could only encode 0--7, which caused a concrete encoding failure documented in [Modifier Field Evolution](/internals/modifier-field/).

### Scope encoding

The 2-bit scope field encodes memory ordering visibility:

| Value | Scope | Meaning |
|-------|-------|---------|
| `00` | Wave | Visible within the executing wave |
| `01` | Workgroup | Visible to all waves in the workgroup |
| `10` | Device | Visible to all waves on the device |
| `11` | System | Visible to the device and host CPU |

## 64-Bit Extended Format

When an instruction needs more than two source registers or an inline immediate, it uses a second 32-bit word:

```
Word 0: standard 32-bit encoding (as above)
Word 1 (extended):
 31  27 26  22 21                              0
┌──────┬──────┬──────────────────────────────────┐
│ RS3  │ RS4  │          reserved / imm           │
│ 5b   │ 5b   │            22b                    │
└──────┴──────┴──────────────────────────────────┘
```

Alternatively, the entire second word can serve as a 32-bit immediate value. The opcode in word 0 determines interpretation: if the opcode is in the extended class, the decoder fetches a second word; otherwise, the instruction is complete at 32 bits.

### When to use 64-bit encoding

- **Fused multiply-add** (`fma r0, r1, r2, r3`): needs RS1, RS2, and RS3.
- **Immediate loads** (`imm r0, 0x3F800000`): the 32-bit immediate occupies the entire second word.
- **Atomic compare-and-swap** (`atom.cas r0, r1, r2, r3`): needs address, expected, desired, and result registers.

## WBIN Container Format

Compiled WAVE programs are distributed as `.wbin` (WAVE Binary) files. The container is designed for direct memory-mapping without parsing overhead.

### Header (32 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | ASCII `"WAVE"` (0x57415645) |
| 4 | 4 | `version` | Container format version |
| 8 | 4 | `code_offset` | Byte offset to code section |
| 12 | 4 | `code_size` | Size of code section in bytes |
| 16 | 4 | `symbol_offset` | Byte offset to symbol table |
| 20 | 4 | `symbol_size` | Size of symbol table in bytes |
| 24 | 4 | `metadata_offset` | Byte offset to metadata section |
| 28 | 4 | `metadata_size` | Size of metadata section in bytes |

### Kernel Metadata

Each kernel entry in the metadata section describes the resource requirements that a backend needs for dispatch:

| Field | Purpose |
|-------|---------|
| `name` | Kernel identifier (null-terminated UTF-8) |
| `register_count` | Number of registers used per thread |
| `local_mem_size` | Bytes of local (workgroup-shared) memory required |
| `workgroup_size_x` | Workgroup dimension X |
| `workgroup_size_y` | Workgroup dimension Y |
| `workgroup_size_z` | Workgroup dimension Z |
| `code_offset` | Byte offset of this kernel's code within the code section |
| `code_size` | Size of this kernel's code in bytes |

### Design rationale

The WBIN format is intentionally minimal. It carries exactly the information a backend runtime needs to allocate resources and dispatch kernels. There is no debug info, no relocations, and no linking metadata in the base format --- these are relegated to optional extension sections identified by the metadata table. This keeps the critical path (load, allocate, dispatch) as fast as possible.
