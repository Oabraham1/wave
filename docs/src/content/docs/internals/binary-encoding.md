---
title: Binary Encoding
description: Deep dive into the WAVE 32-bit and 64-bit instruction encoding formats and the WBIN container layout.
---

The WAVE binary encoding packs every instruction into a fixed 32-bit word, with an optional second word for extended operations, balancing decode simplicity against expressiveness.

## Word0: 32-Bit Base Format

Every WAVE instruction starts with a 32-bit word with the following bit layout:

| Bits | Field | Width | Purpose |
|------|-------|-------|---------|
| 31:24 | `opcode` | 8 bits | Operation class |
| 23:16 | `RD` | 8 bits | Destination register (0--255) |
| 15:8 | `RS1` | 8 bits | Source register 1 (0--255) |
| 7:4 | `modifier` | 4 bits | Operation variant within opcode class (0--15) |
| 3 | `reserved` | 1 bit | Reserved, must be zero |
| 2 | `pred_neg` | 1 bit | Negate predicate condition |
| 1:0 | `pred_reg` | 2 bits | Predicate register index (0=p0, 1=p1, 2=p2, 3=p3) |

```
 31    24 23    16 15     8 7    4  3  2  1  0
┌────────┬────────┬────────┬──────┬───┬───┬────┐
│ opcode │   RD   │  RS1   │ mod  │rsv│neg│pred│
│  8b    │  8b    │  8b    │ 4b   │1b │1b │ 2b │
└────────┴────────┴────────┴──────┴───┴───┴────┘
```

### Predicate encoding

When pred_reg=0 and pred_neg=0, the instruction is unconditional. When pred_reg is nonzero, the instruction executes only if the specified predicate register is true (or false, if pred_neg=1). For example, `@p1` sets pred_reg=1, pred_neg=0; `@!p2` sets pred_reg=2, pred_neg=1.

This encoding was introduced in v0.4. Earlier versions used bits [3:0] for scope and flags, which silently dropped all predication. See [Spec Defects](/internals/spec-defects/) for the full history (Defect 4).

### Why 8-bit register fields?

Eight bits address 256 registers per thread. In practice, most kernels use far fewer, but the wide field simplifies encoding and avoids the v0.1 mismatch between spec text and register field width. Two 8-bit register fields (rd, rs1) fit in word0; additional source registers (rs2, rs3, rs4) are encoded in word1.

### Why a 4-bit modifier?

The modifier field disambiguates variants within an opcode class. For example, the `FUnaryOp` opcode class uses the modifier to select between `frsqrt` (0), `frcp` (1), ... `fsin` (8), `fcos` (9), `fexp2` (10), `flog2` (11). Four bits encode values 0--15, covering all current variants with room to grow. The earlier 3-bit modifier could only encode 0--7, which caused a concrete encoding failure documented in [Modifier Field Evolution](/internals/modifier-field/).

The Control opcode (0x3F) uses modifier values 0--7 for ControlOp (if, else, endif, loop, break, continue, endloop, call) and values 8--15 for SyncOp (return, halt, barrier, fence variants, wait, nop), offset by `SYNC_MODIFIER_OFFSET = 8`.

## Word1: 32-Bit Extended Format

When an instruction needs additional source registers, a memory scope, or an inline immediate, it uses a second 32-bit word. The opcode in word0 determines whether word1 is present.

| Bits | Field | Width | Purpose |
|------|-------|-------|---------|
| 31:24 | `RS2` | 8 bits | Source register 2 (0--255) |
| 23:16 | `RS3` | 8 bits | Source register 3 (0--255) |
| 15:8 | `RS4` | 8 bits | Source register 4 (0--255) |
| 7:2 | `reserved` | 6 bits | Reserved, must be zero |
| 1:0 | `scope` | 2 bits | Memory scope (00=wave, 01=workgroup, 10=device, 11=system) |

```
 31    24 23    16 15     8 7         2 1  0
┌────────┬────────┬────────┬──────────┬────┐
│  RS2   │  RS3   │  RS4   │ reserved │scop│
│  8b    │  8b    │  8b    │   6b     │ 2b │
└────────┴────────┴────────┴──────────┴────┘
```

Alternatively, the entire word1 can serve as a 32-bit immediate value (e.g., for `mov_imm`). The opcode determines interpretation.

### Scope encoding

The 2-bit scope field in word1 encodes memory ordering visibility for scoped instructions (DeviceAtomic, fence):

| Value | Scope | Meaning |
|-------|-------|---------|
| `00` | Wave | Visible within the executing wave |
| `01` | Workgroup | Visible to all waves in the workgroup |
| `10` | Device | Visible to all waves on the device |
| `11` | System | Visible to the device and host CPU |

In v0.3, scope was encoded in word0. It was moved to word1 in v0.4 to free bits for predicate encoding.

### When to use 64-bit encoding

- **Two-source operations** (`iadd r0, r1, r2`): needs RS1 (word0) and RS2 (word1).
- **Fused multiply-add** (`fma r0, r1, r2, r3`): needs RS1, RS2, and RS3.
- **Immediate loads** (`mov_imm r0, 0x3F800000`): the 32-bit immediate occupies the entire word1.
- **Scoped atomics** (`device_atomic_add r0, r1, r2, workgroup`): scope in word1 bits [1:0].
- **Atomic compare-and-swap**: needs address, expected, desired, and result registers.

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
