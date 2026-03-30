# WAVE Specification Verification Test Suite

This test suite verifies the WAVE v0.1 specification against the reference implementation (wave-asm assembler and wave-emu emulator).

## Purpose

The verification suite:
- Tests every claim in the WAVE specification
- Traces each test to a specific spec section
- Uses self-checking tests (assembly writes results to memory, harness verifies)
- Includes real GPU-style programs as stress tests

## Dependencies

Requires `wave-asm` (assembler) and `wave-emu` (emulator) from the parent workspace.

## Running Tests

```bash
# Run the full verification suite
./run_all.sh

# Or manually:
cd tests/spec-verification
cargo build --release
./target/release/run-spec-tests
```

## Test Sections

| Section | Description | Tests |
|---------|-------------|-------|
| Section 2 | Execution Model | Thread identity, wave width, workgroup dimensions, grid, forward progress |
| Section 3 | Register Model | GPRs, predicate registers, special registers, mov_imm |
| Section 4 | Memory Model | Local/device memory, atomics, barriers |
| Section 5 | Control Flow | Branching, divergence, predication, reconvergence |
| Section 6 | Instructions | Arithmetic, logical, bitwise, comparison, memory, wave ops |
| Section 7 | Capabilities | Wave width query, max registers, local memory size, F64 support |
| Section 8 | Binary Encoding | WBIN format, instruction alignment, immediate/register encoding |
| Section 9 | Conformance | Determinism, halt behavior, bounds checking |
| Real Programs | End-to-end | Dot product, matrix multiply, etc. |
| Stress Tests | GPU workloads | Vector add, reduction, histogram, transpose |

## Test Result Interpretation

- **PASS**: The emulator behavior matches the specification claim
- **FAIL**: The emulator behavior differs from the specification

Failed tests indicate either:
1. Bugs in the emulator implementation
2. Gaps in the assembler instruction support
3. Ambiguities in the specification that need clarification

## Adding New Tests

Each test follows this pattern:

```rust
fn test_example() -> TestResult {
    const SOURCE: &str = r#"
; Section: X.Y Description
; Claim: "Quoted spec claim being tested."

.kernel test_example
.registers 8
    ; Assembly code that writes results to device memory
    mov_imm r0, 42
    mov_imm r1, 0
    device_store_u32 r1, r0
    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 42;

            TestResult {
                name: "test_example".to_string(),
                spec_section: "X.Y".to_string(),
                spec_claim: "Description of what's being tested".to_string(),
                passed,
                details: format!("Got {}, expected 42", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_example".to_string(),
            spec_section: "X.Y".to_string(),
            spec_claim: "Description of what's being tested".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
```

## Test Harness API

```rust
// Run a test with assembly source
fn run_test(
    asm_source: &str,           // WAVE assembly source code
    grid: [u32; 3],             // Grid dimensions (workgroup count)
    workgroup: [u32; 3],        // Workgroup dimensions (thread count)
    config: &EmulatorConfig,    // Emulator configuration
    initial_device_memory: Option<&[u8]>,  // Pre-initialized device memory
) -> Result<ExecutionResult, TestError>

// Read values from device memory
fn read_u32(memory: &[u8], offset: usize) -> u32
fn read_i32(memory: &[u8], offset: usize) -> i32
fn read_f32(memory: &[u8], offset: usize) -> f32

// Emulator configuration
struct EmulatorConfig {
    wave_width: usize,          // Default: 4
    max_cycles: u64,            // Default: 100000
    device_memory_size: usize,  // Default: 65536
    local_memory_size: usize,   // Default: 16384
}
```

## License

Copyright (c) 2026 Ojima Abraham. All rights reserved.
Licensed under the Apache License, Version 2.0.
