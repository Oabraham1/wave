// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// Section 4: Memory Model tests
// Verifies local memory, device memory, memory widths, and atomics.

use crate::harness::*;

pub fn run_tests() -> Vec<TestResult> {
    vec![
        test_local_memory_basics(),
        test_local_memory_widths(),
        test_device_memory_basics(),
        test_device_memory_widths(),
        test_atomic_add_i32(),
        test_atomic_sub(),
        test_atomic_min_max(),
        test_atomic_and_or_xor(),
        test_atomic_exchange(),
        test_atomic_cas(),
        test_local_atomic_add(),
        test_barrier_memory_visibility(),
    ]
}

/// Spec Section 4.2: Local memory is byte-addressable within workgroup.
fn test_local_memory_basics() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.2 Local Memory
; Claim: "Local Memory is a flat, byte-addressable array shared by all Waves in a Workgroup."

.kernel test_local_memory_basics
.registers 8
    mov r0, sr_lane_id

    ; Only lane 0 writes to local memory
    mov_imm r5, 0
    icmp_eq p1, r0, r5
    @p1 mov_imm r1, 0x42
    @p1 mov_imm r2, 100
    @p1 local_store_u32 r2, r1      ; local_mem[100] = 0x42

    ; Barrier to ensure write is visible
    barrier

    ; Only lane 1 reads from local memory
    mov_imm r5, 1
    icmp_eq p2, r0, r5
    @p2 mov_imm r2, 100
    @p2 local_load_u32 r3, r2       ; r3 = local_mem[100]
    @p2 mov_imm r4, 0
    @p2 device_store_u32 r4, r3     ; Write to device memory

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 0x42;

            TestResult {
                name: "test_local_memory_basics".to_string(),
                spec_section: "4.2".to_string(),
                spec_claim: "Local memory is shared within workgroup".to_string(),
                passed,
                details: if passed {
                    "Lane 1 read Lane 0's write correctly".to_string()
                } else {
                    format!("Read 0x{:08x}, expected 0x42", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_local_memory_basics".to_string(),
            spec_section: "4.2".to_string(),
            spec_claim: "Local memory is shared within workgroup".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.2: Local memory supports different access widths.
fn test_local_memory_widths() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.2 Local Memory
; Claim: "Supports 8, 16, 32, and 64 bit access widths."

.kernel test_local_memory_widths
.registers 8
    ; Store u32 value
    mov_imm r0, 0xDEADBEEF
    mov_imm r1, 0
    local_store_u32 r1, r0          ; local[0] = 0xDEADBEEF

    ; Read back as u32
    local_load_u32 r2, r1           ; r2 = local[0]
    mov_imm r3, 0
    device_store_u32 r3, r2         ; device[0] = r2

    ; Read as u16 (low 16 bits on little-endian)
    local_load_u16 r4, r1           ; r4 = 0xBEEF
    mov_imm r3, 4
    device_store_u32 r3, r4         ; device[4] = 0xBEEF

    ; Read as u8 (lowest byte)
    local_load_u8 r5, r1            ; r5 = 0xEF
    mov_imm r3, 8
    device_store_u32 r3, r5         ; device[8] = 0xEF

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let u32_val = read_u32(&result.device_memory, 0);
            let u16_val = read_u32(&result.device_memory, 4);
            let u8_val = read_u32(&result.device_memory, 8);

            let passed = u32_val == 0xDEADBEEF && u16_val == 0xBEEF && u8_val == 0xEF;

            TestResult {
                name: "test_local_memory_widths".to_string(),
                spec_section: "4.2".to_string(),
                spec_claim: "Local memory supports multiple widths".to_string(),
                passed,
                details: format!(
                    "u32=0x{:08x}, u16=0x{:04x}, u8=0x{:02x}",
                    u32_val, u16_val, u8_val
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_local_memory_widths".to_string(),
            spec_section: "4.2".to_string(),
            spec_claim: "Local memory supports multiple widths".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.3: Device memory basics.
fn test_device_memory_basics() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.3 Device Memory
; Claim: "Device Memory is byte-addressable with 64-bit virtual addresses."

.kernel test_device_memory_basics
.registers 8
    ; Write to different offsets
    mov_imm r0, 0x11111111
    mov_imm r1, 0
    device_store_u32 r1, r0

    mov_imm r0, 0x22222222
    mov_imm r1, 100
    device_store_u32 r1, r0

    mov_imm r0, 0x33333333
    mov_imm r1, 1000
    device_store_u32 r1, r0

    ; Read back and verify
    mov_imm r1, 0
    device_load_u32 r2, r1
    mov_imm r1, 2000
    device_store_u32 r1, r2         ; Copy to offset 2000

    mov_imm r1, 100
    device_load_u32 r2, r1
    mov_imm r1, 2004
    device_store_u32 r1, r2         ; Copy to offset 2004

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);
            let val100 = read_u32(&result.device_memory, 100);
            let val1000 = read_u32(&result.device_memory, 1000);
            let copy0 = read_u32(&result.device_memory, 2000);
            let copy100 = read_u32(&result.device_memory, 2004);

            let passed = val0 == 0x11111111
                && val100 == 0x22222222
                && val1000 == 0x33333333
                && copy0 == 0x11111111
                && copy100 == 0x22222222;

            TestResult {
                name: "test_device_memory_basics".to_string(),
                spec_section: "4.3".to_string(),
                spec_claim: "Device memory is byte-addressable".to_string(),
                passed,
                details: if passed {
                    "Writes and reads at multiple offsets work".to_string()
                } else {
                    format!(
                        "val0=0x{:x}, val100=0x{:x}, val1000=0x{:x}",
                        val0, val100, val1000
                    )
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_device_memory_basics".to_string(),
            spec_section: "4.3".to_string(),
            spec_claim: "Device memory is byte-addressable".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.3: Device memory supports different access widths.
fn test_device_memory_widths() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.3 Device Memory
; Claim: "Supports 8, 16, 32, 64, and 128 bit loads and stores."

.kernel test_device_memory_widths
.registers 8
    ; Store different widths
    mov_imm r0, 0xAB
    mov_imm r1, 0
    device_store_u8 r1, r0          ; device[0] = 0xAB

    mov_imm r0, 0xCDEF
    mov_imm r1, 16
    device_store_u16 r1, r0         ; device[16] = 0xCDEF

    mov_imm r0, 0x12345678
    mov_imm r1, 32
    device_store_u32 r1, r0         ; device[32] = 0x12345678

    ; Read back and verify
    mov_imm r1, 0
    device_load_u8 r2, r1
    mov_imm r1, 100
    device_store_u32 r1, r2         ; device[100] = u8 value

    mov_imm r1, 16
    device_load_u16 r2, r1
    mov_imm r1, 104
    device_store_u32 r1, r2         ; device[104] = u16 value

    mov_imm r1, 32
    device_load_u32 r2, r1
    mov_imm r1, 108
    device_store_u32 r1, r2         ; device[108] = u32 value

    halt
.end
"#;

    let config = EmulatorConfig::default();

    match run_test(SOURCE, [1, 1, 1], [1, 1, 1], &config, None) {
        Ok(result) => {
            let u8_val = read_u32(&result.device_memory, 100);
            let u16_val = read_u32(&result.device_memory, 104);
            let u32_val = read_u32(&result.device_memory, 108);

            let passed = u8_val == 0xAB && u16_val == 0xCDEF && u32_val == 0x12345678;

            TestResult {
                name: "test_device_memory_widths".to_string(),
                spec_section: "4.3".to_string(),
                spec_claim: "Device memory supports multiple widths".to_string(),
                passed,
                details: format!(
                    "u8=0x{:02x}, u16=0x{:04x}, u32=0x{:08x}",
                    u8_val, u16_val, u32_val
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_device_memory_widths".to_string(),
            spec_section: "4.3".to_string(),
            spec_claim: "Device memory supports multiple widths".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomic add on i32.
fn test_atomic_add_i32() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "atomic_add atomically adds a value."
; Method: All threads atomically add 1 to device_mem[0]. Result should equal thread count.

.kernel test_atomic_add_i32
.registers 4
    mov_imm r0, 0
    mov_imm r1, 1
    atomic_add r2, r0, r1, .device  ; device_mem[0] += 1
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Initialize device memory to 0
    let init_mem = vec![0u8; 4];

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 4; // 4 threads each added 1

            TestResult {
                name: "test_atomic_add_i32".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "atomic_add works correctly".to_string(),
                passed,
                details: if passed {
                    format!("Final value = {} (correct)", value)
                } else {
                    format!("Final value = {}, expected 4", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_atomic_add_i32".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "atomic_add works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomic sub.
fn test_atomic_sub() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "atomic_sub atomically subtracts a value."

.kernel test_atomic_sub
.registers 4
    mov_imm r0, 0
    mov_imm r1, 1
    atomic_sub r2, r0, r1, .device  ; device_mem[0] -= 1
    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Initialize to 100
    let mut init_mem = vec![0u8; 4];
    init_mem[0..4].copy_from_slice(&100u32.to_le_bytes());

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 96; // 100 - 4 = 96

            TestResult {
                name: "test_atomic_sub".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "atomic_sub works correctly".to_string(),
                passed,
                details: if passed {
                    format!("100 - 4 = {} (correct)", value)
                } else {
                    format!("100 - 4 = {}, expected 96", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_atomic_sub".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "atomic_sub works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomic min/max.
fn test_atomic_min_max() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "atomic_min/max compute minimum/maximum."

.kernel test_atomic_min_max
.registers 8
    mov r0, sr_lane_id

    ; Each thread tries to set min to its lane_id
    mov_imm r1, 0               ; address for min
    atomic_min r2, r1, r0, .device

    ; Each thread tries to set max to its lane_id
    mov_imm r1, 4               ; address for max
    atomic_max r2, r1, r0, .device

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Init: min = 100, max = 0
    let mut init_mem = vec![0u8; 8];
    init_mem[0..4].copy_from_slice(&100u32.to_le_bytes());
    init_mem[4..8].copy_from_slice(&0u32.to_le_bytes());

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let min_val = read_u32(&result.device_memory, 0);
            let max_val = read_u32(&result.device_memory, 4);

            // Min should be 0 (lane 0's value), max should be 3 (lane 3's value)
            let passed = min_val == 0 && max_val == 3;

            TestResult {
                name: "test_atomic_min_max".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "atomic_min/max work correctly".to_string(),
                passed,
                details: format!("min={}, max={} (expected 0, 3)", min_val, max_val),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_atomic_min_max".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "atomic_min/max work correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomic bitwise operations.
fn test_atomic_and_or_xor() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "atomic_and/or/xor perform bitwise operations."

.kernel test_atomic_and_or_xor
.registers 8
    mov r0, sr_lane_id

    ; Compute 1 << lane_id
    mov_imm r1, 1
    shl r2, r1, r0              ; r2 = 1 << lane_id

    ; Atomic OR: device[0] |= (1 << lane_id)
    mov_imm r3, 0
    atomic_or r4, r3, r2, .device

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    let init_mem = vec![0u8; 4];

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            // Should be 0b1111 = 0xF (bits 0,1,2,3 all set)
            let passed = value == 0xF;

            TestResult {
                name: "test_atomic_and_or_xor".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "atomic_or works correctly".to_string(),
                passed,
                details: format!("Result = 0x{:x} (expected 0xf)", value),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_atomic_and_or_xor".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "atomic_or works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomic exchange.
fn test_atomic_exchange() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "atomic_exchange atomically swaps values."

.kernel test_atomic_exchange
.registers 8
    mov r0, sr_lane_id
    mov_imm r1, 0

    ; Each thread exchanges its lane_id, gets the previous value
    mov_imm r6, 100
    iadd r2, r0, r6             ; value to write = lane_id + 100
    atomic_exchange r3, r1, r2, .device  ; r3 = old value

    ; Write old value to device memory at offset (lane_id + 1) * 4
    mov_imm r4, 1
    iadd r5, r0, r4
    mov_imm r4, 2
    shl r5, r5, r4              ; offset = (lane_id + 1) * 4
    device_store_u32 r5, r3

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Init device[0] = 42
    let mut init_mem = vec![0u8; 64];
    init_mem[0..4].copy_from_slice(&42u32.to_le_bytes());

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            // One thread got 42, others got the value from the previous thread
            // Final value should be one of 100, 101, 102, 103
            let final_val = read_u32(&result.device_memory, 0);
            let passed = final_val >= 100 && final_val <= 103;

            TestResult {
                name: "test_atomic_exchange".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "atomic_exchange works correctly".to_string(),
                passed,
                details: format!("Final value = {} (expected 100-103)", final_val),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_atomic_exchange".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "atomic_exchange works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomic compare-and-swap.
fn test_atomic_cas() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "atomic_cas compares and conditionally swaps."
; Method: Initialize to 5. Lane 0 does CAS(5 -> 10). Should succeed.

.kernel test_atomic_cas
.registers 8
    mov r0, sr_lane_id

    ; Only lane 0 does CAS
    mov_imm r7, 0
    icmp_eq p1, r0, r7
    @p1 mov_imm r1, 0           ; address
    @p1 mov_imm r2, 5           ; expected
    @p1 mov_imm r3, 10          ; desired
    @p1 atomic_cas r4, r1, r2, r3, .device   ; r4 = old value

    ; Store old value (should be 5)
    @p1 mov_imm r5, 100
    @p1 device_store_u32 r5, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // Init device[0] = 5
    let mut init_mem = vec![0u8; 128];
    init_mem[0..4].copy_from_slice(&5u32.to_le_bytes());

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, Some(&init_mem)) {
        Ok(result) => {
            let final_val = read_u32(&result.device_memory, 0);
            let old_val = read_u32(&result.device_memory, 100);

            // CAS should succeed: final = 10, old = 5
            let passed = final_val == 10 && old_val == 5;

            TestResult {
                name: "test_atomic_cas".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "atomic_cas works correctly".to_string(),
                passed,
                details: format!(
                    "final={}, old={} (expected 10, 5)",
                    final_val, old_val
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_atomic_cas".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "atomic_cas works correctly".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 4.5: Atomics work on local memory.
fn test_local_atomic_add() -> TestResult {
    const SOURCE: &str = r#"
; Section: 4.5 Atomics
; Claim: "Atomics work on local memory."

.kernel test_local_atomic_add
.registers 8
    ; Initialize local_mem[0] = 0 (only lane 0)
    mov r0, sr_lane_id
    mov_imm r7, 0
    icmp_eq p1, r0, r7
    @p1 mov_imm r1, 0
    @p1 mov_imm r2, 0
    @p1 local_store_u32 r1, r2

    ; Barrier
    barrier

    ; All threads atomic add 1
    mov_imm r1, 0
    mov_imm r2, 1
    local_atomic_add r3, r1, r2

    ; Barrier
    barrier

    ; Lane 0 reads result and writes to device memory
    @p1 mov_imm r1, 0
    @p1 local_load_u32 r4, r1
    @p1 device_store_u32 r1, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    match run_test(SOURCE, [1, 1, 1], [4, 1, 1], &config, None) {
        Ok(result) => {
            let value = read_u32(&result.device_memory, 0);
            let passed = value == 4;

            TestResult {
                name: "test_local_atomic_add".to_string(),
                spec_section: "4.5".to_string(),
                spec_claim: "Local memory atomics work".to_string(),
                passed,
                details: if passed {
                    format!("4 threads added 1, result = {}", value)
                } else {
                    format!("Result = {}, expected 4", value)
                },
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_local_atomic_add".to_string(),
            spec_section: "4.5".to_string(),
            spec_claim: "Local memory atomics work".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}

/// Spec Section 6.13: Barrier ensures memory visibility.
fn test_barrier_memory_visibility() -> TestResult {
    const SOURCE: &str = r#"
; Section: 6.13 Synchronization
; Claim: "Memory operations before barrier are visible after barrier."

.kernel test_barrier_memory_visibility
.registers 8
    mov r0, sr_wave_id
    mov r1, sr_lane_id
    mov_imm r7, 0

    ; Wave 0, Lane 0: write values to local memory
    ; Combined check: (wave_id | lane_id) == 0
    or r5, r0, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r2, 0x11111111
    @p3 mov_imm r3, 0
    @p3 local_store_u32 r3, r2

    @p3 mov_imm r2, 0x22222222
    @p3 mov_imm r3, 4
    @p3 local_store_u32 r3, r2

    ; Barrier
    barrier

    ; Wave 1, Lane 0: read values and write to device memory
    ; Combined check: wave_id == 1 AND lane_id == 0
    ; Compute: (wave_id - 1) | lane_id == 0
    mov_imm r6, 1
    isub r5, r0, r6
    or r5, r5, r1
    icmp_eq p3, r5, r7

    @p3 mov_imm r3, 0
    @p3 local_load_u32 r4, r3
    @p3 device_store_u32 r3, r4

    @p3 mov_imm r3, 4
    @p3 local_load_u32 r4, r3
    @p3 device_store_u32 r3, r4

    halt
.end
"#;

    let config = EmulatorConfig {
        wave_width: 4,
        ..Default::default()
    };

    // 2 waves = 8 threads
    match run_test(SOURCE, [1, 1, 1], [8, 1, 1], &config, None) {
        Ok(result) => {
            let val0 = read_u32(&result.device_memory, 0);
            let val4 = read_u32(&result.device_memory, 4);

            let passed = val0 == 0x11111111 && val4 == 0x22222222;

            TestResult {
                name: "test_barrier_memory_visibility".to_string(),
                spec_section: "6.13".to_string(),
                spec_claim: "Barrier ensures memory visibility".to_string(),
                passed,
                details: format!(
                    "val0=0x{:x}, val4=0x{:x} (expected 0x11111111, 0x22222222)",
                    val0, val4
                ),
                cycles: result.cycles,
            }
        }
        Err(e) => TestResult {
            name: "test_barrier_memory_visibility".to_string(),
            spec_section: "6.13".to_string(),
            spec_claim: "Barrier ensures memory visibility".to_string(),
            passed: false,
            details: format!("Execution error: {}", e),
            cycles: 0,
        },
    }
}
