// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Round-trip integration tests. Assembles WAVE source code using wave-asm,
//!
//! loads the binary into wave-emu, executes it, and verifies the results.

use wave_emu::{Emulator, EmulatorConfig};
use wave_asm::assemble;

fn assemble_and_run(source: &str, config: EmulatorConfig) -> Emulator {
    let result = assemble(source, "test.wave").expect("assembly failed");
    let mut emulator = Emulator::new(config);
    emulator.load_binary(&result.binary).expect("failed to load binary");
    emulator.run().expect("execution failed");
    emulator
}

#[test]
fn test_roundtrip_special_register_lane_id() {
    let source = r#"
.kernel test_lane_id
.registers 8
.workgroup_size 4, 1, 1
    mov r0, sr_lane_id
    mov_imm r1, 2
    shl r2, r0, r1           ; r2 = lane_id * 4
    device_store_u32 r2, r0  ; store lane_id at address (lane_id * 4)
    halt
.end
"#;

    let config = EmulatorConfig {
        grid_dim: [1, 1, 1],
        workgroup_dim: [4, 1, 1],
        wave_width: 4,
        device_memory_size: 1024,
        ..Default::default()
    };

    let emulator = assemble_and_run(source, config);

    for i in 0u64..4 {
        let value = emulator.device_memory().read_u32(i * 4).unwrap();
        assert_eq!(value, i as u32, "Address {} should contain lane_id {}", i * 4, i);
    }
}

#[test]
fn test_roundtrip_special_register_thread_id() {
    let source = r#"
.kernel test_thread_id
.registers 8
.workgroup_size 4, 2, 1
    mov r0, sr_thread_id_x
    mov r1, sr_thread_id_y
    mov r3, sr_workgroup_size_x

    ; Calculate linear index: y * workgroup_size_x + x
    imul r4, r1, r3          ; r4 = y * workgroup_size_x
    iadd r5, r4, r0          ; r5 = linear_index

    ; Store x at address linear_index * 8
    ; Store y at address linear_index * 8 + 4
    mov_imm r6, 3
    shl r7, r5, r6           ; r7 = linear_index * 8
    device_store_u32 r7, r0  ; store thread_id_x

    mov_imm r6, 4
    iadd r7, r7, r6          ; r7 += 4
    device_store_u32 r7, r1  ; store thread_id_y

    halt
.end
"#;

    let config = EmulatorConfig {
        grid_dim: [1, 1, 1],
        workgroup_dim: [4, 2, 1],  // 8 threads total
        wave_width: 8,
        device_memory_size: 1024,
        ..Default::default()
    };

    let emulator = assemble_and_run(source, config);

    for y in 0u32..2 {
        for x in 0u32..4 {
            let linear_index = y * 4 + x;
            let base_addr = u64::from(linear_index * 8);
            let stored_x = emulator.device_memory().read_u32(base_addr).unwrap();
            let stored_y = emulator.device_memory().read_u32(base_addr + 4).unwrap();
            assert_eq!(stored_x, x, "Thread ({x},{y}): expected thread_id_x={x}, got {stored_x}");
            assert_eq!(stored_y, y, "Thread ({x},{y}): expected thread_id_y={y}, got {stored_y}");
        }
    }
}

#[test]
fn test_roundtrip_special_register_wave_width() {
    let source = r#"
.kernel test_wave_width
.registers 4
.workgroup_size 4, 1, 1
    mov r0, sr_wave_width
    mov_imm r1, 0
    device_store_u32 r1, r0  ; All threads store wave_width at address 0
    halt
.end
"#;

    let config = EmulatorConfig {
        grid_dim: [1, 1, 1],
        workgroup_dim: [4, 1, 1],
        wave_width: 4,
        device_memory_size: 1024,
        ..Default::default()
    };

    let emulator = assemble_and_run(source, config);

    let value = emulator.device_memory().read_u32(0).unwrap();
    assert_eq!(value, 4, "wave_width should be 4, got {value}");
}

#[test]
fn test_roundtrip_special_register_workgroup_id() {
    let source = r#"
.kernel test_workgroup_id
.registers 8
.workgroup_size 1, 1, 1
    mov r0, sr_workgroup_id_x
    mov r1, sr_workgroup_id_y
    mov r2, sr_workgroup_id_z

    ; Calculate linear workgroup index: z * grid_y * grid_x + y * grid_x + x
    ; For 2x2x1 grid: linear = y * 2 + x
    mov r3, sr_grid_size_x
    imul r4, r1, r3          ; r4 = y * grid_size_x
    iadd r5, r4, r0          ; r5 = linear_index

    ; Store workgroup_id_x at address linear_index * 12
    ; Store workgroup_id_y at address linear_index * 12 + 4
    ; Store workgroup_id_z at address linear_index * 12 + 8
    mov_imm r6, 12
    imul r7, r5, r6          ; r7 = linear_index * 12

    device_store_u32 r7, r0  ; store workgroup_id_x
    mov_imm r6, 4
    iadd r7, r7, r6
    device_store_u32 r7, r1  ; store workgroup_id_y
    iadd r7, r7, r6
    device_store_u32 r7, r2  ; store workgroup_id_z

    halt
.end
"#;

    let config = EmulatorConfig {
        grid_dim: [2, 2, 1],  // 4 workgroups
        workgroup_dim: [1, 1, 1],
        wave_width: 1,
        device_memory_size: 1024,
        ..Default::default()
    };

    let emulator = assemble_and_run(source, config);

    for wg_y in 0u32..2 {
        for wg_x in 0u32..2 {
            let linear_index = wg_y * 2 + wg_x;
            let base_addr = u64::from(linear_index * 12);
            let stored_x = emulator.device_memory().read_u32(base_addr).unwrap();
            let stored_y = emulator.device_memory().read_u32(base_addr + 4).unwrap();
            let stored_z = emulator.device_memory().read_u32(base_addr + 8).unwrap();
            assert_eq!(stored_x, wg_x, "Workgroup ({wg_x},{wg_y}): expected workgroup_id_x={wg_x}, got {stored_x}");
            assert_eq!(stored_y, wg_y, "Workgroup ({wg_x},{wg_y}): expected workgroup_id_y={wg_y}, got {stored_y}");
            assert_eq!(stored_z, 0, "Workgroup ({wg_x},{wg_y}): expected workgroup_id_z=0, got {stored_z}");
        }
    }
}

#[test]
fn test_roundtrip_all_special_registers() {
    let source = r#"
.kernel test_all_sr
.registers 32
.workgroup_size 1, 1, 1
    ; Read all special registers
    mov r0, sr_thread_id_x
    mov r1, sr_thread_id_y
    mov r2, sr_thread_id_z
    mov r3, sr_wave_id
    mov r4, sr_lane_id
    mov r5, sr_workgroup_id_x
    mov r6, sr_workgroup_id_y
    mov r7, sr_workgroup_id_z
    mov r8, sr_workgroup_size_x
    mov r9, sr_workgroup_size_y
    mov r10, sr_workgroup_size_z
    mov r11, sr_grid_size_x
    mov r12, sr_grid_size_y
    mov r13, sr_grid_size_z
    mov r14, sr_wave_width
    mov r15, sr_num_waves

    ; Store all values to memory
    mov_imm r20, 0
    device_store_u32 r20, r0
    mov_imm r20, 4
    device_store_u32 r20, r1
    mov_imm r20, 8
    device_store_u32 r20, r2
    mov_imm r20, 12
    device_store_u32 r20, r3
    mov_imm r20, 16
    device_store_u32 r20, r4
    mov_imm r20, 20
    device_store_u32 r20, r5
    mov_imm r20, 24
    device_store_u32 r20, r6
    mov_imm r20, 28
    device_store_u32 r20, r7
    mov_imm r20, 32
    device_store_u32 r20, r8
    mov_imm r20, 36
    device_store_u32 r20, r9
    mov_imm r20, 40
    device_store_u32 r20, r10
    mov_imm r20, 44
    device_store_u32 r20, r11
    mov_imm r20, 48
    device_store_u32 r20, r12
    mov_imm r20, 52
    device_store_u32 r20, r13
    mov_imm r20, 56
    device_store_u32 r20, r14
    mov_imm r20, 60
    device_store_u32 r20, r15

    halt
.end
"#;

    let config = EmulatorConfig {
        grid_dim: [2, 3, 1],
        workgroup_dim: [1, 1, 1],
        wave_width: 1,
        device_memory_size: 1024,
        register_count: 32,
        ..Default::default()
    };

    let emulator = assemble_and_run(source, config);
    let mem = emulator.device_memory();

    assert_eq!(mem.read_u32(0).unwrap(), 0);
    assert_eq!(mem.read_u32(4).unwrap(), 0);
    assert_eq!(mem.read_u32(8).unwrap(), 0);
    assert_eq!(mem.read_u32(12).unwrap(), 0);
    assert_eq!(mem.read_u32(16).unwrap(), 0);
    assert_eq!(mem.read_u32(20).unwrap(), 1);
    assert_eq!(mem.read_u32(24).unwrap(), 2);
    assert_eq!(mem.read_u32(28).unwrap(), 0);
    assert_eq!(mem.read_u32(32).unwrap(), 1);
    assert_eq!(mem.read_u32(36).unwrap(), 1);
    assert_eq!(mem.read_u32(40).unwrap(), 1);
    assert_eq!(mem.read_u32(44).unwrap(), 2);
    assert_eq!(mem.read_u32(48).unwrap(), 3);
    assert_eq!(mem.read_u32(52).unwrap(), 1);
    assert_eq!(mem.read_u32(56).unwrap(), 1);
    assert_eq!(mem.read_u32(60).unwrap(), 1);
}
