// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the WAVE emulator. Parses command-line arguments for grid
//!
//! and workgroup dimensions, memory sizes, input data, and output options. Loads
//! the WBIN binary, executes it, and reports results or statistics.

#![allow(clippy::cast_possible_truncation)]

use clap::Parser;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use wave_emu::{load_binary_file, Emulator, EmulatorConfig, EmulatorError};

#[derive(Parser, Debug)]
#[command(name = "wave-emu")]
#[command(about = "WAVE ISA emulator - executes .wbin binaries")]
#[command(version)]
struct Args {
    #[arg(help = "WBIN binary file to execute")]
    binary: PathBuf,

    #[arg(
        long,
        value_name = "X,Y,Z",
        default_value = "1,1,1",
        help = "Grid dimensions"
    )]
    grid: String,

    #[arg(
        long,
        value_name = "X,Y,Z",
        default_value = "32,1,1",
        help = "Workgroup dimensions"
    )]
    workgroup: String,

    #[arg(long, default_value = "32", help = "Registers per thread")]
    registers: u32,

    #[arg(
        long,
        value_name = "BYTES",
        default_value = "16384",
        help = "Local memory size"
    )]
    local_memory: usize,

    #[arg(
        long,
        value_name = "BYTES",
        default_value = "1048576",
        help = "Device memory size"
    )]
    device_memory: usize,

    #[arg(
        long,
        value_name = "OFFSET:FILE",
        help = "Load file into device memory at offset"
    )]
    arg: Vec<String>,

    #[arg(long, help = "Print register state after execution")]
    dump_regs: bool,

    #[arg(
        long,
        value_name = "START:END",
        help = "Hex dump device memory range after execution"
    )]
    dump_memory: Option<String>,

    #[arg(long, help = "Print execution statistics")]
    stats: bool,

    #[arg(long, help = "Print each instruction as it executes")]
    trace: bool,

    #[arg(long, default_value = "32", help = "Wave width")]
    wave_width: u32,

    #[arg(
        long,
        default_value = "10000000",
        help = "Maximum instructions to execute (0 = unlimited)"
    )]
    max_instructions: u64,

    #[arg(
        long,
        value_name = "REG:VALUE",
        help = "Set initial register value for all threads (e.g. 0:4096)"
    )]
    set_reg: Vec<String>,

    #[arg(
        long,
        value_name = "OFFSET:TYPE:COUNT:SCALE",
        help = "Fill memory with iota pattern (0, scale, 2*scale, ...)"
    )]
    fill_iota: Vec<String>,

    #[arg(
        long,
        value_name = "OFFSET:TYPE:COUNT",
        help = "Fill memory region with zeros"
    )]
    fill_zero: Vec<String>,

    #[arg(
        long,
        value_name = "OFFSET:COUNT",
        help = "Dump device memory as f32 values, one per line"
    )]
    dump_f32: Option<String>,
}

fn parse_dimensions(s: &str) -> Result<[u32; 3], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected 3 comma-separated values, got {}",
            parts.len()
        ));
    }

    let x = parts[0]
        .parse::<u32>()
        .map_err(|e| format!("invalid x: {e}"))?;
    let y = parts[1]
        .parse::<u32>()
        .map_err(|e| format!("invalid y: {e}"))?;
    let z = parts[2]
        .parse::<u32>()
        .map_err(|e| format!("invalid z: {e}"))?;

    Ok([x, y, z])
}

fn parse_offset(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| format!("invalid hex offset: {e}"))
    } else {
        s.parse::<u64>().map_err(|e| format!("invalid offset: {e}"))
    }
}

fn parse_arg_spec(spec: &str) -> Result<(u64, PathBuf), String> {
    let parts: Vec<&str> = spec.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err("expected format OFFSET:FILE".into());
    }

    let offset = parse_offset(parts[0])?;
    let path = PathBuf::from(parts[1]);

    Ok((offset, path))
}

fn parse_memory_range(s: &str) -> Result<(u64, u64), String> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err("expected format START:END".into());
    }

    let start = parse_offset(parts[0])?;
    let end = parse_offset(parts[1])?;

    if end <= start {
        return Err("END must be greater than START".into());
    }

    Ok((start, end))
}

fn parse_set_reg(spec: &str) -> Result<(u8, u32), String> {
    let parts: Vec<&str> = spec.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err("expected format REG:VALUE".into());
    }
    let reg = parts[0]
        .parse::<u8>()
        .map_err(|e| format!("invalid register: {e}"))?;
    let value = parse_offset(parts[1])? as u32;
    Ok((reg, value))
}

#[allow(clippy::cast_precision_loss)]
fn generate_iota_f32(count: usize, scale: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity(count * 4);
    for i in 0..count {
        let val = i as f32 * scale;
        data.extend_from_slice(&val.to_le_bytes());
    }
    data
}

fn hex_dump(data: &[u8], base_addr: u64) {
    for (i, chunk) in data.chunks(16).enumerate() {
        let addr = base_addr + (i as u64 * 16);
        print!("{addr:08x}: ");

        for (j, byte) in chunk.iter().enumerate() {
            if j == 8 {
                print!(" ");
            }
            print!("{byte:02x} ");
        }

        for _ in chunk.len()..16 {
            print!("   ");
        }
        if chunk.len() <= 8 {
            print!(" ");
        }

        print!(" |");
        for byte in chunk {
            let c = if byte.is_ascii_graphic() || *byte == b' ' {
                *byte as char
            } else {
                '.'
            };
            print!("{c}");
        }
        println!("|");
    }
}

fn build_config(args: &Args) -> Result<EmulatorConfig, EmulatorError> {
    let grid_dim = parse_dimensions(&args.grid).map_err(|e| EmulatorError::InvalidBinary {
        message: format!("invalid --grid: {e}"),
    })?;

    let workgroup_dim =
        parse_dimensions(&args.workgroup).map_err(|e| EmulatorError::InvalidBinary {
            message: format!("invalid --workgroup: {e}"),
        })?;

    let mut initial_registers = Vec::new();
    for spec in &args.set_reg {
        let (reg, value) = parse_set_reg(spec).map_err(|e| EmulatorError::InvalidBinary {
            message: format!("invalid --set-reg: {e}"),
        })?;
        initial_registers.push((reg, value));
    }

    Ok(EmulatorConfig {
        grid_dim,
        workgroup_dim,
        register_count: args.registers,
        local_memory_size: args.local_memory,
        device_memory_size: args.device_memory,
        wave_width: args.wave_width,
        trace_enabled: args.trace,
        f64_enabled: false,
        max_instructions: args.max_instructions,
        initial_registers,
    })
}

fn load_memory_fills(emulator: &mut Emulator, args: &Args) -> Result<(), EmulatorError> {
    for spec in &args.fill_zero {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 3 {
            return Err(EmulatorError::InvalidBinary {
                message: format!("--fill-zero expects OFFSET:TYPE:COUNT, got '{spec}'"),
            });
        }
        let offset = parse_offset(parts[0]).map_err(|e| EmulatorError::InvalidBinary {
            message: format!("--fill-zero offset: {e}"),
        })?;
        let count = parts[2]
            .parse::<usize>()
            .map_err(|e| EmulatorError::InvalidBinary {
                message: format!("--fill-zero count: {e}"),
            })?;
        let data = vec![0u8; count * 4];
        emulator.load_device_memory(offset, &data)?;
    }

    for spec in &args.fill_iota {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() < 3 || parts.len() > 4 {
            return Err(EmulatorError::InvalidBinary {
                message: format!("--fill-iota expects OFFSET:TYPE:COUNT[:SCALE], got '{spec}'"),
            });
        }
        let offset = parse_offset(parts[0]).map_err(|e| EmulatorError::InvalidBinary {
            message: format!("--fill-iota offset: {e}"),
        })?;
        let count = parts[2]
            .parse::<usize>()
            .map_err(|e| EmulatorError::InvalidBinary {
                message: format!("--fill-iota count: {e}"),
            })?;
        let scale: f32 = if parts.len() == 4 {
            parts[3]
                .parse::<f32>()
                .map_err(|e| EmulatorError::InvalidBinary {
                    message: format!("--fill-iota scale: {e}"),
                })?
        } else {
            1.0
        };
        let data = generate_iota_f32(count, scale);
        emulator.load_device_memory(offset, &data)?;
    }

    for arg_spec in &args.arg {
        let (offset, path) =
            parse_arg_spec(arg_spec).map_err(|e| EmulatorError::InvalidBinary {
                message: format!("invalid --arg: {e}"),
            })?;

        let mut file = File::open(&path).map_err(|e| EmulatorError::IoError {
            message: format!("failed to open {}: {}", path.display(), e),
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| EmulatorError::IoError {
                message: format!("failed to read {}: {}", path.display(), e),
            })?;

        emulator.load_device_memory(offset, &data)?;
    }

    Ok(())
}

fn dump_outputs(emulator: &Emulator, args: &Args) -> Result<(), EmulatorError> {
    if let Some(range_str) = &args.dump_memory {
        let (start, end) =
            parse_memory_range(range_str).map_err(|e| EmulatorError::InvalidBinary {
                message: format!("invalid --dump-memory: {e}"),
            })?;

        let len = (end - start) as usize;
        let data = emulator.read_device_memory(start, len)?;

        println!("Device memory 0x{start:08x}-0x{end:08x}:");
        hex_dump(&data, start);
    }

    if let Some(spec) = &args.dump_f32 {
        let parts: Vec<&str> = spec.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(EmulatorError::InvalidBinary {
                message: format!("--dump-f32 expects OFFSET:COUNT, got '{spec}'"),
            });
        }
        let offset = parse_offset(parts[0]).map_err(|e| EmulatorError::InvalidBinary {
            message: format!("--dump-f32 offset: {e}"),
        })?;
        let count = parts[1]
            .parse::<usize>()
            .map_err(|e| EmulatorError::InvalidBinary {
                message: format!("--dump-f32 count: {e}"),
            })?;
        let data = emulator.read_device_memory(offset, count * 4)?;
        for i in 0..count {
            let bytes = [
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ];
            let val = f32::from_le_bytes(bytes);
            println!("{val:?}");
        }
    }

    Ok(())
}

fn run() -> Result<(), EmulatorError> {
    let args = Args::parse();
    let config = build_config(&args)?;

    let binary = load_binary_file(&args.binary)?;
    let mut emulator = Emulator::new(config);
    emulator.load_binary(&binary)?;

    load_memory_fills(&mut emulator, &args)?;

    let result = emulator.run()?;

    if args.dump_regs {
        eprintln!("Register dumps not implemented for multi-wave execution");
    }

    dump_outputs(&emulator, &args)?;

    if args.stats {
        print!("{}", result.stats);
    }

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
