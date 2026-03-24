// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the WAVE HIP backend. Reads a WBIN binary file, translates
//!
//! it to HIP C++ using the wave_hip::compile API, and writes the resulting source
//! to an output file or stdout.

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "wave-hip")]
#[command(about = "WAVE HIP backend - translates WAVE binary (.wbin) to AMD HIP kernel source (.hip)")]
#[command(version)]
struct Cli {
    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    let wbin_data = match fs::read(&cli.input) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("error: cannot read '{}': {e}", cli.input.display());
            process::exit(1);
        }
    };

    let hip = match wave_hip::compile(&wbin_data) {
        Ok(hip) => hip,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    match cli.output {
        Some(path) => {
            if let Err(e) = fs::write(&path, &hip) {
                eprintln!("error: cannot write '{}': {e}", path.display());
                process::exit(1);
            }
        }
        None => {
            print!("{hip}");
        }
    }
}
