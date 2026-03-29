// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the WAVE PTX backend. Reads a WBIN binary file, translates
//!
//! it to PTX assembly using the wave_ptx::compile API, and writes the resulting
//! PTX text to an output file or stdout. Supports --sm flag for target SM version.

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "wave-ptx")]
#[command(
    about = "WAVE PTX backend - translates WAVE binary (.wbin) to NVIDIA PTX assembly (.ptx)"
)]
#[command(version)]
struct Cli {
    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long, default_value = "75")]
    sm: u32,
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

    let ptx = match wave_ptx::compile(&wbin_data, cli.sm) {
        Ok(ptx) => ptx,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    match cli.output {
        Some(path) => {
            if let Err(e) = fs::write(&path, &ptx) {
                eprintln!("error: cannot write '{}': {e}", path.display());
                process::exit(1);
            }
        }
        None => {
            print!("{ptx}");
        }
    }
}
