// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the WAVE SYCL backend. Reads a WBIN binary file, translates
//!
//! it to SYCL C++ using the wave_sycl::compile API, and writes the resulting source
//! to an output file or stdout.

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "wave-sycl")]
#[command(about = "WAVE SYCL backend - translates WAVE binary (.wbin) to Intel SYCL kernel source (.cpp)")]
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

    let sycl = match wave_sycl::compile(&wbin_data) {
        Ok(sycl) => sycl,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    match cli.output {
        Some(path) => {
            if let Err(e) = fs::write(&path, &sycl) {
                eprintln!("error: cannot write '{}': {e}", path.display());
                process::exit(1);
            }
        }
        None => {
            print!("{sycl}");
        }
    }
}
