// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the WAVE Metal backend. Reads a WBIN binary file, translates
//!
//! it to Metal Shading Language using the wave_metal::compile API, and writes the
//! resulting MSL source to an output file or stdout.

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(name = "wave-metal")]
#[command(
    about = "WAVE Metal backend - translates WAVE binary (.wbin) to Metal Shading Language (.metal)"
)]
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

    let msl = match wave_metal::compile(&wbin_data) {
        Ok(msl) => msl,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    match cli.output {
        Some(path) => {
            if let Err(e) = fs::write(&path, &msl) {
                eprintln!("error: cannot write '{}': {e}", path.display());
                process::exit(1);
            }
        }
        None => {
            print!("{msl}");
        }
    }
}
