// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate WBIN files into SYCL C++ from the command line.

use clap::Parser;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "wave-sycl")]
#[command(
    about = "WAVE SYCL backend - translates WAVE binary (.wbin) to Intel SYCL kernel source (.cpp)"
)]
#[command(version)]
struct Cli {
    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let wbin_data = fs::read(&cli.input)
        .map_err(|error| format!("cannot read '{}': {error}", cli.input.display()))?;
    let sycl = wave_sycl::compile(&wbin_data)?;
    if let Some(path) = cli.output {
        fs::write(&path, &sycl)
            .map_err(|error| format!("cannot write '{}': {error}", path.display()))?;
    } else {
        io::stdout().lock().write_all(sycl.as_bytes())?;
    }
    Ok(())
}
