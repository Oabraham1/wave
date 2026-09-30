// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate WBIN files into HIP C++ from the command line.

use clap::Parser;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "wave-hip")]
#[command(
    about = "WAVE HIP backend - translates WAVE binary (.wbin) to AMD HIP kernel source (.hip)"
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
    let hip = wave_hip::compile(&wbin_data)?;
    if let Some(path) = cli.output {
        fs::write(&path, &hip)
            .map_err(|error| format!("cannot write '{}': {error}", path.display()))?;
    } else {
        io::stdout().lock().write_all(hip.as_bytes())?;
    }
    Ok(())
}
