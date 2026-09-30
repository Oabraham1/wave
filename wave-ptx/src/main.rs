// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Translate WBIN files into PTX from the command line.

use clap::Parser;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Debug, Parser)]
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let wbin_data = fs::read(&cli.input)
        .map_err(|error| format!("cannot read '{}': {error}", cli.input.display()))?;
    let ptx = wave_ptx::compile(&wbin_data, cli.sm)?;
    if let Some(path) = cli.output {
        fs::write(&path, &ptx)
            .map_err(|error| format!("cannot write '{}': {error}", path.display()))?;
    } else {
        io::stdout().lock().write_all(ptx.as_bytes())?;
    }
    Ok(())
}
