// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! GPU detection for the WAVE runtime.
//!
//! Probes the system for available GPU hardware and returns a `Device` with
//! vendor and name information. Falls back to the WAVE emulator when no
//! supported GPU is found.

use crate::error::RuntimeError;
use std::fmt;
use std::process::Command;

/// GPU vendor classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    /// Apple GPU via Metal.
    Apple,
    /// NVIDIA GPU via CUDA/PTX.
    Nvidia,
    /// AMD GPU via `ROCm`/HIP.
    Amd,
    /// Intel GPU via oneAPI/SYCL.
    Intel,
    /// Software emulator fallback.
    Emulator,
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Apple => write!(f, "Metal"),
            Self::Nvidia => write!(f, "CUDA"),
            Self::Amd => write!(f, "ROCm"),
            Self::Intel => write!(f, "SYCL"),
            Self::Emulator => write!(f, "Emulator"),
        }
    }
}

/// Detected GPU device.
#[derive(Debug, Clone)]
pub struct Device {
    /// Hardware vendor.
    pub vendor: GpuVendor,
    /// Human-readable device name.
    pub name: String,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.vendor)
    }
}

/// Detect the best available GPU on the system.
///
/// Checks in order: Metal (macOS), CUDA (nvidia-smi), `ROCm` (rocminfo),
/// oneAPI (sycl-ls). Falls back to the WAVE emulator if no GPU is found.
///
/// # Errors
///
/// Returns `RuntimeError::Device` if detection encounters an unrecoverable error.
pub fn detect_gpu() -> Result<Device, RuntimeError> {
    if cfg!(target_os = "macos") {
        return Ok(Device {
            vendor: GpuVendor::Apple,
            name: "Apple GPU (Metal)".into(),
        });
    }

    if let Some(dev) = detect_nvidia() {
        return Ok(dev);
    }

    if let Some(dev) = detect_amd() {
        return Ok(dev);
    }

    if let Some(dev) = detect_intel() {
        return Ok(dev);
    }

    Ok(Device {
        vendor: GpuVendor::Emulator,
        name: "WAVE Emulator (no GPU)".into(),
    })
}

fn detect_nvidia() -> Option<Device> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let name = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("NVIDIA GPU")
        .trim()
        .to_string();

    Some(Device {
        vendor: GpuVendor::Nvidia,
        name: format!("{name} (CUDA)"),
    })
}

fn detect_amd() -> Option<Device> {
    let output = Command::new("rocminfo").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("gfx") {
        return None;
    }

    let name = stdout
        .lines()
        .find(|l| l.contains("Marketing Name"))
        .and_then(|l| l.split(':').nth(1))
        .map_or_else(|| "AMD GPU".into(), |s| s.trim().to_string());

    Some(Device {
        vendor: GpuVendor::Amd,
        name: format!("{name} (ROCm)"),
    })
}

fn detect_intel() -> Option<Device> {
    let output = Command::new("sycl-ls").output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("level_zero:gpu") && !stdout.contains("opencl:gpu") {
        return None;
    }

    Some(Device {
        vendor: GpuVendor::Intel,
        name: "Intel GPU (SYCL)".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gpu_returns_device() {
        let device = detect_gpu().unwrap();
        assert!(!device.name.is_empty());
    }

    #[test]
    fn test_device_display() {
        let device = Device {
            vendor: GpuVendor::Emulator,
            name: "WAVE Emulator (no GPU)".into(),
        };
        assert_eq!(device.to_string(), "WAVE Emulator (no GPU) (Emulator)");
    }
}
