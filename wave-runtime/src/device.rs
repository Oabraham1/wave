// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! GPU detection and device enumeration for the WAVE runtime.
//!
//! Probes the system for available GPU hardware. The `detect_gpu` function
//! returns the single best device for backwards compatibility. The
//! `enumerate_devices` function returns all available GPUs with detailed
//! capability information for multi-GPU workloads.

use crate::error::RuntimeError;
use std::fmt;
use std::process::Command;

/// GPU vendor classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Hardware capability flags for a GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapabilities {
    /// Supports 16-bit float (half precision).
    pub f16: bool,
    /// Supports bfloat16.
    pub bf16: bool,
    /// Supports matrix multiply-accumulate (tensor cores / matrix units).
    pub mma: bool,
}

/// Detailed GPU device information for multi-device workloads.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Zero-based device index (unique across all vendors).
    pub id: usize,
    /// Hardware vendor.
    pub vendor: GpuVendor,
    /// Human-readable model name.
    pub name: String,
    /// Device memory in bytes (0 if unknown).
    pub memory_bytes: u64,
    /// SIMD width (wave/warp size).
    pub wave_width: u32,
    /// Maximum general-purpose registers per thread.
    pub max_registers: u32,
    /// Scratchpad (shared/threadgroup) memory in bytes.
    pub scratchpad_bytes: u32,
    /// Hardware capability flags.
    pub capabilities: DeviceCapabilities,
}

impl fmt::Display for DeviceInfo {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({}, {:.0} MB)",
            self.id,
            self.name,
            self.vendor,
            self.memory_bytes as f64 / (1024.0 * 1024.0)
        )
    }
}

/// Detected GPU device (legacy single-device API).
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

/// Enumerate all available GPU devices on the system.
///
/// Returns a list of all detected GPUs with detailed capability information.
/// On macOS, queries Metal for device properties. On Linux, enumerates NVIDIA
/// GPUs via nvidia-smi, AMD GPUs via rocminfo, and Intel GPUs via sycl-ls.
/// Falls back to the WAVE emulator if no GPUs are found.
///
/// # Errors
///
/// Returns `RuntimeError::Device` if enumeration encounters an unrecoverable error.
pub fn enumerate_devices() -> Result<Vec<DeviceInfo>, RuntimeError> {
    let mut devices = Vec::new();
    let mut next_id: usize = 0;

    if cfg!(target_os = "macos") {
        let apple_devices = enumerate_apple(&mut next_id);
        devices.extend(apple_devices);
    } else {
        devices.extend(enumerate_nvidia(&mut next_id));
        devices.extend(enumerate_amd(&mut next_id));
        devices.extend(enumerate_intel(&mut next_id));
    }

    if devices.is_empty() {
        devices.push(DeviceInfo {
            id: 0,
            vendor: GpuVendor::Emulator,
            name: "WAVE Emulator (no GPU)".into(),
            memory_bytes: 0,
            wave_width: 32,
            max_registers: 256,
            scratchpad_bytes: 32768,
            capabilities: DeviceCapabilities {
                f16: true,
                bf16: true,
                mma: true,
            },
        });
    }

    Ok(devices)
}

fn enumerate_apple(next_id: &mut usize) -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    let (name, memory_bytes) = query_apple_device_properties();

    let id = *next_id;
    *next_id += 1;
    devices.push(DeviceInfo {
        id,
        vendor: GpuVendor::Apple,
        name,
        memory_bytes,
        wave_width: 32,
        max_registers: 32,
        scratchpad_bytes: 32768,
        capabilities: DeviceCapabilities {
            f16: true,
            bf16: false,
            mma: false,
        },
    });

    devices
}

fn query_apple_device_properties() -> (String, u64) {
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output();

    let Ok(output) = output else {
        return ("Apple GPU".into(), 0);
    };

    if !output.status.success() {
        return ("Apple GPU".into(), 0);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    let name = stdout
        .lines()
        .find(|l| l.contains("Chipset Model:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or_else(|| "Apple GPU".into(), |s| s.trim().to_string());

    let memory_bytes = stdout
        .lines()
        .find(|l| l.contains("VRAM") || l.contains("Memory"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| {
            let s = s.trim();
            let numeric: String = s.chars().take_while(char::is_ascii_digit).collect();
            numeric.parse::<u64>().ok()
        })
        .map_or(0, |gb| gb * 1024 * 1024 * 1024);

    (name, memory_bytes)
}

fn enumerate_nvidia(next_id: &mut usize) -> Vec<DeviceInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    let Ok(output) = output else {
        return Vec::new();
    };

    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(2, ',').collect();
        let name = parts.first().map_or("NVIDIA GPU", |s| s.trim()).to_string();
        let memory_mb: u64 = parts
            .get(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let id = *next_id;
        *next_id += 1;
        devices.push(DeviceInfo {
            id,
            vendor: GpuVendor::Nvidia,
            name,
            memory_bytes: memory_mb * 1024 * 1024,
            wave_width: 32,
            max_registers: 255,
            scratchpad_bytes: 49152,
            capabilities: DeviceCapabilities {
                f16: true,
                bf16: true,
                mma: true,
            },
        });
    }

    devices
}

fn enumerate_amd(next_id: &mut usize) -> Vec<DeviceInfo> {
    let output = Command::new("rocminfo").output();

    let Ok(output) = output else {
        return Vec::new();
    };

    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("gfx") {
        return Vec::new();
    }

    let mut devices = Vec::new();
    let mut current_name = String::new();
    let mut in_gpu_agent = false;

    for line in stdout.lines() {
        if line.contains("Agent ") && line.contains("GPU") {
            in_gpu_agent = true;
            current_name = "AMD GPU".into();
        } else if line.contains("Agent ") && !line.contains("GPU") {
            if in_gpu_agent && !current_name.is_empty() {
                let id = *next_id;
                *next_id += 1;
                devices.push(DeviceInfo {
                    id,
                    vendor: GpuVendor::Amd,
                    name: current_name.clone(),
                    memory_bytes: 0,
                    wave_width: 64,
                    max_registers: 256,
                    scratchpad_bytes: 65536,
                    capabilities: DeviceCapabilities {
                        f16: true,
                        bf16: true,
                        mma: true,
                    },
                });
            }
            in_gpu_agent = false;
        } else if in_gpu_agent && line.contains("Marketing Name") {
            if let Some(name) = line.split(':').nth(1) {
                current_name = name.trim().to_string();
            }
        }
    }

    if in_gpu_agent && !current_name.is_empty() {
        let id = *next_id;
        *next_id += 1;
        devices.push(DeviceInfo {
            id,
            vendor: GpuVendor::Amd,
            name: current_name,
            memory_bytes: 0,
            wave_width: 64,
            max_registers: 256,
            scratchpad_bytes: 65536,
            capabilities: DeviceCapabilities {
                f16: true,
                bf16: true,
                mma: true,
            },
        });
    }

    devices
}

fn enumerate_intel(next_id: &mut usize) -> Vec<DeviceInfo> {
    let output = Command::new("sycl-ls").output();

    let Ok(output) = output else {
        return Vec::new();
    };

    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut devices = Vec::new();

    for line in stdout.lines() {
        if line.contains("level_zero:gpu") || line.contains("opencl:gpu") {
            let id = *next_id;
            *next_id += 1;
            devices.push(DeviceInfo {
                id,
                vendor: GpuVendor::Intel,
                name: "Intel GPU".into(),
                memory_bytes: 0,
                wave_width: 16,
                max_registers: 128,
                scratchpad_bytes: 65536,
                capabilities: DeviceCapabilities {
                    f16: true,
                    bf16: false,
                    mma: false,
                },
            });
        }
    }

    devices
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
    let devices = enumerate_devices()?;
    let best = &devices[0];
    Ok(Device {
        vendor: best.vendor,
        name: format!("{} ({})", best.name, best.vendor),
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

    #[test]
    fn test_enumerate_devices_returns_at_least_one() {
        let devices = enumerate_devices().unwrap();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].id, 0);
    }

    #[test]
    fn test_device_info_display() {
        let info = DeviceInfo {
            id: 0,
            vendor: GpuVendor::Apple,
            name: "Apple M4 Pro".into(),
            memory_bytes: 18 * 1024 * 1024 * 1024,
            wave_width: 32,
            max_registers: 32,
            scratchpad_bytes: 32768,
            capabilities: DeviceCapabilities {
                f16: true,
                bf16: false,
                mma: false,
            },
        };
        let display = info.to_string();
        assert!(display.contains("Apple M4 Pro"));
        assert!(display.contains("Metal"));
    }

    #[test]
    fn test_device_capabilities() {
        let caps = DeviceCapabilities {
            f16: true,
            bf16: false,
            mma: true,
        };
        assert!(caps.f16);
        assert!(!caps.bf16);
        assert!(caps.mma);
    }
}
