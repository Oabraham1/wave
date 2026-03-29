// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Backend translation wrapper for the WAVE runtime.
//!
//! Translates compiled WAVE binary (.wbin) data into vendor-specific GPU
//! source code using the appropriate backend crate (Metal, PTX, HIP, or SYCL).

use crate::device::GpuVendor;
use crate::error::RuntimeError;

/// Translate a WAVE binary to vendor-specific source code.
///
/// # Errors
///
/// Returns `RuntimeError::Backend` if the translation fails or the vendor
/// is the emulator (which does not need translation).
pub fn translate_to_vendor(wbin: &[u8], vendor: GpuVendor) -> Result<String, RuntimeError> {
    match vendor {
        GpuVendor::Apple => {
            wave_metal::compile(wbin).map_err(|e| RuntimeError::Backend(e.to_string()))
        }
        GpuVendor::Nvidia => {
            wave_ptx::compile(wbin, 75).map_err(|e| RuntimeError::Backend(e.to_string()))
        }
        GpuVendor::Amd => wave_hip::compile(wbin).map_err(|e| RuntimeError::Backend(e.to_string())),
        GpuVendor::Intel => {
            wave_sycl::compile(wbin).map_err(|e| RuntimeError::Backend(e.to_string()))
        }
        GpuVendor::Emulator => Err(RuntimeError::Backend(
            "emulator does not require backend translation".into(),
        )),
    }
}
