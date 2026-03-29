// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE SDK for Rust: write GPU kernels in Rust, run on any GPU.
//!
//! Thin wrapper around the `wave-runtime` crate, providing a convenient API
//! for kernel compilation, device detection, array management, and kernel launch.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate
)]

pub mod array {
    //! Array types for kernel data.

    pub use wave_runtime::memory::{DeviceBuffer, ElementType};

    /// Create a `DeviceBuffer` from an `f32` slice.
    pub fn from_f32(data: &[f32]) -> DeviceBuffer {
        DeviceBuffer::from_f32(data)
    }

    /// Create a zero-filled `f32` buffer.
    pub fn zeros_f32(count: usize) -> DeviceBuffer {
        DeviceBuffer::zeros_f32(count)
    }

    /// Create a `DeviceBuffer` from a `u32` slice.
    pub fn from_u32(data: &[u32]) -> DeviceBuffer {
        DeviceBuffer::from_u32(data)
    }

    /// Create a zero-filled `u32` buffer.
    pub fn zeros_u32(count: usize) -> DeviceBuffer {
        DeviceBuffer::zeros_u32(count)
    }
}

pub mod device {
    //! GPU detection.

    pub use wave_runtime::device::{detect_gpu as detect, Device, GpuVendor};
}

pub mod kernel {
    //! Kernel compilation and launch.

    use wave_runtime::backend::translate_to_vendor;
    use wave_runtime::device::{Device, GpuVendor};
    use wave_runtime::error::RuntimeError;
    use wave_runtime::launcher::launch_kernel;
    use wave_runtime::memory::DeviceBuffer;

    pub use wave_compiler::Language;

    /// A compiled WAVE kernel ready for launch.
    pub struct CompiledKernel {
        wbin: Vec<u8>,
        vendor_code: Option<String>,
    }

    /// Compile kernel source to a `CompiledKernel`.
    pub fn compile(source: &str, language: Language) -> Result<CompiledKernel, RuntimeError> {
        let wbin = wave_runtime::compiler::compile_kernel(source, language)?;
        Ok(CompiledKernel {
            wbin,
            vendor_code: None,
        })
    }

    impl CompiledKernel {
        /// Launch this kernel on the given device.
        pub fn launch(
            &mut self,
            device: &Device,
            buffers: &mut [&mut DeviceBuffer],
            scalars: &[u32],
            grid: [u32; 3],
            workgroup: [u32; 3],
        ) -> Result<(), RuntimeError> {
            let vendor_code = if device.vendor == GpuVendor::Emulator {
                String::new()
            } else if let Some(code) = &self.vendor_code {
                code.clone()
            } else {
                let code = translate_to_vendor(&self.wbin, device.vendor)?;
                self.vendor_code = Some(code.clone());
                code
            };

            launch_kernel(
                &vendor_code,
                &self.wbin,
                device.vendor,
                buffers,
                scalars,
                grid,
                workgroup,
            )
        }
    }
}

pub use wave_runtime::error::RuntimeError;
pub use wave_runtime::Language;
