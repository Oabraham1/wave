// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE SDK for Rust: write GPU kernels in Rust, run on any GPU. Thin wrapper around the `wave-
//! runtime` crate, providing a convenient API for kernel compilation, device detection, array
//! management, and kernel launch.

pub mod array {
    //! Array types for kernel data.

    pub use wave_runtime::memory::{DeviceBuffer, ElementType};

    /// Create a `DeviceBuffer` from an `f32` slice.
    #[must_use]
    pub fn from_f32(data: &[f32]) -> DeviceBuffer {
        DeviceBuffer::from_f32(data)
    }

    /// Create a zero-filled `f32` buffer.
    #[must_use]
    pub fn zeros_f32(count: usize) -> DeviceBuffer {
        DeviceBuffer::zeros_f32(count)
    }

    /// Create a `DeviceBuffer` from a `u32` slice.
    #[must_use]
    pub fn from_u32(data: &[u32]) -> DeviceBuffer {
        DeviceBuffer::from_u32(data)
    }

    /// Create a zero-filled `u32` buffer.
    #[must_use]
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

    use std::collections::{hash_map::Entry, HashMap};

    use wave_runtime::backend::translate_to_vendor;
    use wave_runtime::device::{Device, GpuVendor};
    use wave_runtime::error::RuntimeError;
    use wave_runtime::launcher::launch_kernel;
    use wave_runtime::memory::DeviceBuffer;

    pub use wave_compiler::Language;

    #[derive(Debug)]
    /// A compiled WAVE kernel ready for launch.
    pub struct CompiledKernel {
        wbin: Vec<u8>,
        vendor_code: HashMap<GpuVendor, String>,
    }

    /// Compile kernel source to a `CompiledKernel`.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing, validation, or code generation fails.
    pub fn compile(source: &str, language: Language) -> Result<CompiledKernel, RuntimeError> {
        let wbin = wave_runtime::compiler::compile_kernel(source, language)?;
        Ok(CompiledKernel {
            wbin,
            vendor_code: HashMap::new(),
        })
    }

    impl CompiledKernel {
        /// Launch this kernel on the given device.
        ///
        /// # Errors
        ///
        /// Returns an error if translation, buffer preparation, compilation, or execution fails.
        pub fn launch(
            &mut self,
            device: &Device,
            buffers: &mut [&mut DeviceBuffer],
            scalars: &[u32],
            grid: [u32; 3],
            workgroup: [u32; 3],
        ) -> Result<(), RuntimeError> {
            self.prepare_vendor(device.vendor)?;
            let vendor_code = self
                .vendor_code
                .get(&device.vendor)
                .map_or("", String::as_str);

            launch_kernel(
                vendor_code,
                &self.wbin,
                device.vendor,
                buffers,
                scalars,
                grid,
                workgroup,
            )
        }

        fn prepare_vendor(&mut self, vendor: GpuVendor) -> Result<(), RuntimeError> {
            if vendor != GpuVendor::Emulator {
                if let Entry::Vacant(entry) = self.vendor_code.entry(vendor) {
                    entry.insert(translate_to_vendor(&self.wbin, vendor)?);
                }
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn compiled_kernel_caches_each_vendor() {
            let mut kernel = compile(
                "@kernel\ndef empty(n: u32):\n    x = n + 1\n",
                Language::Python,
            )
            .unwrap();
            kernel.prepare_vendor(GpuVendor::Emulator).unwrap();
            assert!(kernel.vendor_code.is_empty());
            for vendor in [
                GpuVendor::Apple,
                GpuVendor::Nvidia,
                GpuVendor::Amd,
                GpuVendor::Intel,
            ] {
                kernel.prepare_vendor(vendor).unwrap();
                assert_eq!(
                    kernel.vendor_code[&vendor],
                    translate_to_vendor(&kernel.wbin, vendor).unwrap()
                );
                let pointer = kernel.vendor_code[&vendor].as_ptr();
                kernel.prepare_vendor(vendor).unwrap();
                assert_eq!(kernel.vendor_code[&vendor].as_ptr(), pointer);
            }
            assert_eq!(kernel.vendor_code.len(), 4);
        }
    }
}

pub use wave_runtime::error::RuntimeError;
pub use wave_runtime::Language;
