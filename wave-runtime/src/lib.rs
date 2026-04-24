// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WAVE runtime: shared core for all WAVE SDKs.
//!
//! Provides the complete pipeline from kernel source code to GPU execution:
//! compilation, backend translation, device detection, memory management,
//! and kernel launch. Used by the Python, Rust, C/C++, and JavaScript SDKs.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_pass_by_value
)]

pub mod backend;
pub mod cache;
pub mod compiler;
pub mod device;
pub mod error;
pub mod ffi;
pub mod launcher;
pub mod memory;
pub mod multi;

pub use backend::translate_to_vendor;
pub use cache::{cache_size, clear_cache, compile_cached, translate_cached};
pub use compiler::{compile_kernel, compile_kernel_with_config};
pub use device::{detect_gpu, enumerate_devices, Device, DeviceCapabilities, DeviceInfo, GpuVendor};
pub use error::RuntimeError;
pub use launcher::launch_kernel;
pub use memory::{DeviceBuffer, ElementType};
pub use multi::{
    allreduce_average, gather_shards, launch_on_device, replicate_buffer, select_reduce_strategy,
    shard_tensor, ReduceStrategy, TensorShard,
};

pub use wave_compiler::Language;
