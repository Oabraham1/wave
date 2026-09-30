// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Compilation, backend translation, device detection, buffers, and kernel launch shared by the
//! WAVE SDKs.

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
pub use device::{
    detect_gpu, enumerate_devices, Device, DeviceCapabilities, DeviceInfo, GpuVendor,
};
pub use error::RuntimeError;
pub use launcher::launch_kernel;
pub use memory::{DeviceBuffer, ElementType};
pub use multi::{
    allreduce_average, gather_shards, launch_on_device, replicate_buffer, select_reduce_strategy,
    shard_tensor, ReduceStrategy, TensorShard,
};

pub use wave_compiler::Language;
