// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// WAVE GPU SDK for JavaScript/TypeScript.
///
/// Write GPU kernels in TypeScript, run on any GPU. Supports Apple Metal,
/// NVIDIA CUDA, AMD ROCm, Intel SYCL, and a built-in emulator.

export { device, DeviceInfo, GpuVendor } from "./device";
export { WaveArray, array, zeros, ones, DType } from "./array";
export { kernel, CompiledKernel, Language } from "./kernel";
