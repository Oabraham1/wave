// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Generate HIP C++ kernel declarations and conversion helpers.

#[must_use]
pub fn emit_file_header() -> String {
    [
        "#include <hip/hip_runtime.h>",
        "#include <hip/hip_fp16.h>",
        "#include <cstdint>",
        "",
        "__device__ inline float rf(uint32_t r) { return __uint_as_float(r); }",
        "__device__ inline uint32_t ri(float f) { return __float_as_uint(f); }",
        "__device__ inline __half rh(uint32_t r) { return __ushort_as_half((unsigned short)(r & 0xFFFFu)); }",
        "__device__ inline uint32_t rhi(__half h) { return (uint32_t)__half_as_ushort(h); }",
        "",
    ]
    .join("\n")
}

#[must_use]
pub fn emit_kernel_signature(name: &str) -> String {
    let kernel_name = if name.is_empty() { "wave_kernel" } else { name };

    format!("__global__ void {kernel_name}(uint8_t* device_mem) {{\n")
}

#[must_use]
pub fn emit_kernel_footer() -> &'static str {
    "}"
}
