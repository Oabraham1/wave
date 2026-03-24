// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Kernel wrapper generation. Emits the Metal Shading Language boilerplate that
//!
//! surrounds translated WAVE code: the #include directives, helper inline functions
//! for float bitcasting (rf/ri) and half-precision conversion (rh/rhi/rh2/rh2i),
//! and the kernel function signature with all Metal built-in parameter bindings.

#[must_use]
pub fn emit_file_header() -> String {
    [
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        "inline float rf(uint32_t r) { return as_type<float>(r); }",
        "inline uint32_t ri(float f) { return as_type<uint32_t>(f); }",
        "inline half rh(uint32_t r) { return as_type<half>((ushort)(r & 0xFFFFu)); }",
        "inline uint32_t rhi(half h) { return (uint32_t)as_type<ushort>(h); }",
        "inline half2 rh2(uint32_t r) { return as_type<half2>(r); }",
        "inline uint32_t rh2i(half2 h) { return as_type<uint32_t>(h); }",
        "",
    ]
    .join("\n")
}

#[must_use]
pub fn emit_kernel_signature(name: &str) -> String {
    let kernel_name = if name.is_empty() { "wave_kernel" } else { name };

    [
        format!("kernel void {kernel_name}("),
        "    device uint8_t* device_mem [[buffer(0)]],".to_string(),
        "    uint3 tid [[thread_position_in_threadgroup]],".to_string(),
        "    uint3 gid [[threadgroup_position_in_grid]],".to_string(),
        "    uint3 tsize [[threads_per_threadgroup]],".to_string(),
        "    uint3 grid_dim [[threadgroups_per_grid]],".to_string(),
        "    uint lane_id [[thread_index_in_simdgroup]],".to_string(),
        "    uint simd_id [[simdgroup_index_in_threadgroup]]".to_string(),
        ") {".to_string(),
    ]
    .join("\n")
}

#[must_use]
pub fn emit_kernel_footer() -> &'static str {
    "}"
}
