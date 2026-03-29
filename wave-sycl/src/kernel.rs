// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! SYCL kernel wrapper generation. Emits the #include directives, using-namespace
//!
//! declaration, inline helper functions for bit_cast-based float conversion (rf/ri),
//! and the launch function skeleton that wraps the kernel body in SYCL's
//! queue::submit / handler::parallel_for / nd_range / nd_item lambda structure.
//! Local memory is allocated via sycl::local_accessor and exposed as a raw pointer.

use std::fmt::Write;

#[must_use]
pub fn emit_file_header() -> String {
    [
        "#include <sycl/sycl.hpp>",
        "#include <cstdint>",
        "",
        "using namespace sycl;",
        "",
        "inline float rf(uint32_t r) { return bit_cast<float>(r); }",
        "inline uint32_t ri(float f) { return bit_cast<uint32_t>(f); }",
        "inline half rh(uint32_t r) { return bit_cast<half>((uint16_t)(r & 0xFFFFu)); }",
        "inline uint32_t rhi(half h) { return (uint32_t)bit_cast<uint16_t>(h); }",
        "",
    ]
    .join("\n")
}

#[must_use]
pub fn emit_launch_start(name: &str, local_mem_size: u32) -> String {
    let kernel_name = if name.is_empty() { "wave_kernel" } else { name };

    let mut out = String::new();

    writeln!(
        out,
        "void {kernel_name}_launch(queue& q, uint8_t* device_mem_usm,"
    )
    .unwrap();
    writeln!(out, "        size_t grid_x, size_t grid_y, size_t grid_z,").unwrap();
    writeln!(out, "        size_t wg_x, size_t wg_y, size_t wg_z) {{").unwrap();
    writeln!(out, "    q.submit([&](handler& h) {{").unwrap();

    if local_mem_size > 0 {
        writeln!(
            out,
            "        local_accessor<uint8_t, 1> local_acc(range<1>({local_mem_size}), h);"
        )
        .unwrap();
    }

    writeln!(out, "        h.parallel_for(").unwrap();
    writeln!(
        out,
        "            nd_range<3>(range<3>(grid_x * wg_x, grid_y * wg_y, grid_z * wg_z),"
    )
    .unwrap();
    writeln!(out, "                        range<3>(wg_x, wg_y, wg_z)),").unwrap();
    writeln!(out, "            [=](nd_item<3> it) {{").unwrap();
    writeln!(out, "                auto sg = it.get_sub_group();").unwrap();

    if local_mem_size > 0 {
        writeln!(
            out,
            "                uint8_t* lm = local_acc.get_multi_ptr<access::decorated::no>().get();"
        )
        .unwrap();
    }

    out
}

#[must_use]
pub fn emit_launch_end() -> &'static str {
    "            }\n\
     \x20       );\n\
     \x20   }).wait();\n\
     }\n"
}
