// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Generate PTX kernel declarations and conversion helpers.

#[must_use]
pub fn emit_header(sm_version: u32) -> String {
    format!(
        ".version 7.5\n\
         .target sm_{sm_version}\n\
         .address_size 64\n\n"
    )
}

#[must_use]
pub fn emit_entry_start(name: &str) -> String {
    let kernel_name = if name.is_empty() { "wave_kernel" } else { name };

    format!(
        ".visible .entry {kernel_name}(\n\
         \x20   .param .u64 _device_mem_ptr\n\
         ) {{\n"
    )
}

#[must_use]
pub fn emit_entry_end() -> &'static str {
    "\n    ret;\n}\n"
}
