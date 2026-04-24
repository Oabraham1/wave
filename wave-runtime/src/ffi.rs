// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! C FFI exports for the WAVE runtime shared library.
//!
//! Exposes the in-process compilation, backend translation, and emulator
//! execution pipeline as C-callable functions. Python (via ctypes), Node.js
//! (via ffi-napi), and other language SDKs load the cdylib and call these
//! functions directly, avoiding subprocess overhead entirely.
//!
//! Memory protocol: output buffers are allocated by Rust and must be freed
//! by calling `wave_free`. Error messages are stored in a thread-local
//! string retrievable via `wave_last_error`.

#![allow(unsafe_code)]

use crate::cache;
use crate::device::GpuVendor;
use std::cell::RefCell;
use std::slice;
use wave_compiler::Language;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

fn language_from_u32(v: u32) -> Option<Language> {
    match v {
        0 => Some(Language::Python),
        1 => Some(Language::Rust),
        2 => Some(Language::Cpp),
        3 => Some(Language::TypeScript),
        _ => None,
    }
}

fn vendor_from_u32(v: u32) -> Option<GpuVendor> {
    match v {
        0 => Some(GpuVendor::Apple),
        1 => Some(GpuVendor::Nvidia),
        2 => Some(GpuVendor::Amd),
        3 => Some(GpuVendor::Intel),
        4 => Some(GpuVendor::Emulator),
        _ => None,
    }
}

/// Compile kernel source to WAVE binary (.wbin).
///
/// On success, writes the output pointer and length and returns 0.
/// On failure, stores an error message (retrievable via `wave_last_error`)
/// and returns a nonzero error code.
///
/// The caller must free the output buffer with `wave_free`.
///
/// # Safety
///
/// `source_ptr` must point to `source_len` valid UTF-8 bytes.
/// `out_ptr` and `out_len` must be valid, aligned, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn wave_compile(
    source_ptr: *const u8,
    source_len: usize,
    language: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let source = match std::str::from_utf8(slice::from_raw_parts(source_ptr, source_len)) {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("invalid UTF-8 source: {e}"));
            return 1;
        }
    };

    let Some(lang) = language_from_u32(language) else {
        set_error(format!("invalid language: {language}"));
        return 2;
    };

    match cache::compile_cached(source, lang) {
        Ok(wbin) => {
            let mut boxed = wbin.into_boxed_slice();
            *out_ptr = boxed.as_mut_ptr();
            *out_len = boxed.len();
            std::mem::forget(boxed);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            3
        }
    }
}

/// Translate a WAVE binary to vendor-specific source code.
///
/// # Safety
///
/// `wbin_ptr` must point to `wbin_len` valid bytes.
/// `out_ptr` and `out_len` must be valid, aligned, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn wave_translate(
    wbin_ptr: *const u8,
    wbin_len: usize,
    vendor: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let wbin = slice::from_raw_parts(wbin_ptr, wbin_len);

    let Some(gpu_vendor) = vendor_from_u32(vendor) else {
        set_error(format!("invalid vendor: {vendor}"));
        return 2;
    };

    match crate::backend::translate_to_vendor(wbin, gpu_vendor) {
        Ok(code) => {
            let bytes = code.into_bytes();
            let mut boxed = bytes.into_boxed_slice();
            *out_ptr = boxed.as_mut_ptr();
            *out_len = boxed.len();
            std::mem::forget(boxed);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            3
        }
    }
}

/// Run a WAVE binary on the emulator with the given memory and registers.
///
/// `mem_ptr`/`mem_len` is the device memory buffer (read/write).
/// `reg_ptr`/`reg_count` is an array of (`register_index`, value) pairs.
/// `grid` and `workgroup` are [x, y, z] arrays.
///
/// On success, the memory buffer is modified in place and 0 is returned.
///
/// # Safety
///
/// All pointer arguments must be valid for their stated lengths.
/// `mem_ptr` must be writable for `mem_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn wave_emulate(
    wbin_ptr: *const u8,
    wbin_len: usize,
    mem_ptr: *mut u8,
    mem_len: usize,
    reg_ptr: *const [u32; 2],
    reg_count: usize,
    grid: *const [u32; 3],
    workgroup: *const [u32; 3],
) -> i32 {
    let wbin = slice::from_raw_parts(wbin_ptr, wbin_len);
    let mem = slice::from_raw_parts_mut(mem_ptr, mem_len);
    let regs = slice::from_raw_parts(reg_ptr, reg_count);
    let grid_dim = *grid;
    let wg_dim = *workgroup;

    let mut config = wave_emu::EmulatorConfig {
        grid_dim,
        workgroup_dim: wg_dim,
        device_memory_size: mem_len.max(1024 * 1024),
        ..wave_emu::EmulatorConfig::default()
    };

    let mut initial_regs: Vec<(u8, u32)> = Vec::with_capacity(reg_count);
    for pair in regs {
        #[allow(clippy::cast_possible_truncation)]
        let idx = pair[0] as u8;
        initial_regs.push((idx, pair[1]));
    }
    config.initial_registers = initial_regs;

    let mut emu = wave_emu::Emulator::new(config);
    if let Err(e) = emu.load_binary(wbin) {
        set_error(e.to_string());
        return 3;
    }

    if let Err(e) = emu.load_device_memory(0, mem) {
        set_error(e.to_string());
        return 4;
    }

    if let Err(e) = emu.run() {
        set_error(e.to_string());
        return 5;
    }

    match emu.read_device_memory(0, mem_len) {
        Ok(result) => {
            mem.copy_from_slice(&result);
            0
        }
        Err(e) => {
            set_error(e.to_string());
            6
        }
    }
}

/// Retrieve the last error message.
///
/// Returns a pointer to a UTF-8 string and writes its length to `out_len`.
/// The pointer is valid until the next FFI call on the same thread.
///
/// # Safety
///
/// `out_len` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn wave_last_error(out_len: *mut usize) -> *const u8 {
    LAST_ERROR.with(|e| {
        let s = e.borrow();
        *out_len = s.len();
        s.as_ptr()
    })
}

/// Free a buffer previously returned by `wave_compile` or `wave_translate`.
///
/// # Safety
///
/// `ptr` must have been returned by a previous FFI call, and `len` must
/// match the returned length. Must not be called more than once per buffer.
#[no_mangle]
pub unsafe extern "C" fn wave_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let _: Box<[u8]> = Box::from(slice::from_raw_parts_mut(ptr, len));
    }
}

/// Clear the kernel compilation cache.
#[no_mangle]
pub extern "C" fn wave_cache_clear() {
    cache::clear_cache();
}

/// Return the number of cached kernel entries.
#[no_mangle]
pub extern "C" fn wave_cache_size() -> usize {
    cache::cache_size()
}
