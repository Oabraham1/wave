// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! C interface for compilation, backend translation, and emulation.
//!
//! The runtime owns returned buffers; callers release them with `wave_free`. Errors are available
//! through `wave_last_error` on the calling thread.

// This module is the C ABI boundary; all pointer operations stay in documented blocks.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use crate::cache;
use crate::device::GpuVendor;
use std::cell::RefCell;
use std::{ptr, slice};
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

/// Compile kernel source to WAVE binary.
///
/// On success, writes the output pointer and length and returns 0.
/// On failure, stores an error message, retrievable via `wave_last_error`
/// and returns a nonzero error code.
///
/// The caller must free the output buffer with `wave_free`.
///
/// # Safety
///
/// `source_ptr` must be non-null and readable for `source_len` bytes in one
/// allocation, including when the length is zero. The length must not exceed
/// `isize::MAX`. The source must remain immutable for the duration of the call.
/// `out_ptr` and `out_len` must be aligned, writable, non-null, nonoverlapping
/// output slots that do not alias the source. Invalid UTF-8 returns an error.
#[no_mangle]
pub unsafe extern "C" fn wave_compile(
    source_ptr: *const u8,
    source_len: usize,
    language: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: The caller provides a readable, non-null source allocation for this length.
    let bytes = unsafe { slice::from_raw_parts(source_ptr, source_len) };
    let source = match std::str::from_utf8(bytes) {
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
            let boxed = wbin.into_boxed_slice();
            let len = boxed.len();
            let buffer = Box::into_raw(boxed).cast::<u8>();
            // SAFETY: The caller provides writable, aligned output slots that do not alias.
            unsafe {
                *out_ptr = buffer;
                *out_len = len;
            }
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
/// `wbin_ptr` must be non-null and readable for `wbin_len` bytes in one
/// allocation, including when empty. The length must not exceed `isize::MAX`.
/// The binary must remain immutable during the call. `out_ptr` and `out_len`
/// must be aligned, writable, non-null, nonoverlapping output slots that do
/// not alias the binary.
#[no_mangle]
pub unsafe extern "C" fn wave_translate(
    wbin_ptr: *const u8,
    wbin_len: usize,
    vendor: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    // SAFETY: The caller provides a readable, non-null binary allocation for this length.
    let wbin = unsafe { slice::from_raw_parts(wbin_ptr, wbin_len) };

    let Some(gpu_vendor) = vendor_from_u32(vendor) else {
        set_error(format!("invalid vendor: {vendor}"));
        return 2;
    };

    match crate::backend::translate_to_vendor(wbin, gpu_vendor) {
        Ok(code) => {
            let boxed = code.into_bytes().into_boxed_slice();
            let len = boxed.len();
            let buffer = Box::into_raw(boxed).cast::<u8>();
            // SAFETY: The caller provides writable, aligned output slots that do not alias.
            unsafe {
                *out_ptr = buffer;
                *out_len = len;
            }
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
/// `mem_ptr`/`mem_len` is the device memory buffer, read/write.
/// `reg_ptr`/`reg_count` is an array of, `register_index`, value, pairs.
/// `grid` and `workgroup` are [x, y, z] arrays.
///
/// On success, the memory buffer is modified in place and 0 is returned.
///
/// # Safety
///
/// All pointers must be non-null, aligned, and valid for their stated lengths
/// within single allocations, including zero-length buffers. Each allocation's
/// byte length must not exceed `isize::MAX`. `grid` and `workgroup` each point
/// to one initialized dimension array. `mem_ptr` must be exclusively writable
/// for `mem_len` bytes and must not overlap any other argument. All remaining
/// input allocations must stay immutable throughout the call.
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
    // SAFETY: The caller provides a readable, non-null binary allocation for this length.
    let wbin = unsafe { slice::from_raw_parts(wbin_ptr, wbin_len) };
    // SAFETY: The caller grants exclusive access to this writable memory allocation.
    let mem = unsafe { slice::from_raw_parts_mut(mem_ptr, mem_len) };
    // SAFETY: The caller provides aligned, readable register pairs for the full count.
    let regs = unsafe { slice::from_raw_parts(reg_ptr, reg_count) };
    // SAFETY: Both pointers refer to initialized, aligned dimension arrays.
    let (grid_dim, wg_dim) = unsafe { (*grid, *workgroup) };

    let mut config = wave_emu::EmulatorConfig {
        grid_dim,
        workgroup_dim: wg_dim,
        device_memory_size: mem_len.max(1024 * 1024),
        ..wave_emu::EmulatorConfig::default()
    };

    let mut initial_regs: Vec<(u8, u32)> = Vec::with_capacity(reg_count);
    for pair in regs {
        let Ok(idx) = u8::try_from(pair[0]) else {
            set_error(format!("invalid register index: {} (maximum 255)", pair[0]));
            return 2;
        };
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
/// `out_len` must be an aligned, writable, non-null pointer. The returned
/// bytes must not be modified or freed, and must not be used after a later
/// runtime FFI call on this thread or after this thread exits.
#[no_mangle]
pub unsafe extern "C" fn wave_last_error(out_len: *mut usize) -> *const u8 {
    LAST_ERROR.with(|e| {
        let s = e.borrow();
        // SAFETY: The caller provides an aligned, writable length slot.
        unsafe { *out_len = s.len() };
        s.as_ptr()
    })
}

/// Free a buffer previously returned by `wave_compile` or `wave_translate`.
///
/// # Safety
///
/// `ptr` must have been returned by a previous FFI call, and `len` must
/// match the returned length. Must not be called more than once per buffer.
/// The caller must not retain or use references to the buffer after freeing it.
/// A null pointer is accepted and does nothing.
#[no_mangle]
pub unsafe extern "C" fn wave_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: The pointer and length identify the original Box allocation, and the
        // caller transfers ownership back exactly once. Reconstructing it frees it.
        unsafe { drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ptr, len))) };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_buffer_can_be_read_and_freed() {
        let source = b"@kernel\ndef ffi_example(n: u32):\n    gid = thread_id()\n";
        let mut output = ptr::null_mut();
        let mut len = 0;
        // SAFETY: The source and distinct output slots are live for the call.
        let status = unsafe {
            wave_compile(
                source.as_ptr(),
                source.len(),
                0,
                &raw mut output,
                &raw mut len,
            )
        };
        assert_eq!(status, 0);
        // SAFETY: A successful compile returns an allocation of exactly len bytes.
        let data = unsafe { slice::from_raw_parts(output, len) };
        assert_eq!(&data[..4], b"WAVE");
        // SAFETY: Ownership of the allocation is returned once with its original length.
        unsafe { wave_free(output, len) };
    }

    #[test]
    fn invalid_register_index_returns_an_error() {
        let binary = [];
        let mut memory = [0u8; 4];
        let registers = [[256, 1]];
        let dimensions = [1, 1, 1];
        // SAFETY: All slices and dimension arrays are live and aligned; memory is exclusive.
        let status = unsafe {
            wave_emulate(
                binary.as_ptr(),
                0,
                memory.as_mut_ptr(),
                memory.len(),
                registers.as_ptr(),
                registers.len(),
                &raw const dimensions,
                &raw const dimensions,
            )
        };
        assert_eq!(status, 2);
    }
}
