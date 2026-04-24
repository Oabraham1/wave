// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Kernel compilation cache for the WAVE runtime.
//!
//! Avoids recompiling the same kernel source by caching the compiled WAVE
//! binary and vendor-specific source code. The cache key is a 64-bit hash
//! of the source text combined with the target vendor. First compilation
//! runs the full pipeline; subsequent lookups return cached results in
//! microseconds.

use crate::device::GpuVendor;
use crate::error::RuntimeError;
use std::collections::HashMap;
use std::sync::Mutex;

use wave_compiler::Language;

struct CacheEntry {
    wbin: Vec<u8>,
    vendor_code: HashMap<GpuVendor, String>,
}

static CACHE: Mutex<Option<HashMap<u64, CacheEntry>>> = Mutex::new(None);

fn hash_key(source: &str, language: Language) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in source.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h ^= language as u64;
    h
}

/// Compile kernel source to WAVE binary, returning a cached result if available.
///
/// # Errors
///
/// Returns `RuntimeError::Compile` if the source cannot be compiled.
pub fn compile_cached(source: &str, language: Language) -> Result<Vec<u8>, RuntimeError> {
    let key = hash_key(source, language);

    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cache = guard.get_or_insert_with(HashMap::new);

    if let Some(entry) = cache.get(&key) {
        return Ok(entry.wbin.clone());
    }
    drop(guard);

    let wbin = crate::compiler::compile_kernel(source, language)?;

    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cache = guard.get_or_insert_with(HashMap::new);
    cache.insert(
        key,
        CacheEntry {
            wbin: wbin.clone(),
            vendor_code: HashMap::new(),
        },
    );

    Ok(wbin)
}

/// Translate a cached WAVE binary to vendor-specific source code.
///
/// If the wbin was produced by `compile_cached`, this may hit the cache for
/// the vendor translation as well.
///
/// # Errors
///
/// Returns `RuntimeError::Backend` if the translation fails.
pub fn translate_cached(
    source: &str,
    language: Language,
    vendor: GpuVendor,
) -> Result<String, RuntimeError> {
    let key = hash_key(source, language);

    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cache = guard.get_or_insert_with(HashMap::new);

    if let Some(entry) = cache.get(&key) {
        if let Some(code) = entry.vendor_code.get(&vendor) {
            return Ok(code.clone());
        }
    }
    drop(guard);

    let wbin = compile_cached(source, language)?;
    let vendor_code = crate::backend::translate_to_vendor(&wbin, vendor)?;

    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(entry) = cache.get_mut(&key) {
        entry.vendor_code.insert(vendor, vendor_code.clone());
    }

    Ok(vendor_code)
}

/// Clear the kernel cache.
pub fn clear_cache() {
    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = None;
}

/// Return the number of cached kernel entries.
pub fn cache_size() -> usize {
    let guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard.as_ref().map_or(0, HashMap::len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_cached_returns_wbin() {
        let source = r#"
@kernel
def cache_test_1(n: u32):
    gid = thread_id()
"#;
        let wbin = compile_cached(source, Language::Python).unwrap();
        assert_eq!(&wbin[0..4], b"WAVE");

        let wbin2 = compile_cached(source, Language::Python).unwrap();
        assert_eq!(wbin, wbin2);
    }

    #[test]
    fn test_different_sources_produce_different_wbin() {
        let src1 = r#"
@kernel
def cache_test_a(a: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        a[gid] = a[gid]
"#;
        let src2 = r#"
@kernel
def cache_test_b(a: Buffer[f32], b: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        b[gid] = a[gid]
"#;
        let wbin1 = compile_cached(src1, Language::Python).unwrap();
        let wbin2 = compile_cached(src2, Language::Python).unwrap();
        assert_ne!(wbin1, wbin2);
    }

    #[test]
    fn test_clear_and_size() {
        clear_cache();
        assert_eq!(cache_size(), 0);

        let source = r#"
@kernel
def cache_test_clear(n: u32):
    gid = thread_id()
"#;
        compile_cached(source, Language::Python).unwrap();
        assert!(cache_size() >= 1);

        clear_cache();
        assert_eq!(cache_size(), 0);
    }
}
