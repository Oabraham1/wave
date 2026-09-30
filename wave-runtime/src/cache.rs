// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Cache compiled WAVE binaries and backend source by source text, language, and target vendor.

use crate::device::GpuVendor;
use crate::error::RuntimeError;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use wave_compiler::Language;

#[derive(Debug)]
struct CacheEntry {
    wbin: Arc<[u8]>,
    vendor_code: HashMap<GpuVendor, String>,
}

type KernelCache = HashMap<Language, HashMap<String, CacheEntry>>;
static CACHE: LazyLock<Mutex<KernelCache>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Compile kernel source to WAVE binary, returning a cached result if available.
///
/// # Errors
///
/// Returns `RuntimeError::Compile` if the source cannot be compiled.
pub fn compile_cached(source: &str, language: Language) -> Result<Vec<u8>, RuntimeError> {
    Ok(compile_shared(source, language)?.to_vec())
}

fn compile_shared(source: &str, language: Language) -> Result<Arc<[u8]>, RuntimeError> {
    let guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = guard.get(&language).and_then(|sources| sources.get(source)) {
        return Ok(Arc::clone(&entry.wbin));
    }
    drop(guard);

    let wbin: Arc<[u8]> = crate::compiler::compile_kernel(source, language)?.into();
    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = guard
        .entry(language)
        .or_default()
        .entry(source.to_owned())
        .or_insert_with(|| CacheEntry {
            wbin,
            vendor_code: HashMap::new(),
        });
    Ok(Arc::clone(&entry.wbin))
}

/// Translate kernel source, caching the binary and each vendor's generated source.
///
/// # Errors
///
/// Returns an error if compilation or translation fails.
pub fn translate_cached(
    source: &str,
    language: Language,
    vendor: GpuVendor,
) -> Result<String, RuntimeError> {
    let guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(code) = guard
        .get(&language)
        .and_then(|sources| sources.get(source))
        .and_then(|entry| entry.vendor_code.get(&vendor))
    {
        return Ok(code.clone());
    }
    drop(guard);

    let wbin = compile_shared(source, language)?;
    let vendor_code = crate::backend::translate_to_vendor(&wbin, vendor)?;

    let mut guard = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = guard
        .get_mut(&language)
        .and_then(|sources| sources.get_mut(source))
    {
        entry.vendor_code.insert(vendor, vendor_code.clone());
    }
    Ok(vendor_code)
}

/// Clear the kernel cache.
pub fn clear_cache() {
    CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

/// Return the number of cached kernel entries.
pub fn cache_size() -> usize {
    CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values()
        .map(HashMap::len)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_compile_cached_returns_wbin() {
        let _guard = TEST_LOCK.lock().unwrap();
        let source = r"
@kernel
def cache_test_1(n: u32):
    gid = thread_id()
";
        let wbin = compile_cached(source, Language::Python).unwrap();
        assert_eq!(&wbin[0..4], b"WAVE");

        let wbin2 = compile_cached(source, Language::Python).unwrap();
        assert_eq!(wbin, wbin2);
    }

    #[test]
    fn test_different_sources_produce_different_wbin() {
        let _guard = TEST_LOCK.lock().unwrap();
        let src1 = r"
@kernel
def cache_test_a(a: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        a[gid] = a[gid]
";
        let src2 = r"
@kernel
def cache_test_b(a: Buffer[f32], b: Buffer[f32], n: u32):
    gid = thread_id()
    if gid < n:
        b[gid] = a[gid]
";
        let wbin1 = compile_cached(src1, Language::Python).unwrap();
        let wbin2 = compile_cached(src2, Language::Python).unwrap();
        assert_ne!(wbin1, wbin2);
    }

    #[test]
    fn test_clear_and_size() {
        let _guard = TEST_LOCK.lock().unwrap();
        clear_cache();
        assert_eq!(cache_size(), 0);

        let source = r"
@kernel
def cache_test_clear(n: u32):
    gid = thread_id()
";
        compile_cached(source, Language::Python).unwrap();
        assert!(cache_size() >= 1);

        clear_cache();
        assert_eq!(cache_size(), 0);
    }
}
