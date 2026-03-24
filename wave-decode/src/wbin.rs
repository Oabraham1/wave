// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WBIN container reader. Parses the .wbin binary format produced by wave-asm,
//!
//! extracting header information, code sections, symbols, and kernel metadata.

use thiserror::Error;

pub const WBIN_MAGIC: &[u8; 4] = b"WAVE";
pub const WBIN_VERSION: u16 = 0x0001;
pub const WBIN_HEADER_SIZE: usize = 32;
pub const KERNEL_METADATA_SIZE: usize = 32;

#[derive(Debug, Error)]
pub enum WbinError {
    #[error("file too small: expected at least {expected} bytes, got {actual}")]
    FileTooSmall { expected: usize, actual: usize },

    #[error("invalid magic: expected 'WAVE', got {actual:?}")]
    InvalidMagic { actual: [u8; 4] },

    #[error("unsupported version: {version}")]
    UnsupportedVersion { version: u16 },

    #[error("invalid offset: {field} offset {offset} exceeds file size {file_size}")]
    InvalidOffset {
        field: &'static str,
        offset: u32,
        file_size: usize,
    },

    #[error("invalid size: {field} at offset {offset} with size {size} exceeds file size {file_size}")]
    InvalidSize {
        field: &'static str,
        offset: u32,
        size: u32,
        file_size: usize,
    },

    #[error("unterminated string in symbol table at offset {offset}")]
    UnterminatedString { offset: u32 },

    #[error("invalid kernel count: {count} kernels but metadata size is {metadata_size}")]
    InvalidKernelCount { count: u32, metadata_size: u32 },
}

#[derive(Debug, Clone)]
pub struct WbinHeader {
    pub version: u16,
    pub flags: u16,
    pub code_offset: u32,
    pub code_size: u32,
    pub symbol_offset: u32,
    pub symbol_size: u32,
    pub metadata_offset: u32,
    pub metadata_size: u32,
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub name: String,
    pub register_count: u32,
    pub local_memory_size: u32,
    pub workgroup_size: [u32; 3],
    pub code_offset: u32,
    pub code_size: u32,
}

#[derive(Debug, Clone)]
pub struct WbinFile<'a> {
    data: &'a [u8],
    pub header: WbinHeader,
    pub kernels: Vec<KernelInfo>,
}

impl<'a> WbinFile<'a> {
    #[allow(clippy::missing_panics_doc)]
    pub fn parse(data: &'a [u8]) -> Result<Self, WbinError> {
        if data.len() < WBIN_HEADER_SIZE {
            return Err(WbinError::FileTooSmall {
                expected: WBIN_HEADER_SIZE,
                actual: data.len(),
            });
        }

        let magic: [u8; 4] = data[0..4].try_into().unwrap();
        if &magic != WBIN_MAGIC {
            return Err(WbinError::InvalidMagic { actual: magic });
        }

        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != WBIN_VERSION {
            return Err(WbinError::UnsupportedVersion { version });
        }

        let flags = u16::from_le_bytes([data[6], data[7]]);
        let code_offset = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let code_size = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let symbol_offset = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let symbol_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let metadata_offset = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        let metadata_size = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);

        Self::validate_section(data.len(), "code", code_offset, code_size)?;
        if symbol_offset != 0 || symbol_size != 0 {
            Self::validate_section(data.len(), "symbol", symbol_offset, symbol_size)?;
        }
        Self::validate_section(data.len(), "metadata", metadata_offset, metadata_size)?;

        let header = WbinHeader {
            version,
            flags,
            code_offset,
            code_size,
            symbol_offset,
            symbol_size,
            metadata_offset,
            metadata_size,
        };

        let kernels = Self::parse_kernels(data, &header)?;

        Ok(Self {
            data,
            header,
            kernels,
        })
    }

    fn validate_section(
        file_size: usize,
        field: &'static str,
        offset: u32,
        size: u32,
    ) -> Result<(), WbinError> {
        if offset as usize > file_size {
            return Err(WbinError::InvalidOffset {
                field,
                offset,
                file_size,
            });
        }
        if (offset as usize + size as usize) > file_size {
            return Err(WbinError::InvalidSize {
                field,
                offset,
                size,
                file_size,
            });
        }
        Ok(())
    }

    fn parse_kernels(data: &[u8], header: &WbinHeader) -> Result<Vec<KernelInfo>, WbinError> {
        if header.metadata_size < 4 {
            return Ok(Vec::new());
        }

        let meta_start = header.metadata_offset as usize;
        let kernel_count =
            u32::from_le_bytes([data[meta_start], data[meta_start + 1], data[meta_start + 2], data[meta_start + 3]]);

        #[allow(clippy::cast_possible_truncation)]
        let expected_size = 4 + kernel_count * (KERNEL_METADATA_SIZE as u32);
        if header.metadata_size < expected_size {
            return Err(WbinError::InvalidKernelCount {
                count: kernel_count,
                metadata_size: header.metadata_size,
            });
        }

        let mut kernels = Vec::with_capacity(kernel_count as usize);
        let mut offset = meta_start + 4;

        for _ in 0..kernel_count {
            let name_offset = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
            let register_count = u32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]);
            let local_memory_size = u32::from_le_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]]);
            let ws_x = u32::from_le_bytes([data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15]]);
            let ws_y = u32::from_le_bytes([data[offset + 16], data[offset + 17], data[offset + 18], data[offset + 19]]);
            let ws_z = u32::from_le_bytes([data[offset + 20], data[offset + 21], data[offset + 22], data[offset + 23]]);
            let code_offset = u32::from_le_bytes([data[offset + 24], data[offset + 25], data[offset + 26], data[offset + 27]]);
            let code_size = u32::from_le_bytes([data[offset + 28], data[offset + 29], data[offset + 30], data[offset + 31]]);

            let name = if name_offset != 0 && header.symbol_size > 0 {
                Self::read_string(data, name_offset as usize)?
            } else {
                String::new()
            };

            kernels.push(KernelInfo {
                name,
                register_count,
                local_memory_size,
                workgroup_size: [ws_x, ws_y, ws_z],
                code_offset,
                code_size,
            });

            offset += KERNEL_METADATA_SIZE;
        }

        Ok(kernels)
    }

    fn read_string(data: &[u8], offset: usize) -> Result<String, WbinError> {
        let mut end = offset;
        while end < data.len() && data[end] != 0 {
            end += 1;
        }
        if end >= data.len() {
            #[allow(clippy::cast_possible_truncation)]
            return Err(WbinError::UnterminatedString {
                offset: offset as u32,
            });
        }
        Ok(String::from_utf8_lossy(&data[offset..end]).to_string())
    }

    #[must_use]
    pub fn code(&self) -> &[u8] {
        let start = self.header.code_offset as usize;
        let end = start + self.header.code_size as usize;
        &self.data[start..end]
    }

    #[must_use]
    pub fn kernel_code(&self, kernel_index: usize) -> Option<&[u8]> {
        let kernel = self.kernels.get(kernel_index)?;
        let start = self.header.code_offset as usize + kernel.code_offset as usize;
        let end = start + kernel.code_size as usize;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    #[must_use]
    pub fn find_kernel(&self, name: &str) -> Option<&KernelInfo> {
        self.kernels.iter().find(|k| k.name == name)
    }

    #[must_use]
    pub fn has_symbols(&self) -> bool {
        self.header.symbol_size > 0
    }

    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_wbin() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(&WBIN_VERSION.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&0x20u32.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0x24u32.to_le_bytes());
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        data.extend_from_slice(&0u32.to_le_bytes());
        data
    }

    fn make_wbin_with_kernel() -> Vec<u8> {
        let mut data = Vec::new();
        let kernel_name = b"test_kernel\0";
        let code_bytes: [u8; 8] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];

        let code_offset: u32 = 0x20;
        let code_size: u32 = 8;
        let symbol_offset: u32 = code_offset + code_size;
        let symbol_size: u32 = kernel_name.len() as u32;
        let metadata_offset: u32 = symbol_offset + symbol_size;
        let metadata_size: u32 = 4 + 32;

        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(&WBIN_VERSION.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&code_offset.to_le_bytes());
        data.extend_from_slice(&code_size.to_le_bytes());
        data.extend_from_slice(&symbol_offset.to_le_bytes());
        data.extend_from_slice(&symbol_size.to_le_bytes());
        data.extend_from_slice(&metadata_offset.to_le_bytes());
        data.extend_from_slice(&metadata_size.to_le_bytes());
        data.extend_from_slice(&code_bytes);
        data.extend_from_slice(kernel_name);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&symbol_offset.to_le_bytes());
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1024u32.to_le_bytes());
        data.extend_from_slice(&256u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&8u32.to_le_bytes());
        data
    }

    #[test]
    fn test_parse_minimal() {
        let data = make_minimal_wbin();
        let wbin = WbinFile::parse(&data).unwrap();

        assert_eq!(wbin.header.version, WBIN_VERSION);
        assert_eq!(wbin.header.code_size, 4);
        assert_eq!(wbin.kernels.len(), 0);
    }

    #[test]
    fn test_parse_with_kernel() {
        let data = make_wbin_with_kernel();
        let wbin = WbinFile::parse(&data).unwrap();

        assert_eq!(wbin.kernels.len(), 1);
        assert_eq!(wbin.kernels[0].name, "test_kernel");
        assert_eq!(wbin.kernels[0].register_count, 16);
        assert_eq!(wbin.kernels[0].local_memory_size, 1024);
        assert_eq!(wbin.kernels[0].workgroup_size, [256, 1, 1]);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = make_minimal_wbin();
        data[0] = b'X';

        let err = WbinFile::parse(&data).unwrap_err();
        assert!(matches!(err, WbinError::InvalidMagic { .. }));
    }

    #[test]
    fn test_file_too_small() {
        let data = vec![0u8; 16];
        let err = WbinFile::parse(&data).unwrap_err();
        assert!(matches!(err, WbinError::FileTooSmall { .. }));
    }

    #[test]
    fn test_get_code() {
        let data = make_wbin_with_kernel();
        let wbin = WbinFile::parse(&data).unwrap();

        let code = wbin.code();
        assert_eq!(code.len(), 8);
        assert_eq!(code, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
    }

    #[test]
    fn test_kernel_code() {
        let data = make_wbin_with_kernel();
        let wbin = WbinFile::parse(&data).unwrap();

        let code = wbin.kernel_code(0).unwrap();
        assert_eq!(code.len(), 8);
    }

    #[test]
    fn test_find_kernel() {
        let data = make_wbin_with_kernel();
        let wbin = WbinFile::parse(&data).unwrap();

        let kernel = wbin.find_kernel("test_kernel").unwrap();
        assert_eq!(kernel.register_count, 16);

        assert!(wbin.find_kernel("nonexistent").is_none());
    }

    #[test]
    fn test_has_symbols() {
        let data_with_symbols = make_wbin_with_kernel();
        let wbin_with = WbinFile::parse(&data_with_symbols).unwrap();
        assert!(wbin_with.has_symbols());

        let data_stripped = make_minimal_wbin();
        let wbin_stripped = WbinFile::parse(&data_stripped).unwrap();
        assert!(!wbin_stripped.has_symbols());
    }
}
