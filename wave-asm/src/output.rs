// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! WBIN binary writer. Produces the final output format: 32-byte header, code section, symbol
//! table, optional, and kernel metadata. Supports multiple kernels and stripping symbols for
//! release builds.

use crate::ast::KernelMetadata;
use crate::diagnostics::AssemblerError;
use crate::encoder::EncodedInstruction;
use std::io::Write;

const WBIN_VERSION: u16 = 0x0001;

#[derive(Debug)]
pub struct WbinWriter {
    kernels: Vec<KernelMetadata>,
    code: Vec<u8>,
    strip_symbols: bool,
}

impl WbinWriter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            code: Vec::new(),
            strip_symbols: false,
        }
    }

    pub fn set_strip_symbols(&mut self, strip: bool) {
        self.strip_symbols = strip;
    }

    /// Begin a kernel at the current code offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the code offset exceeds the WBIN size limit.
    pub fn begin_kernel(&mut self, name: String) -> Result<(), AssemblerError> {
        let metadata = KernelMetadata {
            name,
            code_offset: binary_size(self.code.len())?,
            ..Default::default()
        };
        self.kernels.push(metadata);
        Ok(())
    }

    pub fn set_register_count(&mut self, count: u32) {
        if let Some(kernel) = self.kernels.last_mut() {
            kernel.register_count = count;
        }
    }

    pub fn set_local_memory_size(&mut self, size: u32) {
        if let Some(kernel) = self.kernels.last_mut() {
            kernel.local_memory_size = size;
        }
    }

    pub fn set_workgroup_size(&mut self, x: u32, y: u32, z: u32) {
        if let Some(kernel) = self.kernels.last_mut() {
            kernel.workgroup_size = [x, y, z];
        }
    }

    /// Record the size of the current kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the code size exceeds the WBIN size limit.
    pub fn end_kernel(&mut self) -> Result<(), AssemblerError> {
        if let Some(kernel) = self.kernels.last_mut() {
            kernel.code_size = binary_size(self.code.len())? - kernel.code_offset;
        }
        Ok(())
    }

    pub fn write_instruction(&mut self, inst: &EncodedInstruction) {
        self.code.extend_from_slice(&inst.word0.to_le_bytes());
        if let Some(word1) = inst.word1 {
            self.code.extend_from_slice(&word1.to_le_bytes());
        }
    }

    /// Write a WBIN file to the output stream.
    ///
    /// # Errors
    ///
    /// Returns an error if a section exceeds the WBIN size limit or the output write fails.
    pub fn finish<W: Write>(&self, writer: &mut W) -> Result<(), AssemblerError> {
        let code_offset = 32u32;
        let code_size = binary_size(self.code.len())?;
        let code_end = code_offset
            .checked_add(code_size)
            .ok_or(AssemblerError::BinaryTooLarge)?;
        let symbol_offset = if self.strip_symbols { 0 } else { code_end };
        let mut symbols = Vec::new();
        let mut name_offsets = Vec::with_capacity(self.kernels.len());
        for kernel in &self.kernels {
            if self.strip_symbols {
                name_offsets.push(0);
            } else {
                name_offsets.push(
                    symbol_offset
                        .checked_add(binary_size(symbols.len())?)
                        .ok_or(AssemblerError::BinaryTooLarge)?,
                );
                symbols.extend_from_slice(kernel.name.as_bytes());
                symbols.push(0);
            }
        }
        let symbol_size = binary_size(symbols.len())?;
        let metadata_offset = code_end
            .checked_add(symbol_size)
            .ok_or(AssemblerError::BinaryTooLarge)?;
        let count = binary_size(self.kernels.len())?;
        let metadata_size = count
            .checked_mul(32)
            .and_then(|size| size.checked_add(4))
            .ok_or(AssemblerError::BinaryTooLarge)?;
        metadata_offset
            .checked_add(metadata_size)
            .ok_or(AssemblerError::BinaryTooLarge)?;

        let mut header = Vec::with_capacity(32);
        header.extend_from_slice(b"WAVE");
        header.extend_from_slice(&WBIN_VERSION.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes());
        for value in [
            code_offset,
            code_size,
            symbol_offset,
            symbol_size,
            metadata_offset,
            metadata_size,
        ] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        let write_error = |error: std::io::Error| AssemblerError::IoError {
            message: format!("failed to write WBIN: {error}"),
        };
        writer.write_all(&header).map_err(write_error)?;
        writer.write_all(&self.code).map_err(write_error)?;
        writer.write_all(&symbols).map_err(write_error)?;
        writer
            .write_all(&count.to_le_bytes())
            .map_err(write_error)?;
        for (kernel, name_offset) in self.kernels.iter().zip(name_offsets) {
            for value in [
                name_offset,
                kernel.register_count,
                kernel.local_memory_size,
                kernel.workgroup_size[0],
                kernel.workgroup_size[1],
                kernel.workgroup_size[2],
                kernel.code_offset,
                kernel.code_size,
            ] {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(write_error)?;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn code_bytes(&self) -> &[u8] {
        &self.code
    }

    #[must_use]
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }
}

fn binary_size(size: usize) -> Result<u32, AssemblerError> {
    u32::try_from(size).map_err(|_| AssemblerError::BinaryTooLarge)
}

impl Default for WbinWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_wbin_magic() {
        let writer = WbinWriter::new();
        let mut output = Vec::new();
        writer.finish(&mut output).unwrap();

        assert_eq!(&output[0..4], b"WAVE");
        assert_eq!(u16::from_le_bytes([output[4], output[5]]), WBIN_VERSION);
    }

    #[test]
    fn test_output_single_kernel() {
        let mut writer = WbinWriter::new();
        writer.begin_kernel("test_kernel".into()).unwrap();
        writer.set_register_count(16);
        writer.set_local_memory_size(1024);
        writer.set_workgroup_size(256, 1, 1);

        let inst = EncodedInstruction::single(0x0042_0821);
        writer.write_instruction(&inst);

        writer.end_kernel().unwrap();

        let mut output = Vec::new();
        writer.finish(&mut output).unwrap();

        assert!(output.len() >= 0x20);
    }

    #[test]
    fn test_output_instruction_encoding() {
        let mut writer = WbinWriter::new();
        writer.begin_kernel("test".into()).unwrap();

        let inst = EncodedInstruction::extended(0xAAAA_BBBB, 0xCCCC_DDDD);
        writer.write_instruction(&inst);

        writer.end_kernel().unwrap();

        let code = writer.code_bytes();
        assert_eq!(code.len(), 8);

        let word0 = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(word0, 0xAAAA_BBBB);

        let word1 = u32::from_le_bytes([code[4], code[5], code[6], code[7]]);
        assert_eq!(word1, 0xCCCC_DDDD);
    }

    #[test]
    fn test_output_multiple_kernels() {
        let mut writer = WbinWriter::new();

        writer.begin_kernel("kernel1".into()).unwrap();
        writer.write_instruction(&EncodedInstruction::single(0x1111_1111));
        writer.end_kernel().unwrap();

        writer.begin_kernel("kernel2".into()).unwrap();
        writer.write_instruction(&EncodedInstruction::single(0x2222_2222));
        writer.write_instruction(&EncodedInstruction::single(0x3333_3333));
        writer.end_kernel().unwrap();

        assert_eq!(writer.kernel_count(), 2);
        assert_eq!(writer.code_bytes().len(), 12);
    }

    #[test]
    fn test_output_register_cap() {
        let mut writer = WbinWriter::new();
        writer.begin_kernel("test".into()).unwrap();
        writer.set_register_count(64);
        writer.end_kernel().unwrap();

        assert_eq!(writer.kernels[0].register_count, 64);
    }

    #[test]
    fn test_output_stripped_symbols() {
        let mut writer = WbinWriter::new();
        writer.set_strip_symbols(true);
        writer.begin_kernel("test".into()).unwrap();
        writer.write_instruction(&EncodedInstruction::single(0x1234_5678));
        writer.end_kernel().unwrap();

        let mut output = Vec::new();
        writer.finish(&mut output).unwrap();

        let symbol_offset =
            u32::from_le_bytes([output[0x10], output[0x11], output[0x12], output[0x13]]);
        assert_eq!(symbol_offset, 0);

        let symbol_size =
            u32::from_le_bytes([output[0x14], output[0x15], output[0x16], output[0x17]]);
        assert_eq!(symbol_size, 0);
    }
}
