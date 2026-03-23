// Copyright (c) 2026 Ojima Abraham. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE file for details.

// WBIN binary writer. Produces the final output format: 32-byte header, code
// section, symbol table (optional), and kernel metadata. Supports multiple
// kernels and stripping symbols for release builds.

use crate::ast::KernelMetadata;
use crate::diagnostics::AssemblerError;
use crate::encoder::EncodedInstruction;
use std::io::Write;

const WBIN_VERSION: u16 = 0x0001;

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

    pub fn begin_kernel(&mut self, name: String) {
        let metadata = KernelMetadata {
            name,
            code_offset: self.code.len() as u32,
            ..Default::default()
        };
        self.kernels.push(metadata);
    }

    pub fn set_register_count(&mut self, count: u32) {
        if let Some(kernel) = self.kernels.last_mut() {
            kernel.register_count = count.min(32);
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

    pub fn end_kernel(&mut self) {
        if let Some(kernel) = self.kernels.last_mut() {
            kernel.code_size = self.code.len() as u32 - kernel.code_offset;
        }
    }

    pub fn write_instruction(&mut self, inst: &EncodedInstruction) {
        self.code.extend_from_slice(&inst.word0.to_le_bytes());
        if let Some(word1) = inst.word1 {
            self.code.extend_from_slice(&word1.to_le_bytes());
        }
    }

    pub fn finish<W: Write>(&self, writer: &mut W) -> Result<(), AssemblerError> {
        let header_size: u32 = 0x20;
        let code_offset = header_size;
        let code_size = self.code.len() as u32;

        let symbol_offset = if self.strip_symbols { 0 } else { code_offset + code_size };
        let symbol_size = if self.strip_symbols {
            0
        } else {
            self.kernels.iter().map(|k| k.name.len() + 1).sum::<usize>() as u32
        };

        let metadata_offset = symbol_offset + symbol_size;
        let metadata_size = self.compute_metadata_size();

        writer
            .write_all(b"WAVE")
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&WBIN_VERSION.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&0u16.to_le_bytes()) // flags (reserved)
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&code_offset.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&code_size.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&symbol_offset.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&symbol_size.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&metadata_offset.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&metadata_size.to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        writer
            .write_all(&self.code)
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        if !self.strip_symbols {
            for kernel in &self.kernels {
                writer
                    .write_all(kernel.name.as_bytes())
                    .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;
                writer
                    .write_all(&[0u8])
                    .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;
            }
        }

        writer
            .write_all(&(self.kernels.len() as u32).to_le_bytes())
            .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

        for kernel in &self.kernels {
            let name_offset = if self.strip_symbols {
                0u32
            } else {
                symbol_offset + self.get_name_offset(&kernel.name)
            };

            writer
                .write_all(&name_offset.to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.register_count.to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.local_memory_size.to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.workgroup_size[0].to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.workgroup_size[1].to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.workgroup_size[2].to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.code_offset.to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;

            writer
                .write_all(&kernel.code_size.to_le_bytes())
                .map_err(|e| AssemblerError::IoError { message: e.to_string() })?;
        }

        Ok(())
    }

    fn compute_metadata_size(&self) -> u32 {
        4 + (self.kernels.len() as u32 * 32)
    }

    fn get_name_offset(&self, target_name: &str) -> u32 {
        let mut offset = 0u32;
        for kernel in &self.kernels {
            if kernel.name == target_name {
                return offset;
            }
            offset += kernel.name.len() as u32 + 1;
        }
        0
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
        assert_eq!(
            u16::from_le_bytes([output[4], output[5]]),
            WBIN_VERSION
        );
    }

    #[test]
    fn test_output_single_kernel() {
        let mut writer = WbinWriter::new();
        writer.begin_kernel("test_kernel".into());
        writer.set_register_count(16);
        writer.set_local_memory_size(1024);
        writer.set_workgroup_size(256, 1, 1);

        let inst = EncodedInstruction::single(0x0042_0821);
        writer.write_instruction(&inst);

        writer.end_kernel();

        let mut output = Vec::new();
        writer.finish(&mut output).unwrap();

        assert!(output.len() >= 0x20);
    }

    #[test]
    fn test_output_instruction_encoding() {
        let mut writer = WbinWriter::new();
        writer.begin_kernel("test".into());

        let inst = EncodedInstruction::extended(0xAAAA_BBBB, 0xCCCC_DDDD);
        writer.write_instruction(&inst);

        writer.end_kernel();

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

        writer.begin_kernel("kernel1".into());
        writer.write_instruction(&EncodedInstruction::single(0x1111_1111));
        writer.end_kernel();

        writer.begin_kernel("kernel2".into());
        writer.write_instruction(&EncodedInstruction::single(0x2222_2222));
        writer.write_instruction(&EncodedInstruction::single(0x3333_3333));
        writer.end_kernel();

        assert_eq!(writer.kernel_count(), 2);
        assert_eq!(writer.code_bytes().len(), 12);
    }

    #[test]
    fn test_output_register_cap() {
        let mut writer = WbinWriter::new();
        writer.begin_kernel("test".into());
        writer.set_register_count(64);
        writer.end_kernel();

        assert_eq!(writer.kernels[0].register_count, 32);
    }

    #[test]
    fn test_output_stripped_symbols() {
        let mut writer = WbinWriter::new();
        writer.set_strip_symbols(true);
        writer.begin_kernel("test".into());
        writer.write_instruction(&EncodedInstruction::single(0x1234_5678));
        writer.end_kernel();

        let mut output = Vec::new();
        writer.finish(&mut output).unwrap();

        let symbol_offset = u32::from_le_bytes([output[0x10], output[0x11], output[0x12], output[0x13]]);
        assert_eq!(symbol_offset, 0);

        let symbol_size = u32::from_le_bytes([output[0x14], output[0x15], output[0x16], output[0x17]]);
        assert_eq!(symbol_size, 0);
    }
}
