// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Launch kernels through generated vendor host programs or the emulator library.

use crate::device::GpuVendor;
use crate::error::RuntimeError;
use crate::memory::DeviceBuffer;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Launch a compiled kernel on the specified vendor.
///
/// # Errors
///
/// Returns `RuntimeError::Launch` if the kernel cannot be launched, compiled
/// or executed on the target device.
/// Metal and emulator launches use `wbin`; other launches use `vendor_code`.
pub fn launch_kernel(
    vendor_code: &str,
    wbin: &[u8],
    vendor: GpuVendor,
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    match vendor {
        GpuVendor::Apple => launch_metal(wbin, buffers, scalars, grid, workgroup),
        GpuVendor::Nvidia => launch_cuda(vendor_code, buffers, scalars, grid, workgroup),
        GpuVendor::Amd => launch_hip(vendor_code, buffers, scalars, grid, workgroup),
        GpuVendor::Intel => launch_sycl(vendor_code, buffers, scalars, grid, workgroup),
        GpuVendor::Emulator => launch_emulator(wbin, buffers, scalars, grid, workgroup),
    }
}

fn launch_emulator(
    wbin: &[u8],
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    let packed = pack_arguments(buffers, scalars)?;
    let device_mem_size = packed
        .memory
        .len()
        .checked_add(4096)
        .ok_or_else(|| RuntimeError::Launch("device memory size exceeds host limits".into()))?
        .max(1024 * 1024);
    let initial_registers = packed
        .initial_registers
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            (
                u8::try_from(index).expect("argument count is at most 256"),
                value,
            )
        })
        .collect();
    let config = wave_emu::EmulatorConfig {
        grid_dim: grid,
        workgroup_dim: workgroup,
        use_kernel_workgroup_size: false,
        device_memory_size: device_mem_size,
        initial_registers,
        ..wave_emu::EmulatorConfig::default()
    };
    let mut emu = wave_emu::Emulator::new(config);
    emu.load_binary(wbin)?;
    emu.load_device_memory(0, &packed.memory)?;
    emu.run()?;
    packed.copy_outputs(
        emu.device_memory().read_slice(0, packed.memory.len())?,
        buffers,
    )?;

    Ok(())
}

fn write_buffer_files(
    dir: &Path,
    buffers: &[&mut DeviceBuffer],
) -> Result<Vec<String>, RuntimeError> {
    let mut paths = Vec::new();
    for (i, buf) in buffers.iter().enumerate() {
        let path = dir.join(format!("buf_{i}.bin"));
        fs::write(&path, &buf.data)?;
        paths.push(
            path.to_str()
                .ok_or_else(|| RuntimeError::Io("invalid path".into()))?
                .to_string(),
        );
    }
    Ok(paths)
}

fn read_buffer_files(dir: &Path, buffers: &mut [&mut DeviceBuffer]) -> Result<(), RuntimeError> {
    for (i, buf) in buffers.iter_mut().enumerate() {
        let path = dir.join(format!("buf_{i}.bin"));
        buf.data = fs::read(&path)?;
    }
    Ok(())
}

fn launch_metal(
    wbin: &[u8],
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    if grid.contains(&0) || workgroup.contains(&0) {
        return Err(RuntimeError::InvalidArgument(
            "launch dimensions must be positive".into(),
        ));
    }
    let binary = wave_decode::WbinFile::parse(wbin)
        .map_err(|error| RuntimeError::Backend(error.to_string()))?;
    let kernel = binary
        .kernels
        .first()
        .ok_or_else(|| RuntimeError::Launch("binary contains no kernels".into()))?;
    let mut packed = pack_arguments(buffers, scalars)?;
    let source = wave_metal::compile_with_initial_registers(wbin, &packed.initial_registers)
        .map_err(|error| RuntimeError::Backend(error.to_string()))?;
    packed.memory.resize(packed.memory.len().max(4), 0);

    let dir = tempfile::tempdir()?;
    let metal_path = dir.path().join("kernel.metal");
    let memory_path = dir.path().join("memory.bin");
    let host_path = dir.path().join("host.swift");
    let exe_path = dir.path().join("host");
    fs::write(&metal_path, source)?;
    fs::write(&memory_path, &packed.memory)?;
    fs::write(&host_path, include_str!("metal_host.swift"))?;
    run_launch_command(
        Command::new("swiftc")
            .arg("-O")
            .arg("-o")
            .arg(&exe_path)
            .arg(&host_path)
            .args(["-framework", "Metal", "-framework", "Foundation"]),
        "Swift host compilation",
    )?;
    let kernel_name = if kernel.name.is_empty() {
        "wave_kernel"
    } else {
        &kernel.name
    };
    run_launch_command(
        Command::new(&exe_path)
            .arg(&metal_path)
            .arg(&memory_path)
            .arg(kernel_name)
            .args(
                grid.into_iter()
                    .chain(workgroup)
                    .map(|value| value.to_string()),
            ),
        "Metal kernel execution",
    )?;
    let output = fs::read(&memory_path)?;
    packed.copy_outputs(&output, buffers)
}

#[derive(Debug)]
struct PackedArguments {
    memory: Vec<u8>,
    initial_registers: Vec<u32>,
    buffer_offsets: Vec<usize>,
}

impl PackedArguments {
    fn copy_outputs(
        &self,
        output: &[u8],
        buffers: &mut [&mut DeviceBuffer],
    ) -> Result<(), RuntimeError> {
        if output.len() != self.memory.len() || buffers.len() != self.buffer_offsets.len() {
            return Err(RuntimeError::Launch(
                "packed output memory layout changed".into(),
            ));
        }
        for (buffer, &offset) in buffers.iter_mut().zip(&self.buffer_offsets) {
            buffer.validate()?;
            let end = offset
                .checked_add(buffer.data.len())
                .ok_or_else(|| RuntimeError::Memory("buffer range exceeds host limits".into()))?;
            let bytes = output.get(offset..end).ok_or_else(|| {
                RuntimeError::Memory("buffer extends beyond packed output memory".into())
            })?;
            buffer.data.copy_from_slice(bytes);
        }
        Ok(())
    }
}

fn pack_arguments(
    buffers: &[&mut DeviceBuffer],
    scalars: &[u32],
) -> Result<PackedArguments, RuntimeError> {
    let argument_count = buffers
        .len()
        .checked_add(scalars.len())
        .filter(|count| *count <= 256)
        .ok_or_else(|| {
            RuntimeError::InvalidArgument("kernel arguments exceed 256 registers".into())
        })?;
    let mut initial_registers = Vec::with_capacity(argument_count);
    let mut buffer_offsets = Vec::with_capacity(buffers.len());
    let mut total = 0_u32;
    for buffer in buffers {
        buffer.validate()?;
        // WAVE memory instructions include 16-byte vector accesses.
        total = total
            .checked_add(15)
            .map(|value| value & !15)
            .ok_or_else(|| {
                RuntimeError::Memory("aligned buffers exceed the 32-bit address space".into())
            })?;
        initial_registers.push(total);
        buffer_offsets.push(
            usize::try_from(total)
                .map_err(|_| RuntimeError::Memory("buffer offset exceeds host limits".into()))?,
        );
        let size = u32::try_from(buffer.data.len())
            .map_err(|_| RuntimeError::Memory("buffer exceeds the 32-bit address space".into()))?;
        total = total.checked_add(size).ok_or_else(|| {
            RuntimeError::Memory("buffers exceed the 32-bit address space".into())
        })?;
    }
    initial_registers.extend_from_slice(scalars);
    let capacity = usize::try_from(total)
        .map_err(|_| RuntimeError::Memory("packed memory exceeds host limits".into()))?;
    let mut memory = Vec::with_capacity(capacity);
    for (buffer, &offset) in buffers.iter().zip(&buffer_offsets) {
        memory.resize(offset, 0);
        memory.extend_from_slice(&buffer.data);
    }
    Ok(PackedArguments {
        memory,
        initial_registers,
        buffer_offsets,
    })
}

fn run_launch_command(command: &mut Command, operation: &str) -> Result<(), RuntimeError> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(RuntimeError::Launch(format!(
            "{operation} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim(),
        )));
    }
    Ok(())
}

fn launch_cuda(
    vendor_code: &str,
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    let dir = tempfile::tempdir().map_err(|e| RuntimeError::Io(e.to_string()))?;

    let host_src = generate_cuda_host(vendor_code, &[], scalars, grid, workgroup, buffers);
    let cu_path = dir.path().join("kernel.cu");
    fs::write(&cu_path, &host_src)?;

    let buf_paths = write_buffer_files(dir.path(), buffers)?;

    let exe_path = dir.path().join("kernel");
    let status = Command::new("nvcc")
        .arg("-o")
        .arg(&exe_path)
        .arg(&cu_path)
        .status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("CUDA compilation failed".into()));
    }

    let status = Command::new(&exe_path).args(&buf_paths).status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("CUDA kernel execution failed".into()));
    }

    read_buffer_files(dir.path(), buffers)?;
    Ok(())
}

fn launch_hip(
    vendor_code: &str,
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    let dir = tempfile::tempdir().map_err(|e| RuntimeError::Io(e.to_string()))?;

    let host_src = generate_hip_host(vendor_code, &[], scalars, grid, workgroup, buffers);
    let hip_path = dir.path().join("kernel.hip");
    fs::write(&hip_path, &host_src)?;

    let buf_paths = write_buffer_files(dir.path(), buffers)?;

    let exe_path = dir.path().join("kernel");
    let status = Command::new("hipcc")
        .arg("-o")
        .arg(&exe_path)
        .arg(&hip_path)
        .status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("HIP compilation failed".into()));
    }

    let status = Command::new(&exe_path).args(&buf_paths).status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("HIP kernel execution failed".into()));
    }

    read_buffer_files(dir.path(), buffers)?;
    Ok(())
}

fn launch_sycl(
    vendor_code: &str,
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    let dir = tempfile::tempdir().map_err(|e| RuntimeError::Io(e.to_string()))?;

    let host_src = generate_sycl_host(vendor_code, &[], scalars, grid, workgroup, buffers);
    let cpp_path = dir.path().join("kernel.cpp");
    fs::write(&cpp_path, &host_src)?;

    let buf_paths = write_buffer_files(dir.path(), buffers)?;

    let exe_path = dir.path().join("kernel");
    let status = Command::new("icpx")
        .arg("-fsycl")
        .arg("-o")
        .arg(&exe_path)
        .arg(&cpp_path)
        .status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("SYCL compilation failed".into()));
    }

    let status = Command::new(&exe_path).args(&buf_paths).status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("SYCL kernel execution failed".into()));
    }

    read_buffer_files(dir.path(), buffers)?;
    Ok(())
}

fn generate_cuda_host(
    kernel_code: &str,
    _buf_paths: &[String],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
    buffers: &[&mut DeviceBuffer],
) -> String {
    let mut src = String::from("#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n\n");
    src.push_str(kernel_code);
    src.push_str("\n\nint main(int argc, char** argv) {\n");

    for (i, buf) in buffers.iter().enumerate() {
        writeln!(src, "    float* d_buf{i};").expect("writing to String cannot fail");
        writeln!(src, "    cudaMalloc(&d_buf{i}, {});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(
            src,
            "    FILE* f{i} = fopen(argv[{idx}], \"rb\");",
            idx = i + 1
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    float* h{i} = (float*)malloc({});",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
        writeln!(src, "    fread(h{i}, 1, {}, f{i});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(src, "    fclose(f{i});").expect("writing to String cannot fail");
        writeln!(
            src,
            "    cudaMemcpy(d_buf{i}, h{i}, {}, cudaMemcpyHostToDevice);",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
    }

    let scalar_args: Vec<String> = scalars.iter().map(|s| format!("{s}")).collect();
    let buf_args: Vec<String> = (0..buffers.len()).map(|i| format!("d_buf{i}")).collect();
    let mut all_args = buf_args;
    all_args.extend(scalar_args);

    writeln!(src, "    dim3 grid({}, {}, {});", grid[0], grid[1], grid[2])
        .expect("writing to String cannot fail");
    writeln!(
        src,
        "    dim3 block({}, {}, {});",
        workgroup[0], workgroup[1], workgroup[2]
    )
    .expect("writing to String cannot fail");

    writeln!(
        src,
        "    vector_add<<<grid, block>>>({});",
        all_args.join(", ")
    )
    .expect("writing to String cannot fail");
    src.push_str("    cudaDeviceSynchronize();\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        writeln!(
            src,
            "    cudaMemcpy(h{i}, d_buf{i}, {}, cudaMemcpyDeviceToHost);",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    FILE* o{i} = fopen(argv[{idx}], \"wb\");",
            idx = i + 1
        )
        .expect("writing to String cannot fail");
        writeln!(src, "    fwrite(h{i}, 1, {}, o{i});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(src, "    fclose(o{i});").expect("writing to String cannot fail");
        writeln!(src, "    cudaFree(d_buf{i});").expect("writing to String cannot fail");
        writeln!(src, "    free(h{i});").expect("writing to String cannot fail");
    }

    src.push_str("    return 0;\n}\n");
    src
}

fn generate_hip_host(
    kernel_code: &str,
    _buf_paths: &[String],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
    buffers: &[&mut DeviceBuffer],
) -> String {
    let mut src =
        String::from("#include <hip/hip_runtime.h>\n#include <cstdio>\n#include <cstdlib>\n\n");
    src.push_str(kernel_code);
    src.push_str("\n\nint main(int argc, char** argv) {\n");

    for (i, buf) in buffers.iter().enumerate() {
        writeln!(src, "    float* d_buf{i};").expect("writing to String cannot fail");
        writeln!(src, "    hipMalloc(&d_buf{i}, {});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(
            src,
            "    FILE* f{i} = fopen(argv[{idx}], \"rb\");",
            idx = i + 1
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    float* h{i} = (float*)malloc({});",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
        writeln!(src, "    fread(h{i}, 1, {}, f{i});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(src, "    fclose(f{i});").expect("writing to String cannot fail");
        writeln!(
            src,
            "    hipMemcpy(d_buf{i}, h{i}, {}, hipMemcpyHostToDevice);",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
    }

    let scalar_args: Vec<String> = scalars.iter().map(|s| format!("{s}")).collect();
    let buf_args: Vec<String> = (0..buffers.len()).map(|i| format!("d_buf{i}")).collect();
    let mut all_args = buf_args;
    all_args.extend(scalar_args);

    writeln!(src, "    dim3 grid({}, {}, {});", grid[0], grid[1], grid[2])
        .expect("writing to String cannot fail");
    writeln!(
        src,
        "    dim3 block({}, {}, {});",
        workgroup[0], workgroup[1], workgroup[2]
    )
    .expect("writing to String cannot fail");
    writeln!(
        src,
        "    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, {});",
        all_args.join(", ")
    )
    .expect("writing to String cannot fail");
    src.push_str("    hipDeviceSynchronize();\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        writeln!(
            src,
            "    hipMemcpy(h{i}, d_buf{i}, {}, hipMemcpyDeviceToHost);",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    FILE* o{i} = fopen(argv[{idx}], \"wb\");",
            idx = i + 1
        )
        .expect("writing to String cannot fail");
        writeln!(src, "    fwrite(h{i}, 1, {}, o{i});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(src, "    fclose(o{i});").expect("writing to String cannot fail");
        writeln!(src, "    hipFree(d_buf{i});").expect("writing to String cannot fail");
        writeln!(src, "    free(h{i});").expect("writing to String cannot fail");
    }

    src.push_str("    return 0;\n}\n");
    src
}

fn generate_sycl_host(
    kernel_code: &str,
    _buf_paths: &[String],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
    buffers: &[&mut DeviceBuffer],
) -> String {
    let mut src =
        String::from("#include <sycl/sycl.hpp>\n#include <cstdio>\n#include <cstdlib>\n\n");
    src.push_str(kernel_code);
    src.push_str("\n\nint main(int argc, char** argv) {\n");
    src.push_str("    sycl::queue q;\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        let count = buf.count;
        writeln!(
            src,
            "    float* d_buf{i} = sycl::malloc_device<float>({count}, q);"
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    FILE* f{i} = fopen(argv[{idx}], \"rb\");",
            idx = i + 1
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    float* h{i} = (float*)malloc({});",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
        writeln!(src, "    fread(h{i}, 1, {}, f{i});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(src, "    fclose(f{i});").expect("writing to String cannot fail");
        writeln!(
            src,
            "    q.memcpy(d_buf{i}, h{i}, {}).wait();",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
    }

    let total_threads = grid[0] * workgroup[0];
    writeln!(
        src,
        "    q.parallel_for(sycl::range<1>({total_threads}), [=](sycl::id<1> idx) {{"
    )
    .expect("writing to String cannot fail");
    src.push_str("        uint32_t gid = idx[0];\n");

    let scalar_vals: Vec<String> = scalars.iter().map(|s| format!("{s}")).collect();
    let buf_args: Vec<String> = (0..buffers.len()).map(|i| format!("d_buf{i}")).collect();
    let mut all_args = buf_args;
    all_args.extend(scalar_vals);

    writeln!(src, "        kernel_func({});", all_args.join(", "))
        .expect("writing to String cannot fail");
    src.push_str("    }).wait();\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        writeln!(
            src,
            "    q.memcpy(h{i}, d_buf{i}, {}).wait();",
            buf.size_bytes()
        )
        .expect("writing to String cannot fail");
        writeln!(
            src,
            "    FILE* o{i} = fopen(argv[{idx}], \"wb\");",
            idx = i + 1
        )
        .expect("writing to String cannot fail");
        writeln!(src, "    fwrite(h{i}, 1, {}, o{i});", buf.size_bytes())
            .expect("writing to String cannot fail");
        writeln!(src, "    fclose(o{i});").expect("writing to String cannot fail");
        writeln!(src, "    sycl::free(d_buf{i}, q);").expect("writing to String cannot fail");
        writeln!(src, "    free(h{i});").expect("writing to String cannot fail");
    }

    src.push_str("    return 0;\n}\n");
    src
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ElementType;

    fn mixed_width_buffers() -> [DeviceBuffer; 3] {
        [
            DeviceBuffer {
                data: 258_u16.to_le_bytes().to_vec(),
                count: 1,
                element_type: ElementType::F16,
            },
            DeviceBuffer::from_u32(&[100]),
            DeviceBuffer {
                data: 0x1234_0000_5678_0000_u64.to_le_bytes().to_vec(),
                count: 1,
                element_type: ElementType::F64,
            },
        ]
    }

    #[test]
    fn packed_buffers_align_and_copy_back_mixed_widths() -> Result<(), RuntimeError> {
        let [mut half, mut word, mut double] = mixed_width_buffers();
        let mut buffers = [&mut half, &mut word, &mut double];
        let packed = pack_arguments(&buffers, &[5])?;
        assert_eq!(packed.initial_registers, [0, 16, 32, 5]);
        assert_eq!(packed.memory.len(), 40);
        assert_eq!(packed.memory[2..16], [0; 14]);
        assert_eq!(packed.memory[20..32], [0; 12]);
        let mut output = packed.memory.clone();
        output[0..2].copy_from_slice(&263_u16.to_le_bytes());
        output[16..20].copy_from_slice(&105_u32.to_le_bytes());
        output[32..40].copy_from_slice(&0x1234_0000_5678_0005_u64.to_le_bytes());
        assert!(packed.copy_outputs(&output[..39], &mut buffers).is_err());
        packed.copy_outputs(&output, &mut buffers)?;
        assert_eq!(half.data, 263_u16.to_le_bytes());
        assert_eq!(word.to_u32()?, [105]);
        assert_eq!(double.data, 0x1234_0000_5678_0005_u64.to_le_bytes());
        Ok(())
    }

    #[test]
    fn packing_rejects_inconsistent_buffer_metadata() {
        let mut buffer = DeviceBuffer::from_f32(&[1.0]);
        buffer.count = 2;
        assert!(matches!(
            pack_arguments(&[&mut buffer], &[]),
            Err(RuntimeError::Memory(_))
        ));
        assert!(matches!(
            pack_arguments(&[], &[0; 257]),
            Err(RuntimeError::InvalidArgument(_))
        ));
    }

    #[test]
    fn emulator_updates_mixed_width_buffers() -> Result<(), Box<dyn std::error::Error>> {
        let source = r"
.kernel mixed_width
.registers 6
.workgroup_size 1, 1, 1
    device_load_u16 r4, r0
    iadd r4, r4, r3
    device_store_u16 r0, r4
    device_load_u32 r4, r1
    iadd r4, r4, r3
    device_store_u32 r1, r4
    device_load_u64 r4, r2
    iadd r4, r4, r3
    device_store_u64 r2, r4
    halt
.end
";
        let assembled = wave_asm::assemble(source, "mixed_width.wave")?;
        let [mut half, mut word, mut double] = mixed_width_buffers();
        launch_emulator(
            &assembled.binary,
            &mut [&mut half, &mut word, &mut double],
            &[5],
            [1, 1, 1],
            [1, 1, 1],
        )?;
        assert_eq!(half.data, 263_u16.to_le_bytes());
        assert_eq!(word.to_u32()?, [105]);
        assert_eq!(double.data, 0x1234_0000_5678_0005_u64.to_le_bytes());
        Ok(())
    }
}
