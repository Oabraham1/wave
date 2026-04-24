// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

//! Kernel launch and synchronization for the WAVE runtime.
//!
//! Dispatches compiled kernels to the appropriate GPU vendor or the WAVE
//! emulator. For vendor backends (Metal, CUDA, HIP, SYCL), generates a host
//! program, compiles it via subprocess, and runs it. The emulator path calls
//! `wave_emu` directly as a library.

#![allow(clippy::format_push_string)]

use crate::device::GpuVendor;
use crate::error::RuntimeError;
use crate::memory::DeviceBuffer;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Launch a compiled kernel on the specified vendor.
///
/// # Errors
///
/// Returns `RuntimeError::Launch` if the kernel cannot be launched, compiled,
/// or executed on the target device.
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
        GpuVendor::Apple => launch_metal(vendor_code, buffers, scalars, grid, workgroup),
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
    let total_buffer_bytes: usize = buffers.iter().map(|b| b.size_bytes()).sum();
    let device_mem_size = (total_buffer_bytes + 4096).max(1024 * 1024);

    let mut config = wave_emu::EmulatorConfig {
        grid_dim: grid,
        workgroup_dim: workgroup,
        device_memory_size: device_mem_size,
        ..wave_emu::EmulatorConfig::default()
    };

    let mut initial_regs: Vec<(u8, u32)> = Vec::new();
    let mut offset: u32 = 0;
    for (i, buf) in buffers.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let reg_idx = i as u8;
        initial_regs.push((reg_idx, offset));
        #[allow(clippy::cast_possible_truncation)]
        let size = buf.size_bytes() as u32;
        offset += size;
    }
    for (i, &scalar) in scalars.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let reg_idx = (buffers.len() + i) as u8;
        initial_regs.push((reg_idx, scalar));
    }
    config.initial_registers = initial_regs;

    let mut emu = wave_emu::Emulator::new(config);
    emu.load_binary(wbin)?;

    let mut mem_offset: u64 = 0;
    for buf in buffers.iter() {
        emu.load_device_memory(mem_offset, &buf.data)?;
        #[allow(clippy::cast_possible_truncation)]
        let size = buf.size_bytes() as u64;
        mem_offset += size;
    }

    emu.run()?;

    let mut read_offset: u64 = 0;
    for buf in buffers.iter_mut() {
        let size = buf.size_bytes();
        let result = emu.read_device_memory(read_offset, size)?;
        buf.data = result;
        read_offset += size as u64;
    }

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
    vendor_code: &str,
    buffers: &mut [&mut DeviceBuffer],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> Result<(), RuntimeError> {
    let dir = tempfile::tempdir().map_err(|e| RuntimeError::Io(e.to_string()))?;
    let metal_path = dir.path().join("kernel.metal");
    fs::write(&metal_path, vendor_code)?;

    let buf_paths = write_buffer_files(dir.path(), buffers)?;

    let host_src = generate_metal_host(&buf_paths, scalars, grid, workgroup);
    let host_path = dir.path().join("host.swift");
    fs::write(&host_path, &host_src)?;

    let lib_path = dir.path().join("kernel.metallib");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-o"])
        .arg(&lib_path)
        .arg(&metal_path)
        .status()?;
    if !status.success() {
        return Err(RuntimeError::Launch(
            "Metal shader compilation failed".into(),
        ));
    }

    let exe_path = dir.path().join("host");
    let status = Command::new("swiftc")
        .arg("-o")
        .arg(&exe_path)
        .arg(&host_path)
        .arg("-framework")
        .arg("Metal")
        .arg("-framework")
        .arg("Foundation")
        .status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("Swift host compilation failed".into()));
    }

    let status = Command::new(&exe_path).arg(&lib_path).status()?;
    if !status.success() {
        return Err(RuntimeError::Launch("Metal kernel execution failed".into()));
    }

    read_buffer_files(dir.path(), buffers)?;
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

fn generate_metal_host(
    buf_paths: &[String],
    scalars: &[u32],
    grid: [u32; 3],
    workgroup: [u32; 3],
) -> String {
    let mut src = String::from("import Metal\nimport Foundation\n\n");
    src.push_str("let device = MTLCreateSystemDefaultDevice()!\n");
    src.push_str("let lib = try! device.makeLibrary(filepath: CommandLine.arguments[1])\n");
    src.push_str(
        "let function = lib.functionNames.first.flatMap { lib.makeFunction(name: $0) }!\n",
    );
    src.push_str("let pipeline = try! device.makeComputePipelineState(function: function)\n");
    src.push_str("let queue = device.makeCommandQueue()!\n");
    src.push_str("let cmd = queue.makeCommandBuffer()!\n");
    src.push_str("let enc = cmd.makeComputeCommandEncoder()!\n");
    src.push_str("enc.setComputePipelineState(pipeline)\n\n");

    for (i, path) in buf_paths.iter().enumerate() {
        src.push_str(&format!(
            "let data{i} = try! Data(contentsOf: URL(fileURLWithPath: \"{path}\"))\n"
        ));
        src.push_str(&format!(
            "let buf{i} = device.makeBuffer(bytes: (data{i} as NSData).bytes, length: data{i}.count, options: .storageModeShared)!\n"
        ));
        src.push_str(&format!("enc.setBuffer(buf{i}, offset: 0, index: {i})\n"));
    }

    for (i, &s) in scalars.iter().enumerate() {
        let idx = buf_paths.len() + i;
        src.push_str(&format!("var scalar{i}: UInt32 = {s}\n"));
        src.push_str(&format!(
            "enc.setBytes(&scalar{i}, length: 4, index: {idx})\n"
        ));
    }

    src.push_str(&format!(
        "\nenc.dispatchThreadgroups(MTLSize(width: {}, height: {}, depth: {}), threadsPerThreadgroup: MTLSize(width: {}, height: {}, depth: {}))\n",
        grid[0], grid[1], grid[2], workgroup[0], workgroup[1], workgroup[2]
    ));
    src.push_str("enc.endEncoding()\ncmd.commit()\ncmd.waitUntilCompleted()\n\n");

    for (i, path) in buf_paths.iter().enumerate() {
        src.push_str(&format!(
            "let out{i} = Data(bytes: buf{i}.contents(), count: buf{i}.length)\n"
        ));
        src.push_str(&format!(
            "try! out{i}.write(to: URL(fileURLWithPath: \"{path}\"))\n"
        ));
    }

    src
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
        src.push_str(&format!("    float* d_buf{i};\n"));
        src.push_str(&format!(
            "    cudaMalloc(&d_buf{i}, {});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    FILE* f{i} = fopen(argv[{idx}], \"rb\");\n",
            idx = i + 1
        ));
        src.push_str(&format!(
            "    float* h{i} = (float*)malloc({});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    fread(h{i}, 1, {}, f{i});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!("    fclose(f{i});\n"));
        src.push_str(&format!(
            "    cudaMemcpy(d_buf{i}, h{i}, {}, cudaMemcpyHostToDevice);\n",
            buf.size_bytes()
        ));
    }

    let scalar_args: Vec<String> = scalars.iter().map(|s| format!("{s}")).collect();
    let buf_args: Vec<String> = (0..buffers.len()).map(|i| format!("d_buf{i}")).collect();
    let mut all_args = buf_args;
    all_args.extend(scalar_args);

    src.push_str(&format!(
        "    dim3 grid({}, {}, {});\n",
        grid[0], grid[1], grid[2]
    ));
    src.push_str(&format!(
        "    dim3 block({}, {}, {});\n",
        workgroup[0], workgroup[1], workgroup[2]
    ));

    src.push_str(&format!(
        "    vector_add<<<grid, block>>>({});\n",
        all_args.join(", ")
    ));
    src.push_str("    cudaDeviceSynchronize();\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        src.push_str(&format!(
            "    cudaMemcpy(h{i}, d_buf{i}, {}, cudaMemcpyDeviceToHost);\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    FILE* o{i} = fopen(argv[{idx}], \"wb\");\n",
            idx = i + 1
        ));
        src.push_str(&format!(
            "    fwrite(h{i}, 1, {}, o{i});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!("    fclose(o{i});\n"));
        src.push_str(&format!("    cudaFree(d_buf{i});\n"));
        src.push_str(&format!("    free(h{i});\n"));
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
        src.push_str(&format!("    float* d_buf{i};\n"));
        src.push_str(&format!(
            "    hipMalloc(&d_buf{i}, {});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    FILE* f{i} = fopen(argv[{idx}], \"rb\");\n",
            idx = i + 1
        ));
        src.push_str(&format!(
            "    float* h{i} = (float*)malloc({});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    fread(h{i}, 1, {}, f{i});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!("    fclose(f{i});\n"));
        src.push_str(&format!(
            "    hipMemcpy(d_buf{i}, h{i}, {}, hipMemcpyHostToDevice);\n",
            buf.size_bytes()
        ));
    }

    let scalar_args: Vec<String> = scalars.iter().map(|s| format!("{s}")).collect();
    let buf_args: Vec<String> = (0..buffers.len()).map(|i| format!("d_buf{i}")).collect();
    let mut all_args = buf_args;
    all_args.extend(scalar_args);

    src.push_str(&format!(
        "    dim3 grid({}, {}, {});\n",
        grid[0], grid[1], grid[2]
    ));
    src.push_str(&format!(
        "    dim3 block({}, {}, {});\n",
        workgroup[0], workgroup[1], workgroup[2]
    ));
    src.push_str(&format!(
        "    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, {});\n",
        all_args.join(", ")
    ));
    src.push_str("    hipDeviceSynchronize();\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        src.push_str(&format!(
            "    hipMemcpy(h{i}, d_buf{i}, {}, hipMemcpyDeviceToHost);\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    FILE* o{i} = fopen(argv[{idx}], \"wb\");\n",
            idx = i + 1
        ));
        src.push_str(&format!(
            "    fwrite(h{i}, 1, {}, o{i});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!("    fclose(o{i});\n"));
        src.push_str(&format!("    hipFree(d_buf{i});\n"));
        src.push_str(&format!("    free(h{i});\n"));
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
        src.push_str(&format!(
            "    float* d_buf{i} = sycl::malloc_device<float>({count}, q);\n"
        ));
        src.push_str(&format!(
            "    FILE* f{i} = fopen(argv[{idx}], \"rb\");\n",
            idx = i + 1
        ));
        src.push_str(&format!(
            "    float* h{i} = (float*)malloc({});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    fread(h{i}, 1, {}, f{i});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!("    fclose(f{i});\n"));
        src.push_str(&format!(
            "    q.memcpy(d_buf{i}, h{i}, {}).wait();\n",
            buf.size_bytes()
        ));
    }

    let total_threads = grid[0] * workgroup[0];
    src.push_str(&format!(
        "    q.parallel_for(sycl::range<1>({total_threads}), [=](sycl::id<1> idx) {{\n"
    ));
    src.push_str("        uint32_t gid = idx[0];\n");

    let scalar_vals: Vec<String> = scalars.iter().map(|s| format!("{s}")).collect();
    let buf_args: Vec<String> = (0..buffers.len()).map(|i| format!("d_buf{i}")).collect();
    let mut all_args = buf_args;
    all_args.extend(scalar_vals);

    src.push_str(&format!("        kernel_func({});\n", all_args.join(", ")));
    src.push_str("    }).wait();\n\n");

    for (i, buf) in buffers.iter().enumerate() {
        src.push_str(&format!(
            "    q.memcpy(h{i}, d_buf{i}, {}).wait();\n",
            buf.size_bytes()
        ));
        src.push_str(&format!(
            "    FILE* o{i} = fopen(argv[{idx}], \"wb\");\n",
            idx = i + 1
        ));
        src.push_str(&format!(
            "    fwrite(h{i}, 1, {}, o{i});\n",
            buf.size_bytes()
        ));
        src.push_str(&format!("    fclose(o{i});\n"));
        src.push_str(&format!("    sycl::free(d_buf{i}, q);\n"));
        src.push_str(&format!("    free(h{i});\n"));
    }

    src.push_str("    return 0;\n}\n");
    src
}
