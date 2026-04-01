# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import re
import struct
import subprocess
import tempfile
import numpy as np

WAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METAL_BACKEND = os.path.join(WAVE_ROOT, "wave-metal", "target", "release", "wave-metal")

_msl_cache = {}


def wbin_to_msl(wbin_path):
    if wbin_path in _msl_cache:
        return _msl_cache[wbin_path]

    result = subprocess.run(
        [METAL_BACKEND, wbin_path, "-o", "/dev/stdout"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as f:
            tmp_metal = f.name
        result = subprocess.run(
            [METAL_BACKEND, wbin_path, "-o", tmp_metal],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"wave-metal failed: {result.stderr}")
        with open(tmp_metal) as f:
            msl = f.read()
        os.unlink(tmp_metal)
    else:
        msl = result.stdout

    _msl_cache[wbin_path] = msl
    return msl


def patch_msl_registers(msl, reg_values):
    lines = msl.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("bool p0"):
            insert_idx = i + 1
            break
        if line.strip().startswith("uint wave_count"):
            insert_idx = i + 1
            break

    if insert_idx is None:
        for i, line in enumerate(lines):
            if "uint32_t r0 = 0" in line:
                insert_idx = i + 1
                while insert_idx < len(lines) and lines[insert_idx].strip().startswith("uint32_t r"):
                    insert_idx += 1
                break

    if insert_idx is None:
        raise RuntimeError("Could not find register declarations in MSL")

    init_lines = []
    for reg_idx in sorted(reg_values.keys()):
        val = reg_values[reg_idx]
        init_lines.append(f"    r{reg_idx} = {val}u;")

    lines.insert(insert_idx, "\n".join(init_lines))
    return "\n".join(lines)


def generate_swift_host(metal_lib_path, buf_bin_path, buf_total_bytes, grid, workgroup):
    src = """import Metal
import Foundation

guard CommandLine.arguments.count >= 3 else {
    fputs("Usage: host <metallib> <buffer.bin>\\n", stderr)
    exit(1)
}

let device: MTLDevice
if let d = MTLCreateSystemDefaultDevice() {
    device = d
} else {
    let allDevices = MTLCopyAllDevices()
    guard let d = allDevices.first else {
        fputs("Error: No Metal device found\\n", stderr)
        exit(1)
    }
    device = d
}

let libPath = CommandLine.arguments[1]
let binPath = CommandLine.arguments[2]

do {
    let lib = try device.makeLibrary(filepath: libPath)
    guard let fnName = lib.functionNames.first else {
        fputs("Error: No functions in metallib\\n", stderr)
        exit(1)
    }
    guard let function = lib.makeFunction(name: fnName) else {
        fputs("Error: Could not create function '\\(fnName)'\\n", stderr)
        exit(1)
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    guard let queue = device.makeCommandQueue() else {
        fputs("Error: Could not create command queue\\n", stderr)
        exit(1)
    }
    guard let cmd = queue.makeCommandBuffer() else {
        fputs("Error: Could not create command buffer\\n", stderr)
        exit(1)
    }
    guard let enc = cmd.makeComputeCommandEncoder() else {
        fputs("Error: Could not create compute encoder\\n", stderr)
        exit(1)
    }
    enc.setComputePipelineState(pipeline)

    let bufData = try Data(contentsOf: URL(fileURLWithPath: binPath))
    guard let buf = device.makeBuffer(bytes: (bufData as NSData).bytes, length: bufData.count, options: .storageModeShared) else {
        fputs("Error: Could not create Metal buffer (\\(bufData.count) bytes)\\n", stderr)
        exit(1)
    }
    enc.setBuffer(buf, offset: 0, index: 0)

"""
    src += f"    enc.dispatchThreadgroups(MTLSize(width: {grid[0]}, height: {grid[1]}, depth: {grid[2]}), threadsPerThreadgroup: MTLSize(width: {workgroup[0]}, height: {workgroup[1]}, depth: {workgroup[2]}))\n"
    src += """    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    if let error = cmd.error {
        fputs("Metal execution error: \\(error)\\n", stderr)
        exit(1)
    }

    let outData = Data(bytes: buf.contents(), count: buf.length)
    try outData.write(to: URL(fileURLWithPath: binPath))
} catch {
    fputs("Error: \\(error)\\n", stderr)
    exit(1)
}
"""
    return src


def run_kernel_metal(wbin_path, buffers, scalars, output_specs, grid=(1, 1, 1), workgroup=None):
    total_elements = max((buf.size for buf in buffers), default=1)
    if workgroup is None:
        wg_x = min(256, total_elements)
        workgroup = (wg_x, 1, 1)

    msl = wbin_to_msl(wbin_path)

    offset = 0
    buf_offsets = []
    parts = []
    for buf in buffers:
        buf_offsets.append(offset)
        flat = np.ascontiguousarray(buf.flatten(), dtype=np.float32)
        parts.append(flat.tobytes())
        offset += flat.size * 4

    reg_values = {}
    for i, bo in enumerate(buf_offsets):
        reg_values[i] = bo

    reg_idx = len(buffers)
    for val, stype in scalars:
        if stype == "u32":
            reg_values[reg_idx] = int(val)
        elif stype == "f32":
            reg_values[reg_idx] = struct.unpack("<I", struct.pack("<f", float(val)))[0]
        reg_idx += 1

    patched_msl = patch_msl_registers(msl, reg_values)

    with tempfile.TemporaryDirectory() as tmpdir:
        metal_path = os.path.join(tmpdir, "kernel.metal")
        with open(metal_path, "w") as f:
            f.write(patched_msl)

        buf_bin_path = os.path.join(tmpdir, "device_mem.bin")
        with open(buf_bin_path, "wb") as f:
            for part in parts:
                f.write(part)

        total_bytes = offset
        host_src = generate_swift_host(
            "kernel.metallib", buf_bin_path, total_bytes, grid, workgroup
        )
        host_path = os.path.join(tmpdir, "host.swift")
        with open(host_path, "w") as f:
            f.write(host_src)

        lib_path = os.path.join(tmpdir, "kernel.metallib")
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-O2", "-o", lib_path, metal_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Metal compilation failed:\n{result.stderr}")

        exe_path = os.path.join(tmpdir, "host")
        result = subprocess.run(
            ["swiftc", "-O", "-o", exe_path, host_path,
             "-framework", "Metal", "-framework", "Foundation"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Swift compilation failed:\n{result.stderr}")

        result = subprocess.run(
            [exe_path, lib_path, buf_bin_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(f"Metal execution failed:\n{result.stderr}")

        device_mem = np.fromfile(buf_bin_path, dtype=np.uint8)

        outputs = []
        for buf_idx, count in output_specs:
            start_byte = buf_offsets[buf_idx]
            end_byte = start_byte + count * 4
            out_data = device_mem[start_byte:end_byte].view(np.float32).copy()
            outputs.append(out_data)

        return outputs
