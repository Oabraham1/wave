# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import re
import struct
import subprocess
import tempfile
import numpy as np

WAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PTX_BACKEND = os.path.join(WAVE_ROOT, "wave-ptx", "target", "release", "wave-ptx")

_ptx_cache = {}


def wbin_to_ptx(wbin_path):
    if wbin_path in _ptx_cache:
        return _ptx_cache[wbin_path]

    result = subprocess.run(
        [PTX_BACKEND, wbin_path, "-o", "/dev/stdout"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as f:
            tmp_ptx = f.name
        result = subprocess.run(
            [PTX_BACKEND, wbin_path, "-o", tmp_ptx],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"wave-ptx failed: {result.stderr}")
        with open(tmp_ptx) as f:
            ptx = f.read()
        os.unlink(tmp_ptx)
    else:
        ptx = result.stdout

    _ptx_cache[wbin_path] = ptx
    return ptx


def patch_ptx_registers(ptx, reg_values):
    lines = ptx.split("\n")
    insert_idx = None
    for i, line in enumerate(lines):
        if "ld.param.u64" in line and "_device_mem_ptr" in line:
            insert_idx = i + 1
            break

    if insert_idx is None:
        for i, line in enumerate(lines):
            if line.strip().startswith(".reg"):
                insert_idx = i + 1

    if insert_idx is None:
        raise RuntimeError("Could not find insertion point in PTX")

    while insert_idx < len(lines) and lines[insert_idx].strip().startswith(".reg"):
        insert_idx += 1

    init_lines = []
    for reg_idx in sorted(reg_values.keys()):
        val = reg_values[reg_idx]
        init_lines.append(f"    mov.b32 %r{reg_idx}, {val};")

    lines.insert(insert_idx, "\n".join(init_lines))
    return "\n".join(lines)


def get_kernel_name(ptx):
    match = re.search(r'\.visible\s+\.entry\s+(\w+)\s*\(', ptx)
    if match:
        return match.group(1)
    raise RuntimeError("Could not find kernel name in PTX")


def run_kernel_cuda(wbin_path, buffers, scalars, output_specs, grid=(1, 1, 1), workgroup=None):
    import pycuda.driver as cuda
    import pycuda.autoinit

    total_elements = max((buf.size for buf in buffers), default=1)
    if workgroup is None:
        wg_x = min(256, total_elements)
        workgroup = (wg_x, 1, 1)

    ptx = wbin_to_ptx(wbin_path)

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

    patched_ptx = patch_ptx_registers(ptx, reg_values)
    kernel_name = get_kernel_name(patched_ptx)

    mod = cuda.module_from_buffer(patched_ptx.encode())
    func = mod.get_function(kernel_name)

    host_buf = bytearray()
    for part in parts:
        host_buf.extend(part)

    total_bytes = len(host_buf)
    host_array = np.array(bytearray(host_buf), dtype=np.uint8)

    d_mem = cuda.mem_alloc(max(total_bytes, 4))
    cuda.memcpy_htod(d_mem, host_array)

    func(d_mem, block=workgroup, grid=grid)

    cuda.memcpy_dtoh(host_array, d_mem)
    device_mem = np.array(host_array)

    outputs = []
    for buf_idx, count in output_specs:
        start_byte = buf_offsets[buf_idx]
        end_byte = start_byte + count * 4
        out_data = device_mem[start_byte:end_byte].view(np.float32).copy()
        outputs.append(out_data)

    d_mem.free()
    return outputs
