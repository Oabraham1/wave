# Copyright 2026 Ojima Abraham, Onyinye Okoli
# SPDX-License-Identifier: Apache-2.0

import os
import struct
import subprocess
import tempfile
import numpy as np

WAVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPILER = os.path.join(WAVE_ROOT, "wave-compiler", "target", "release", "wave-compiler")
EMU = os.path.join(WAVE_ROOT, "wave-emu", "target", "release", "wave-emu")

_kernel_cache = {}


def compile_kernel(name, source):
    if name in _kernel_cache:
        return _kernel_cache[name]

    cache_dir = os.path.join(WAVE_ROOT, "pytorch-integration", ".kernel_cache")
    os.makedirs(cache_dir, exist_ok=True)
    wbin_path = os.path.join(cache_dir, f"{name}.wbin")

    src_path = os.path.join(cache_dir, f"{name}.py")
    with open(src_path, "w") as f:
        f.write(source)

    result = subprocess.run(
        [COMPILER, src_path, "--output", wbin_path, "--lang", "python"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kernel compilation failed for {name}:\n{result.stderr}")

    _kernel_cache[name] = wbin_path
    return wbin_path


def _write_f32_bin(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(path)


def _read_f32_bin(path, count):
    return np.fromfile(path, dtype=np.float32, count=count)


def run_kernel(wbin_path, buffers, scalars, output_specs, grid=(1, 1, 1), workgroup=None):
    total_elements = max(
        (buf.size for buf in buffers),
        default=1
    )
    if workgroup is None:
        wg_x = min(256, total_elements)
        workgroup = (wg_x, 1, 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        offset = 0
        buf_offsets = []
        buf_paths = []

        for i, buf in enumerate(buffers):
            buf_offsets.append(offset)
            buf_path = os.path.join(tmpdir, f"buf_{i}.bin")
            _write_f32_bin(buf_path, buf)
            buf_paths.append(buf_path)
            offset += buf.size * 4

        total_mem = max(offset + 4096, 1048576)
        cmd = [
            EMU, wbin_path,
            "--grid", f"{grid[0]},{grid[1]},{grid[2]}",
            "--workgroup", f"{workgroup[0]},{workgroup[1]},{workgroup[2]}",
            "--device-memory", str(total_mem),
        ]

        for i, buf in enumerate(buffers):
            cmd.extend(["--fill-zero", f"{buf_offsets[i]}:f32:{buf.size}"])
            cmd.extend(["--arg", f"{buf_offsets[i]}:{buf_paths[i]}"])

        for i, buf_offset in enumerate(buf_offsets):
            cmd.extend(["--set-reg", f"{i}:{buf_offset}"])

        reg_idx = len(buffers)
        for val, stype in scalars:
            if stype == "u32":
                cmd.extend(["--set-reg", f"{reg_idx}:{int(val)}"])
            elif stype == "f32":
                float_bits = struct.unpack("<I", struct.pack("<f", float(val)))[0]
                cmd.extend(["--set-reg", f"{reg_idx}:{float_bits}"])
            reg_idx += 1

        dump_start = min(buf_offsets[idx] for idx, _ in output_specs)
        dump_end = max(buf_offsets[idx] + count * 4 for idx, count in output_specs)
        dump_count = (dump_end - dump_start) // 4
        cmd.extend(["--dump-f32", f"{dump_start}:{dump_count}"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"Emulator failed:\ncmd: {' '.join(cmd)}\nstderr: {result.stderr}"
            )

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        all_values = []
        for line in lines:
            try:
                all_values.append(float(line))
            except ValueError:
                break

        if len(all_values) != dump_count:
            raise RuntimeError(
                f"Expected {dump_count} values from emulator, got {len(all_values)}"
            )

        dump_array = np.array(all_values, dtype=np.float32)
        outputs = []
        for buf_idx, count in output_specs:
            start_elem = (buf_offsets[buf_idx] - dump_start) // 4
            outputs.append(dump_array[start_elem:start_elem + count].copy())

        return outputs
