# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Full pipeline orchestration for the WAVE Python SDK.

Handles compilation, backend translation, and kernel launch by calling
the WAVE tool chain via subprocess.
"""

import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .array import WaveArray
from .device import detect_gpu

_VENDOR_TO_LANG = {
    "apple": "metal",
    "nvidia": "ptx",
    "amd": "hip",
    "intel": "sycl",
}


def _find_tool(name: str) -> str:
    """Find a WAVE tool binary, checking target/release/ first, then PATH."""
    repo_root = Path(__file__).resolve().parents[5]
    release_path = repo_root / "target" / "release" / name
    if release_path.exists():
        return str(release_path)

    debug_path = repo_root / "target" / "debug" / name
    if debug_path.exists():
        return str(debug_path)

    for crate_dir in repo_root.iterdir():
        if crate_dir.name.startswith("wave-") and crate_dir.is_dir():
            candidate = crate_dir / "target" / "release" / name
            if candidate.exists():
                return str(candidate)
            candidate = crate_dir / "target" / "debug" / name
            if candidate.exists():
                return str(candidate)

    return name


def compile_kernel(source: str, language: str = "python") -> bytes:
    """Compile kernel source to WAVE binary (.wbin) bytes."""
    compiler = _find_tool("wave-compiler")

    ext_map = {"python": ".py", "rust": ".rs", "cpp": ".cpp", "typescript": ".ts"}
    ext = ext_map.get(language, ".py")

    with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
        f.write(source)
        src_path = f.name

    wbin_path = src_path + ".wbin"

    try:
        result = subprocess.run(
            [compiler, src_path, "-o", wbin_path, "-l", language],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}\n{result.stdout}")
        with open(wbin_path, "rb") as f:
            return f.read()
    finally:
        for p in [src_path, wbin_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def translate_wbin(wbin: bytes, vendor: str) -> str:
    """Translate .wbin to vendor-specific source code."""
    if vendor == "emulator":
        raise ValueError("Emulator does not need backend translation")

    backend_name = f"wave-{_VENDOR_TO_LANG[vendor]}"
    backend = _find_tool(backend_name)

    with tempfile.NamedTemporaryFile(suffix=".wbin", delete=False) as f:
        f.write(wbin)
        wbin_path = f.name

    try:
        result = subprocess.run(
            [backend, wbin_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Backend translation failed:\n{result.stderr}\n{result.stdout}"
            )
        return result.stdout
    finally:
        try:
            os.unlink(wbin_path)
        except OSError:
            pass


def launch_emulator(
    wbin: bytes,
    buffers: List[WaveArray],
    scalars: List[int],
    grid: Tuple[int, int, int],
    workgroup: Tuple[int, int, int],
) -> None:
    """Launch a kernel on the WAVE emulator."""
    emulator = _find_tool("wave-emu")

    with tempfile.TemporaryDirectory() as tmpdir:
        wbin_path = os.path.join(tmpdir, "kernel.wbin")
        with open(wbin_path, "wb") as f:
            f.write(wbin)

        mem_path = os.path.join(tmpdir, "memory.bin")
        offsets: List[int] = []
        offset = 0

        mem_data = bytearray()
        for buf in buffers:
            offsets.append(offset)
            for val in buf.data:
                mem_data.extend(struct.pack("<f", val))
            offset += len(buf.data) * 4

        with open(mem_path, "wb") as f:
            f.write(mem_data)

        cmd = [
            emulator,
            wbin_path,
            "--memory-file",
            mem_path,
            "--grid",
            f"{grid[0]},{grid[1]},{grid[2]}",
            "--workgroup",
            f"{workgroup[0]},{workgroup[1]},{workgroup[2]}",
        ]

        for i, off in enumerate(offsets):
            cmd.extend(["--reg", f"{i}={off}"])
        for i, scalar in enumerate(scalars):
            cmd.extend(["--reg", f"{len(buffers) + i}={scalar}"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Emulator execution failed:\n{result.stderr}\n{result.stdout}"
            )

        with open(mem_path, "rb") as f:
            mem_data = bytearray(f.read())

        offset = 0
        for buf in buffers:
            size = len(buf.data) * 4
            chunk = mem_data[offset : offset + size]
            buf.data = [
                struct.unpack("<f", chunk[j : j + 4])[0]
                for j in range(0, len(chunk), 4)
            ]
            offset += size


class CompiledKernel:
    """A compiled WAVE kernel ready for launch."""

    def __init__(self, source: str, language: str = "python") -> None:
        self._source = source
        self._language = language
        self._wbin: Optional[bytes] = None
        self._vendor_cache: Dict[str, str] = {}

    def _ensure_compiled(self) -> bytes:
        if self._wbin is None:
            self._wbin = compile_kernel(self._source, self._language)
        return self._wbin

    def launch(
        self,
        buffers: List[WaveArray],
        scalars: List[int],
        grid: Tuple[int, int, int] = (1, 1, 1),
        workgroup: Tuple[int, int, int] = (256, 1, 1),
    ) -> None:
        """Launch the kernel with the given arguments."""
        wbin = self._ensure_compiled()
        vendor, _ = detect_gpu()

        if vendor == "emulator":
            launch_emulator(wbin, buffers, scalars, grid, workgroup)
        else:
            if vendor not in self._vendor_cache:
                self._vendor_cache[vendor] = translate_wbin(wbin, vendor)
            raise NotImplementedError(
                f"Direct {vendor} launch not yet implemented in Python SDK. "
                "Use the emulator or the Rust SDK."
            )
