# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""Full pipeline orchestration for the WAVE Python SDK.

Provides in-process compilation, backend translation, and kernel execution
by loading the wave-runtime shared library via ctypes. Falls back to
subprocess invocation if the shared library is not available. The in-process
path keeps all data in memory (no temp files, no process spawning) and
includes a kernel cache that makes repeated compilations near-instant.
"""

import ctypes
import os
import platform
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

_LANGUAGE_IDS = {
    "python": 0,
    "rust": 1,
    "cpp": 2,
    "typescript": 3,
}

_VENDOR_IDS = {
    "apple": 0,
    "nvidia": 1,
    "amd": 2,
    "intel": 3,
    "emulator": 4,
}

_lib: Optional[ctypes.CDLL] = None
_lib_load_attempted = False


def _find_shared_lib() -> Optional[str]:
    """Find the wave-runtime shared library."""
    if platform.system() == "Darwin":
        lib_name = "libwave_runtime.dylib"
    elif platform.system() == "Linux":
        lib_name = "libwave_runtime.so"
    else:
        lib_name = "wave_runtime.dll"

    repo_root = Path(__file__).resolve().parents[4]

    for search_dir in [
        repo_root / "target" / "release",
        repo_root / "target" / "debug",
        repo_root / "wave-runtime" / "target" / "release",
        repo_root / "wave-runtime" / "target" / "debug",
    ]:
        candidate = search_dir / lib_name
        if candidate.exists():
            return str(candidate)

    return None


def _load_lib() -> Optional[ctypes.CDLL]:
    """Load the wave-runtime shared library, returning None if unavailable."""
    global _lib, _lib_load_attempted
    if _lib_load_attempted:
        return _lib
    _lib_load_attempted = True

    path = _find_shared_lib()
    if path is None:
        return None

    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None

    lib.wave_compile.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.wave_compile.restype = ctypes.c_int32

    lib.wave_translate.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.wave_translate.restype = ctypes.c_int32

    lib.wave_emulate.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.wave_emulate.restype = ctypes.c_int32

    lib.wave_last_error.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    lib.wave_last_error.restype = ctypes.c_char_p

    lib.wave_free.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.wave_free.restype = None

    lib.wave_cache_clear.argtypes = []
    lib.wave_cache_clear.restype = None

    lib.wave_cache_size.argtypes = []
    lib.wave_cache_size.restype = ctypes.c_size_t

    _lib = lib
    return lib


def _get_error(lib: ctypes.CDLL) -> str:
    """Retrieve the last error message from the runtime."""
    err_len = ctypes.c_size_t(0)
    err_ptr = lib.wave_last_error(ctypes.byref(err_len))
    if err_ptr and err_len.value > 0:
        return ctypes.string_at(err_ptr, err_len.value).decode("utf-8", errors="replace")
    return "unknown error"


def _compile_inprocess(source: str, language: str) -> bytes:
    """Compile kernel source in-process via the shared library."""
    lib = _load_lib()
    if lib is None:
        return _compile_subprocess(source, language)

    lang_id = _LANGUAGE_IDS.get(language)
    if lang_id is None:
        raise ValueError(f"unsupported language: {language}")

    src_bytes = source.encode("utf-8")
    out_ptr = ctypes.c_char_p()
    out_len = ctypes.c_size_t(0)

    rc = lib.wave_compile(
        src_bytes,
        len(src_bytes),
        lang_id,
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
    )

    if rc != 0:
        raise RuntimeError(f"Compilation failed: {_get_error(lib)}")

    result = ctypes.string_at(out_ptr, out_len.value)
    lib.wave_free(out_ptr, out_len.value)
    return result


def _translate_inprocess(wbin: bytes, vendor: str) -> str:
    """Translate wbin to vendor source in-process via the shared library."""
    lib = _load_lib()
    if lib is None:
        return _translate_subprocess(wbin, vendor)

    vendor_id = _VENDOR_IDS.get(vendor)
    if vendor_id is None:
        raise ValueError(f"unsupported vendor: {vendor}")

    out_ptr = ctypes.c_char_p()
    out_len = ctypes.c_size_t(0)

    rc = lib.wave_translate(
        wbin,
        len(wbin),
        vendor_id,
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
    )

    if rc != 0:
        raise RuntimeError(f"Backend translation failed: {_get_error(lib)}")

    result = ctypes.string_at(out_ptr, out_len.value).decode("utf-8")
    lib.wave_free(out_ptr, out_len.value)
    return result


def _emulate_inprocess(
    wbin: bytes,
    buffers: List[WaveArray],
    scalars: List[int],
    grid: Tuple[int, int, int],
    workgroup: Tuple[int, int, int],
) -> None:
    """Run a kernel on the emulator in-process via the shared library."""
    lib = _load_lib()
    if lib is None:
        _emulate_subprocess(wbin, buffers, scalars, grid, workgroup)
        return

    offsets: List[int] = []
    offset = 0
    mem_data = bytearray()
    for buf in buffers:
        offsets.append(offset)
        for val in buf.data:
            mem_data.extend(struct.pack("<f", val))
        offset += len(buf.data) * 4

    mem_size = max(len(mem_data), 1024 * 1024)
    mem_data.extend(b"\x00" * (mem_size - len(mem_data)))
    mem_buf = (ctypes.c_uint8 * mem_size).from_buffer(mem_data)

    reg_count = len(buffers) + len(scalars)
    RegPair = ctypes.c_uint32 * 2
    regs_array = (RegPair * reg_count)()
    for i, off in enumerate(offsets):
        regs_array[i][0] = i
        regs_array[i][1] = off
    for i, scalar in enumerate(scalars):
        regs_array[len(buffers) + i][0] = len(buffers) + i
        regs_array[len(buffers) + i][1] = scalar

    GridType = ctypes.c_uint32 * 3
    grid_arr = GridType(grid[0], grid[1], grid[2])
    wg_arr = GridType(workgroup[0], workgroup[1], workgroup[2])

    rc = lib.wave_emulate(
        wbin,
        len(wbin),
        ctypes.cast(mem_buf, ctypes.c_char_p),
        mem_size,
        ctypes.cast(regs_array, ctypes.c_void_p),
        reg_count,
        ctypes.cast(grid_arr, ctypes.c_void_p),
        ctypes.cast(wg_arr, ctypes.c_void_p),
    )

    if rc != 0:
        raise RuntimeError(f"Emulator execution failed: {_get_error(lib)}")

    offset = 0
    for buf in buffers:
        size = len(buf.data) * 4
        chunk = bytes(mem_buf[offset : offset + size])
        buf.data = [
            struct.unpack("<f", chunk[j : j + 4])[0] for j in range(0, len(chunk), 4)
        ]
        offset += size


def _find_tool(name: str) -> str:
    """Find a WAVE tool binary, checking target/release/ first, then PATH."""
    repo_root = Path(__file__).resolve().parents[4]
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


def _compile_subprocess(source: str, language: str = "python") -> bytes:
    """Compile kernel source via subprocess (fallback)."""
    compiler = _find_tool("wave-compiler")

    ext_map = {"python": ".py", "rust": ".rs", "cpp": ".cpp", "typescript": ".ts"}
    ext = ext_map.get(language, ".py")

    with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
        f.write(source)
        src_path = f.name

    wbin_path = src_path + ".wbin"

    try:
        result = subprocess.run(
            [compiler, src_path, "--output", wbin_path, "--lang", language],
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


def _translate_subprocess(wbin: bytes, vendor: str) -> str:
    """Translate wbin via subprocess (fallback)."""
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


def _emulate_subprocess(
    wbin: bytes,
    buffers: List[WaveArray],
    scalars: List[int],
    grid: Tuple[int, int, int],
    workgroup: Tuple[int, int, int],
) -> None:
    """Run a kernel on the emulator via subprocess (fallback)."""
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


def compile_kernel(source: str, language: str = "python") -> bytes:
    """Compile kernel source to WAVE binary (.wbin) bytes."""
    return _compile_inprocess(source, language)


def translate_wbin(wbin: bytes, vendor: str) -> str:
    """Translate .wbin to vendor-specific source code."""
    if vendor == "emulator":
        raise ValueError("Emulator does not need backend translation")
    return _translate_inprocess(wbin, vendor)


def launch_emulator(
    wbin: bytes,
    buffers: List[WaveArray],
    scalars: List[int],
    grid: Tuple[int, int, int],
    workgroup: Tuple[int, int, int],
) -> None:
    """Launch a kernel on the WAVE emulator."""
    _emulate_inprocess(wbin, buffers, scalars, grid, workgroup)


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
