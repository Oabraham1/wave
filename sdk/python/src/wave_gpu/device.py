# Copyright 2026 Ojima Abraham
# SPDX-License-Identifier: Apache-2.0

"""GPU detection for the WAVE Python SDK."""

import platform
import subprocess
from typing import Tuple


class DeviceInfo:
    """Detected GPU device information."""

    def __init__(self, vendor: str, name: str) -> None:
        self.vendor: str = vendor
        self.name: str = name

    def __repr__(self) -> str:
        return self.name


def detect_gpu() -> Tuple[str, str]:
    """Detect the best available GPU.

    Returns a (vendor, name) tuple. Vendor is one of: 'apple', 'nvidia',
    'amd', 'intel', 'emulator'.
    """
    if platform.system() == "Darwin":
        return ("apple", "Apple GPU (Metal)")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
        )
        name = result.stdout.strip().split("\n")[0]
        return ("nvidia", f"{name} (CUDA)")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, check=True, text=True
        )
        if "gfx" in result.stdout:
            for line in result.stdout.splitlines():
                if "Marketing Name" in line:
                    name = line.split(":", 1)[1].strip()
                    return ("amd", f"{name} (ROCm)")
            return ("amd", "AMD GPU (ROCm)")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        result = subprocess.run(["sycl-ls"], capture_output=True, check=True, text=True)
        if "level_zero:gpu" in result.stdout or "opencl:gpu" in result.stdout:
            return ("intel", "Intel GPU (SYCL)")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return ("emulator", "WAVE Emulator (no GPU)")


def device() -> DeviceInfo:
    """Detect and return the best available GPU device."""
    vendor, name = detect_gpu()
    return DeviceInfo(vendor, name)
