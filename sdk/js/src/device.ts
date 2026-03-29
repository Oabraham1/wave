// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// GPU detection for the WAVE JavaScript/TypeScript SDK.

import { execSync } from "child_process";
import { platform } from "os";

/** GPU vendor classification. */
export type GpuVendor = "apple" | "nvidia" | "amd" | "intel" | "emulator";

/** Detected GPU device information. */
export interface DeviceInfo {
  vendor: GpuVendor;
  name: string;
}

/** Detect the best available GPU device. */
export async function device(): Promise<DeviceInfo> {
  if (platform() === "darwin") {
    return { vendor: "apple", name: "Apple GPU (Metal)" };
  }

  try {
    const out = execSync("nvidia-smi --query-gpu=name --format=csv,noheader", {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    const name = out.trim().split("\n")[0];
    return { vendor: "nvidia", name: `${name} (CUDA)` };
  } catch {
    // nvidia-smi not available
  }

  try {
    const out = execSync("rocminfo", {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    if (out.includes("gfx")) {
      return { vendor: "amd", name: "AMD GPU (ROCm)" };
    }
  } catch {
    // rocminfo not available
  }

  try {
    const out = execSync("sycl-ls", {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    if (out.includes("level_zero:gpu") || out.includes("opencl:gpu")) {
      return { vendor: "intel", name: "Intel GPU (SYCL)" };
    }
  } catch {
    // sycl-ls not available
  }

  return { vendor: "emulator", name: "WAVE Emulator (no GPU)" };
}
