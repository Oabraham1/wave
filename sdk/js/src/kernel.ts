// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// Kernel compilation and launch for the WAVE JavaScript/TypeScript SDK.

import { DeviceInfo } from "./device";
import { WaveArray } from "./array";
import { compileKernel, launchEmulator } from "./runtime";

/** Supported source languages. */
export type Language = "python" | "rust" | "cpp" | "typescript";

/** A compiled WAVE kernel ready for launch. */
export class CompiledKernel {
  private source: string;
  private language: Language;
  private wbin: Buffer | null = null;

  constructor(source: string, language: Language = "typescript") {
    this.source = source;
    this.language = language;
  }

  private async ensureCompiled(): Promise<Buffer> {
    if (this.wbin === null) {
      this.wbin = await compileKernel(this.source, this.language);
    }
    return this.wbin;
  }

  /** Launch the kernel on the given device. */
  async launch(
    dev: DeviceInfo,
    buffers: WaveArray[],
    scalars: number[],
    grid: [number, number, number] = [1, 1, 1],
    workgroup: [number, number, number] = [256, 1, 1],
  ): Promise<void> {
    const wbin = await this.ensureCompiled();

    if (dev.vendor === "emulator") {
      await launchEmulator(wbin, buffers, scalars, grid, workgroup);
    } else {
      throw new Error(
        `Direct ${dev.vendor} launch not yet implemented in JS SDK. ` +
          "Use the emulator or the Rust SDK.",
      );
    }
  }
}

/** Create a compiled kernel from source code. */
export function kernel(
  source: string,
  language: Language = "typescript",
): CompiledKernel {
  return new CompiledKernel(source, language);
}
