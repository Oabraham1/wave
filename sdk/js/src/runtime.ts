// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// Full pipeline orchestration for the WAVE JavaScript/TypeScript SDK.

import { execSync } from "child_process";
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import { WaveArray } from "./array";

const EXT_MAP: Record<string, string> = {
  python: ".py",
  rust: ".rs",
  cpp: ".cpp",
  typescript: ".ts",
};

/** Find a WAVE tool binary, checking target/ directories first, then PATH. */
function findTool(name: string): string {
  const repoRoot = resolve(__dirname, "..", "..", "..", "..");
  const candidates = [
    join(repoRoot, "target", "release", name),
    join(repoRoot, "target", "debug", name),
  ];

  for (const path of candidates) {
    if (existsSync(path)) {
      return path;
    }
  }

  return name;
}

/** Compile kernel source to WAVE binary (.wbin) bytes. */
export async function compileKernel(
  source: string,
  language: string,
): Promise<Buffer> {
  const compiler = findTool("wave-compiler");
  const ext = EXT_MAP[language] || ".py";

  const dir = mkdtempSync(join(tmpdir(), "wave-"));
  const srcPath = join(dir, `kernel${ext}`);
  const wbinPath = join(dir, "kernel.wbin");

  writeFileSync(srcPath, source);

  try {
    execSync(`${compiler} ${srcPath} -o ${wbinPath} -l ${language}`, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`Compilation failed: ${msg}`);
  }

  const wbin = readFileSync(wbinPath);

  try {
    unlinkSync(srcPath);
    unlinkSync(wbinPath);
  } catch {
    // cleanup best-effort
  }

  return wbin;
}

/** Launch a kernel on the WAVE emulator via subprocess. */
export async function launchEmulator(
  wbin: Buffer,
  buffers: WaveArray[],
  scalars: number[],
  grid: [number, number, number],
  workgroup: [number, number, number],
): Promise<void> {
  const emulator = findTool("wave-emu");
  const dir = mkdtempSync(join(tmpdir(), "wave-emu-"));
  const wbinPath = join(dir, "kernel.wbin");
  const memPath = join(dir, "memory.bin");

  writeFileSync(wbinPath, wbin);

  const offsets: number[] = [];
  const memParts: Buffer[] = [];
  let offset = 0;

  for (const buf of buffers) {
    offsets.push(offset);
    const binBuf = buf.toBuffer();
    memParts.push(binBuf);
    offset += binBuf.length;
  }

  writeFileSync(memPath, Buffer.concat(memParts));

  let cmd = `${emulator} ${wbinPath}`;
  cmd += ` --memory-file ${memPath}`;
  cmd += ` --grid ${grid[0]},${grid[1]},${grid[2]}`;
  cmd += ` --workgroup ${workgroup[0]},${workgroup[1]},${workgroup[2]}`;

  for (let i = 0; i < offsets.length; i++) {
    cmd += ` --reg ${i}=${offsets[i]}`;
  }
  for (let i = 0; i < scalars.length; i++) {
    cmd += ` --reg ${buffers.length + i}=${scalars[i]}`;
  }

  try {
    execSync(cmd, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`Emulator execution failed: ${msg}`);
  }

  const resultMem = readFileSync(memPath);

  let readOffset = 0;
  for (const buf of buffers) {
    const size = buf.length * 4;
    const chunk = resultMem.subarray(readOffset, readOffset + size);
    for (let i = 0; i < buf.length; i++) {
      buf.data[i] = chunk.readFloatLE(i * 4);
    }
    readOffset += size;
  }

  try {
    unlinkSync(wbinPath);
    unlinkSync(memPath);
  } catch {
    // cleanup best-effort
  }
}
