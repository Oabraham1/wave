// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// Device and host-buffer checks for the WAVE JavaScript/TypeScript SDK.

import { strict as assert } from "node:assert";
import { device, array, zeros, WaveArray } from "../src/index";

async function testDevice(): Promise<void> {
  const dev = await device();
  assert.notEqual(dev.vendor, undefined);
  assert.notEqual(dev.name, "");
  console.log(`  Device: ${dev.name}`);
}

function testArray(): void {
  const a = array([1, 2, 3, 4]);
  assert.equal(a.length, 4);
  assert.deepEqual(a.toArray(), [1, 2, 3, 4]);

  const z = zeros(5);
  assert.equal(z.length, 5);
  assert.deepEqual(z.toArray(), [0, 0, 0, 0, 0]);
}

function testBufferRoundtrip(): void {
  const a = array([1.5, 2.5, 3.5]);
  const buf = a.toBuffer();
  const b = WaveArray.fromBuffer(buf, 3);
  assert.deepEqual(b.toArray(), a.toArray());
}

async function main(): Promise<void> {
  console.log("Running WAVE JS/TS SDK tests...");

  await testDevice();
  console.log("  [PASS] device detection");

  testArray();
  console.log("  [PASS] array creation");

  testBufferRoundtrip();
  console.log("  [PASS] buffer roundtrip");

  console.log("JS/TS SDK checks passed.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
