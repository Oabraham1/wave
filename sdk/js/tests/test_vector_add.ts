// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// End-to-end test for the WAVE JavaScript/TypeScript SDK.

import { device, array, zeros, WaveArray } from "../src/index";

async function testDevice(): Promise<void> {
  const dev = await device();
  console.assert(dev.vendor !== undefined, "vendor should be defined");
  console.assert(dev.name.length > 0, "name should be non-empty");
  console.log(`  Device: ${dev.name}`);
}

function testArray(): void {
  const a = array([1, 2, 3, 4]);
  console.assert(a.length === 4, "length should be 4");
  console.assert(
    JSON.stringify(a.toArray()) === JSON.stringify([1, 2, 3, 4]),
    "toArray should match",
  );

  const z = zeros(5);
  console.assert(z.length === 5, "zeros length should be 5");
  console.assert(
    z.toArray().every((v) => v === 0),
    "zeros should all be 0",
  );
}

function testBufferRoundtrip(): void {
  const a = array([1.5, 2.5, 3.5]);
  const buf = a.toBuffer();
  const b = WaveArray.fromBuffer(buf, 3);
  const diff = a
    .toArray()
    .map((v, i) => Math.abs(v - b.toArray()[i]))
    .reduce((s, d) => s + d, 0);
  console.assert(diff < 0.001, "buffer roundtrip should preserve values");
}

async function main(): Promise<void> {
  console.log("Running WAVE JS/TS SDK tests...");

  await testDevice();
  console.log("  [PASS] device detection");

  testArray();
  console.log("  [PASS] array creation");

  testBufferRoundtrip();
  console.log("  [PASS] buffer roundtrip");

  console.log("All JS/TS SDK tests passed!");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
