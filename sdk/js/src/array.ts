// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

/// Array types for the WAVE JavaScript/TypeScript SDK.

/** Supported element data types. */
export type DType = "f32" | "u32" | "i32" | "f16" | "f64";

/** CPU-side array that can be passed to WAVE GPU kernels. */
export class WaveArray {
  data: number[];
  dtype: DType;

  constructor(data: number[], dtype: DType = "f32") {
    this.data = [...data];
    this.dtype = dtype;
  }

  /** Return the array contents as a plain JavaScript array. */
  toArray(): number[] {
    return [...this.data];
  }

  /** Number of elements. */
  get length(): number {
    return this.data.length;
  }

  /** Convert array data to a binary Buffer (little-endian f32). */
  toBuffer(): Buffer {
    const buf = Buffer.alloc(this.data.length * 4);
    for (let i = 0; i < this.data.length; i++) {
      buf.writeFloatLE(this.data[i], i * 4);
    }
    return buf;
  }

  /** Read data from a binary Buffer (little-endian f32). */
  static fromBuffer(
    buf: Buffer,
    count: number,
    dtype: DType = "f32",
  ): WaveArray {
    const data: number[] = [];
    for (let i = 0; i < count; i++) {
      data.push(buf.readFloatLE(i * 4));
    }
    return new WaveArray(data, dtype);
  }
}

/** Create a WaveArray from a JavaScript array. */
export function array(data: number[], dtype: DType = "f32"): WaveArray {
  return new WaveArray(data, dtype);
}

/** Create a zero-filled WaveArray. */
export function zeros(n: number, dtype: DType = "f32"): WaveArray {
  return new WaveArray(new Array(n).fill(0), dtype);
}

/** Create a WaveArray filled with ones. */
export function ones(n: number, dtype: DType = "f32"): WaveArray {
  return new WaveArray(new Array(n).fill(1), dtype);
}
