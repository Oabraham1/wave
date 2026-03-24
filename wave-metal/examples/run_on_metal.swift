// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

// Swift host program that loads a Metal Shading Language source file compiled by
// wave-metal, creates a Metal compute pipeline, allocates device memory with test
// data, dispatches the kernel on the GPU, and reads back results. Demonstrates
// end-to-end execution of a WAVE program on Apple Silicon via the Metal API.

import Metal
import Foundation

guard CommandLine.arguments.count >= 2 else {
    print("Usage: run_on_metal <program.metal> [workgroup_count]")
    exit(1)
}

let mslPath = CommandLine.arguments[1]
let workgroupCount = CommandLine.arguments.count >= 3
    ? Int(CommandLine.arguments[2]) ?? 1
    : 1

guard let device = MTLCreateSystemDefaultDevice() else {
    print("Error: no Metal device available")
    exit(1)
}

guard let queue = device.makeCommandQueue() else {
    print("Error: cannot create command queue")
    exit(1)
}

let mslSource: String
do {
    mslSource = try String(contentsOfFile: mslPath, encoding: .utf8)
} catch {
    print("Error: cannot read '\(mslPath)': \(error)")
    exit(1)
}

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: mslSource, options: nil)
} catch {
    print("Error: Metal compilation failed: \(error)")
    exit(1)
}

let functionNames = library.functionNames
guard let firstFunction = functionNames.first else {
    print("Error: no kernel functions found in MSL source")
    exit(1)
}

guard let function = library.makeFunction(name: firstFunction) else {
    print("Error: cannot load function '\(firstFunction)'")
    exit(1)
}

let pipeline: MTLComputePipelineState
do {
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    print("Error: cannot create pipeline: \(error)")
    exit(1)
}

let elementCount = 256 * workgroupCount
let bufferSize = elementCount * MemoryLayout<Float>.size * 3
guard let deviceMem = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
    print("Error: cannot allocate device buffer")
    exit(1)
}

let ptr = deviceMem.contents().bindMemory(to: Float.self, capacity: elementCount * 3)
for i in 0..<elementCount {
    ptr[i] = Float(i)
    ptr[elementCount + i] = Float(i) * 2.0
    ptr[elementCount * 2 + i] = 0.0
}

guard let commandBuffer = queue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else {
    print("Error: cannot create command encoder")
    exit(1)
}

encoder.setComputePipelineState(pipeline)
encoder.setBuffer(deviceMem, offset: 0, index: 0)
encoder.setThreadgroupMemoryLength(16384, index: 0)
encoder.dispatchThreadgroups(
    MTLSize(width: workgroupCount, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

if let error = commandBuffer.error {
    print("Error: GPU execution failed: \(error)")
    exit(1)
}

let output = deviceMem.contents()
    .advanced(by: elementCount * 2 * MemoryLayout<Float>.size)
    .bindMemory(to: Float.self, capacity: elementCount)

print("Results (first 16 elements):")
for i in 0..<min(16, elementCount) {
    print("  c[\(i)] = \(output[i])")
}
