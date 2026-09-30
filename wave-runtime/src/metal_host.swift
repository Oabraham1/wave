// Copyright 2026 Ojima Abraham
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Metal

enum LaunchError: Error {
    case invalid(String)
}

func launch() throws {
    let args = CommandLine.arguments
    guard args.count == 10,
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else {
        throw LaunchError.invalid("Metal device or launch arguments unavailable")
    }
    let dimensions = try args[4...9].map { value in
        guard let dimension = Int(value), dimension > 0 else {
            throw LaunchError.invalid("launch dimensions must be positive")
        }
        return dimension
    }
    let source = try String(contentsOfFile: args[1], encoding: .utf8)
    let options = MTLCompileOptions()
    if #available(macOS 15.0, *) {
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
    } else {
        options.fastMathEnabled = false
    }
    let library = try device.makeLibrary(source: source, options: options)
    guard let function = library.makeFunction(name: args[3]) else {
        throw LaunchError.invalid("kernel entry point not found")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    var threads = 1
    for axis in 0..<3 {
        let limit = [device.maxThreadsPerThreadgroup.width,
                     device.maxThreadsPerThreadgroup.height,
                     device.maxThreadsPerThreadgroup.depth][axis]
        let product = threads.multipliedReportingOverflow(by: dimensions[axis + 3])
        guard dimensions[axis + 3] <= limit, !product.overflow else {
            throw LaunchError.invalid("workgroup dimensions exceed device limits")
        }
        threads = product.partialValue
    }
    guard threads <= pipeline.maxTotalThreadsPerThreadgroup else {
        throw LaunchError.invalid("workgroup size exceeds kernel limit")
    }
    let memoryURL = URL(fileURLWithPath: args[2])
    let data = try Data(contentsOf: memoryURL)
    guard !data.isEmpty, data.count <= device.maxBufferLength,
          let buffer = data.withUnsafeBytes({ bytes in
              device.makeBuffer(bytes: bytes.baseAddress!, length: data.count,
                                options: .storageModeShared)
          }),
          let command = queue.makeCommandBuffer(),
          let encoder = command.makeComputeCommandEncoder() else {
        throw LaunchError.invalid("Metal allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.dispatchThreadgroups(
        MTLSize(width: dimensions[0], height: dimensions[1], depth: dimensions[2]),
        threadsPerThreadgroup: MTLSize(width: dimensions[3], height: dimensions[4], depth: dimensions[5])
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
        throw LaunchError.invalid(command.error?.localizedDescription ?? "GPU command failed")
    }
    try Data(bytes: buffer.contents(), count: data.count).write(to: memoryURL)
}

do {
    try launch()
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
