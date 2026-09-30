// SPDX-License-Identifier: Apache-2.0
// Execute matched kernels and validate the complete allocation after every sample.

import Foundation
import Metal

struct CaseConfig: Decodable {
    let name: String
    let size: Int
    let groups: Int
    let threads: Int
    let output_word_offset: Int
    let output_words: Int
    let output_type: String
    let initial_path: String
    let expected_path: String
    let wave_metal_path: String
    let native_metal_path: String
    let variant_metal_paths: [String: String]?
    let variant_entry_points: [String: String]?
}

struct RunConfig: Decodable {
    let cases: [CaseConfig]
    let warmup: Int
    let repeats: Int
    let order_seed: UInt64
    let dispatches_per_sample: Int
    let variant_names: [String]?
}

struct ExperimentError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func emit(_ record: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([10]))
}

func clockSeconds() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1e9
}

func validate(_ buffer: MTLBuffer, _ expected: Data, _ item: CaseConfig) throws {
    let actual = buffer.contents().bindMemory(to: UInt32.self, capacity: expected.count / 4)
    try expected.withUnsafeBytes { bytes in
        let wanted = bytes.bindMemory(to: UInt32.self)
        for index in wanted.indices {
            if index >= item.output_word_offset && index < item.output_word_offset + item.output_words
                && item.output_type == "f32" && !Float(bitPattern: actual[index]).isFinite {
                throw ExperimentError("\(item.name): non-finite output at word \(index)")
            }
            if actual[index] != wanted[index] {
                throw ExperimentError("\(item.name): word \(index) is \(actual[index]), expected \(wanted[index])")
            }
        }
    }
}

func run() throws {
    guard CommandLine.arguments.count == 2 else {
        throw ExperimentError("usage: metal_runner CONFIG.json")
    }
    let configURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let artifactDirectory = configURL.deletingLastPathComponent()
    func artifactURL(_ path: String) -> URL {
        URL(fileURLWithPath: path, relativeTo: artifactDirectory)
    }
    let config = try JSONDecoder().decode(RunConfig.self, from: Data(contentsOf: configURL))
    let variants = config.variant_names ?? ["wave", "native"]
    guard config.warmup >= 0 && config.repeats > 0 && config.dispatches_per_sample > 0 else {
        throw ExperimentError("warmup must be nonnegative; repeats and dispatches must be positive")
    }
    guard variants.count >= 2, Set(variants).count == variants.count else {
        throw ExperimentError("at least two distinct variants are required")
    }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw ExperimentError("no Metal device or command queue")
    }
    try emit([
        "kind": "device", "name": device.name, "registry_id": String(device.registryID),
        "process_id": ProcessInfo.processInfo.processIdentifier,
        "unified_memory": device.hasUnifiedMemory,
        "max_buffer_length": device.maxBufferLength,
        "recommended_working_set_bytes": device.recommendedMaxWorkingSetSize,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "math_mode": "safe", "math_functions": "precise", "metal_language": "3.1",
    ])
    var randomState = config.order_seed
    for item in config.cases {
        let initial = try Data(contentsOf: artifactURL(item.initial_path))
        let expected = try Data(contentsOf: artifactURL(item.expected_path))
        guard initial.count == expected.count, initial.count % 4 == 0,
              item.output_word_offset >= 0, item.output_words > 0,
              item.output_word_offset + item.output_words <= initial.count / 4,
              let buffer = device.makeBuffer(length: initial.count,
                  options: [.storageModeShared, .hazardTrackingModeTracked]) else {
            throw ExperimentError("invalid allocation or output range for \(item.name)")
        }
        var pipelines: [String: MTLComputePipelineState] = [:]
        for variant in variants {
            let path: String
            if let selected = item.variant_metal_paths?[variant] {
                path = selected
            } else if variant == "wave" {
                path = item.wave_metal_path
            } else if variant == "native" {
                path = item.native_metal_path
            } else {
                throw ExperimentError("missing source path for variant \(variant)")
            }
            let source = try String(contentsOf: artifactURL(path), encoding: .utf8)
            let options = MTLCompileOptions()
            guard #available(macOS 15.0, *) else {
                throw ExperimentError("the experiment requires macOS 15 or later")
            }
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
            options.languageVersion = .version3_1
            let compileStart = clockSeconds()
            let library = try device.makeLibrary(source: source, options: options)
            let entryPoint = item.variant_entry_points?[variant] ?? item.name
            guard let function = library.makeFunction(name: entryPoint) else {
                throw ExperimentError("missing function \(entryPoint) in \(variant)")
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            guard pipeline.threadExecutionWidth == 32,
                  pipeline.maxTotalThreadsPerThreadgroup >= item.threads else {
                throw ExperimentError("\(item.name) requires SIMD width 32 and \(item.threads) threads")
            }
            pipelines[variant] = pipeline
            try emit([
                "kind": "pipeline", "workload": item.name, "size": item.size, "variant": variant,
                "compile_pipeline_seconds": clockSeconds() - compileStart,
                "thread_execution_width": pipeline.threadExecutionWidth,
                "max_threads_per_group": pipeline.maxTotalThreadsPerThreadgroup,
                "static_threadgroup_memory_bytes": pipeline.staticThreadgroupMemoryLength,
            ])
        }
        for repeatIndex in (-config.warmup)..<config.repeats {
            var order = variants
            if variants.count == 2 {
                randomState = randomState &* 6364136223846793005 &+ 1442695040888963407
                if (randomState >> 63) != 0 { order.reverse() }
            } else {
                for index in stride(from: order.count - 1, through: 1, by: -1) {
                    randomState = randomState &* 6364136223846793005 &+ 1442695040888963407
                    order.swapAt(index, Int((randomState >> 32) % UInt64(index + 1)))
                }
            }
            for (orderIndex, variant) in order.enumerated() {
                initial.copyBytes(to: buffer.contents().assumingMemoryBound(to: UInt8.self), count: initial.count)
                let start = clockSeconds()
                guard let pipeline = pipelines[variant], let command = queue.makeCommandBuffer() else {
                    throw ExperimentError("cannot encode \(item.name)")
                }
                for _ in 0..<config.dispatches_per_sample {
                    guard let encoder = command.makeComputeCommandEncoder() else {
                        throw ExperimentError("cannot create compute encoder for \(item.name)")
                    }
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(buffer, offset: 0, index: 0)
                    encoder.dispatchThreadgroups(MTLSize(width: item.groups, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: item.threads, height: 1, depth: 1))
                    encoder.endEncoding()
                }
                command.commit()
                command.waitUntilCompleted()
                let wall = clockSeconds() - start
                guard command.status == .completed && command.error == nil else {
                    throw ExperimentError("\(item.name): GPU failed: \(String(describing: command.error))")
                }
                let gpu = command.gpuEndTime - command.gpuStartTime
                guard gpu.isFinite && gpu > 0 && wall.isFinite && wall > 0 else {
                    throw ExperimentError("\(item.name): invalid command timestamps")
                }
                try validate(buffer, expected, item)
                try emit([
                    "kind": "sample", "workload": item.name, "size": item.size,
                    "variant": variant, "repeat": repeatIndex, "order_in_pair": orderIndex,
                    "warmup": repeatIndex < 0,
                    "dispatches": config.dispatches_per_sample,
                    "gpu_batch_seconds": gpu, "host_batch_seconds": wall,
                    "gpu_seconds": gpu / Double(config.dispatches_per_sample),
                    "host_dispatch_seconds": wall / Double(config.dispatches_per_sample), "correct": true,
                    "gpu_status": "completed", "thermal_state": ProcessInfo.processInfo.thermalState.rawValue,
                ])
            }
        }
    }
}

do {
    try run()
} catch {
    FileHandle.standardError.write(Data("experiment failed: \(error)\n".utf8))
    exit(1)
}
