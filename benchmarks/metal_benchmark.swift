// SPDX-License-Identifier: Apache-2.0
// Measure generated WAVE GEMM kernels and MPS with full-output validation.

import Foundation

@main
struct MetalBenchmark {
    static func main() {
        var output = FileHandle.standardOutput
        do {
            var metalDirectory: String?
            var sizes = [128, 256, 512, 1024, 2048, 4096]
            var measurement = MeasurementConfig()
            var outputPath: String?
            var args = ArraySlice(CommandLine.arguments.dropFirst())
            while let option = args.popFirst() {
                if option == "--help" {
                    print("usage: metal_benchmark --metal-dir DIR [--sizes N,N,...] [--warmup N] [--iterations N] [--seed N] [--output FILE]")
                    return
                }
                guard let value = args.popFirst() else {
                    throw BenchmarkFailure(description: "missing value for \(option)")
                }
                switch option {
                case "--metal-dir": metalDirectory = value
                case "--sizes": sizes = try parseSizes(value)
                case "--warmup": measurement.warmup = try positiveInt(value, name: "warmup", allowZero: true)
                case "--iterations": measurement.iterations = try positiveInt(value, name: "iterations")
                case "--seed":
                    guard let seed = UInt64(value) else { throw BenchmarkFailure(description: "invalid seed") }
                    measurement.seed = seed
                case "--output": outputPath = value
                default: throw BenchmarkFailure(description: "unknown option: \(option)")
                }
            }
            guard let directory = metalDirectory else {
                throw BenchmarkFailure(description: "--metal-dir is required")
            }
            if let path = outputPath {
                guard FileManager.default.createFile(atPath: path, contents: nil) else {
                    throw BenchmarkFailure(description: "cannot create output file: \(path)")
                }
                output = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
            }
            let definitions = [
                ("blocked_f32", "gemm_blocked", "gemm_register_blocked_8x8", false, "none"),
                ("blocked_f16", "gemm_blocked_f16", "gemm_register_blocked_8x8_f16", true, "none"),
                ("fused_relu", "gemm_bias_relu", "gemm_bias_relu_fused", false, "bias_relu"),
                ("fused_sigmoid_gelu", "gemm_bias_gelu", "gemm_bias_gelu_fused", false, "bias_sigmoid_gelu"),
                ("fused_relu_f16", "gemm_bias_relu_f16", "gemm_bias_relu_f16_fused", true, "bias_relu")
            ]
            let kernels = definitions.map { name, file, entry, isF16, epilogue in
                KernelConfig(name: name, path: URL(fileURLWithPath: directory).appendingPathComponent(file + ".metal").path,
                             entry: entry, tileM: 128, tileN: 128, tileK: 8,
                             groupX: 16, groupY: 16, isF16: isF16, epilogue: epilogue)
            }
            let bench = try MetalBench(measurement: measurement)
            var failed = false
            for kernel in kernels {
                do {
                    let (pipeline, sourceHash, compileSeconds) = try bench.compile(kernel)
                    for size in sizes {
                        do {
                            let record = try bench.runWave(kernel, size: size, pipeline: pipeline,
                                                           sourceHash: sourceHash, compileSeconds: compileSeconds)
                            try writeRecord(record, to: output)
                            if record["correct"] as? Bool != true { failed = true }
                        } catch {
                            failed = true
                            var record = bench.baseRecord(name: kernel.name, size: size)
                            record.merge(failureRecord(error)) { _, new in new }
                            try writeRecord(record, to: output)
                        }
                    }
                } catch {
                    failed = true
                    for size in sizes {
                        var record = bench.baseRecord(name: kernel.name, size: size)
                        record.merge(failureRecord(error)) { _, new in new }
                        try writeRecord(record, to: output)
                    }
                }
            }
            for size in sizes {
                do {
                    let record = try bench.runMPS(size: size)
                    try writeRecord(record, to: output)
                    if record["correct"] as? Bool != true { failed = true }
                } catch {
                    failed = true
                    var record = bench.baseRecord(name: "mps_f32", size: size)
                    record.merge(failureRecord(error)) { _, new in new }
                    try writeRecord(record, to: output)
                }
            }
            if failed { exit(1) }
        } catch {
            do { try writeRecord(failureRecord(error), to: output) }
            catch { fputs("cannot write benchmark error: \(error)\n", stderr) }
            exit(1)
        }
    }
}
