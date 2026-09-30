// SPDX-License-Identifier: Apache-2.0
// Measure one generated GEMM configuration and validate the measured output.

import Foundation

@main
struct TuneBenchmark {
    static func main() {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard args.count >= 8, args.count <= 12 else {
                throw BenchmarkFailure(description: "usage: tune_benchmark METAL ENTRY TILE_M TILE_N WG_X WG_Y HAS_BIAS SIZE [WARMUP ITERATIONS SEED TILE_K]")
            }
            let tileM = try positiveInt(args[2], name: "tile_m")
            let tileN = try positiveInt(args[3], name: "tile_n")
            let groupX = try positiveInt(args[4], name: "wg_x")
            let groupY = try positiveInt(args[5], name: "wg_y")
            guard args[6] == "0" || args[6] == "1" else {
                throw BenchmarkFailure(description: "has_bias must be 0 or 1")
            }
            let hasBias = args[6] == "1"
            let entry = args[1]
            let epilogue = entry.contains("gelu") ? "bias_sigmoid_gelu" : entry.contains("relu") ? "bias_relu" : "none"
            guard hasBias == (epilogue != "none") else {
                throw BenchmarkFailure(description: "has_bias disagrees with the kernel entry")
            }
            let size = try positiveInt(args[7], name: "size")
            var measurement = MeasurementConfig()
            if args.count > 8 { measurement.warmup = try positiveInt(args[8], name: "warmup", allowZero: true) }
            if args.count > 9 { measurement.iterations = try positiveInt(args[9], name: "iterations") }
            if args.count > 10 {
                guard let seed = UInt64(args[10]) else { throw BenchmarkFailure(description: "invalid seed") }
                measurement.seed = seed
            }
            let tileK = args.count > 11 ? try positiveInt(args[11], name: "tile_k") : 8
            let kernel = KernelConfig(name: entry, path: args[0], entry: entry,
                                      tileM: tileM, tileN: tileN, tileK: tileK,
                                      groupX: groupX, groupY: groupY,
                                      isF16: entry.contains("f16"), epilogue: epilogue)
            let bench = try MetalBench(measurement: measurement)
            let (pipeline, sourceHash, compileSeconds) = try bench.compile(kernel)
            let record = try bench.runWave(kernel, size: size, pipeline: pipeline,
                                           sourceHash: sourceHash, compileSeconds: compileSeconds)
            try writeRecord(record)
            if record["correct"] as? Bool != true { exit(1) }
        } catch {
            do { try writeRecord(failureRecord(error)) }
            catch { fputs("cannot write benchmark error: \(error)\n", stderr) }
            exit(1)
        }
    }
}
