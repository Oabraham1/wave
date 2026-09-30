// SPDX-License-Identifier: Apache-2.0
// Shared measurement and full-output validation for Metal GEMM workloads.

import Accelerate
import CryptoKit
import Foundation
import Metal
import MetalPerformanceShaders

struct BenchmarkFailure: Error, CustomStringConvertible {
    let description: String
    var samples: [Double] = []
}

struct MeasurementConfig {
    var warmup = 5
    var iterations = 20
    var seed: UInt64 = 42

    var json: [String: Any] {
        ["warmup_iterations": warmup, "timed_iterations": iterations,
         "seed": String(seed), "input_generator": "splitmix64_uniform_minus1_plus1",
         "timing": "MTLCommandBuffer.gpuEndTime-gpuStartTime",
         "sample_unit": "seconds", "dispatches_per_sample": 1,
         "fast_math_enabled": false, "math_mode": "safe",
         "math_floating_point_functions": "precise", "allocation_and_compilation_timed": false]
    }
}

struct KernelConfig {
    let name: String
    let path: String
    let entry: String
    let tileM: Int
    let tileN: Int
    let tileK: Int
    let groupX: Int
    let groupY: Int
    let isF16: Bool
    let epilogue: String

    var hasBias: Bool { epilogue != "none" }
}

struct SplitMix64 {
    var state: UInt64

    mutating func nextFloat() -> Float {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        value ^= value >> 31
        return Float(value >> 40) / 8_388_608 - 1
    }
}

func jsonData(_ record: [String: Any]) throws -> Data {
    var data = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
    data.append(10)
    return data
}

func writeRecord(_ record: [String: Any], to output: FileHandle = .standardOutput) throws {
    try output.write(contentsOf: jsonData(record))
}

func failureRecord(_ error: Error) -> [String: Any] {
    var record: [String: Any] = ["schema_version": 1, "status": "error", "correct": false,
                               "error": String(describing: error)]
    if let failure = error as? BenchmarkFailure {
        record["gpu_seconds"] = failure.samples
    }
    return record
}

func positiveInt(_ text: String, name: String, allowZero: Bool = false) throws -> Int {
    guard let value = Int(text), value >= (allowZero ? 0 : 1) else {
        throw BenchmarkFailure(description: "\(name) must be \(allowZero ? "nonnegative" : "positive"): \(text)")
    }
    return value
}

func parseSizes(_ text: String) throws -> [Int] {
    let values = try text.split(separator: ",", omittingEmptySubsequences: false).map {
        try positiveInt(String($0), name: "size")
    }
    guard Set(values).count == values.count else {
        throw BenchmarkFailure(description: "sizes must be distinct")
    }
    return values
}

func median(_ samples: [Double]) -> Double {
    let ordered = samples.sorted()
    let mid = ordered.count / 2
    return ordered.count.isMultiple(of: 2) ? (ordered[mid - 1] + ordered[mid]) / 2 : ordered[mid]
}

func referenceGemm(_ a: [Float], _ b: [Float], size: Int, epilogue: String, isF16: Bool) -> [Float] {
    var c = [Float](repeating: 0, count: size * size)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(size), Int32(size), Int32(size),
                1, a, Int32(size), b, Int32(size), 0, &c, Int32(size))
    for index in c.indices {
        if epilogue == "bias_relu" {
            c[index] = max(0, c[index] + 0.5)
        } else if epilogue == "bias_sigmoid_gelu" {
            let x = c[index] + 0.5
            // The kernel uses exp2 with this exact binary32 coefficient.
            let coefficient = Float(bitPattern: 0x401D1463)
            c[index] = x / (1 + exp2(-coefficient * x))
        }
        if isF16 { c[index] = Float(Float16(c[index])) }
    }
    return c
}

func validateOutput(_ actual: [Float], _ expected: [Float], isF16: Bool) -> [String: Any] {
    let absoluteTolerance: Float = isF16 ? 0.002 : 0.001
    let relativeTolerance: Float = isF16 ? 0.001 : 0.0001
    var maxAbsolute: Double = 0
    var maxNormalized: Double = 0
    var failed = 0
    var nonfinite = 0
    for (value, reference) in zip(actual, expected) {
        guard value.isFinite, reference.isFinite else {
            nonfinite += 1
            failed += 1
            continue
        }
        let error = abs(Double(value) - Double(reference))
        let bound = Double(absoluteTolerance) + Double(relativeTolerance) * abs(Double(reference))
        maxAbsolute = max(maxAbsolute, error)
        maxNormalized = max(maxNormalized, error / bound)
        if error > bound { failed += 1 }
    }
    return ["correct": failed == 0 && actual.count == expected.count,
            "elements_checked": actual.count, "elements_expected": expected.count,
            "failed_elements": failed, "nonfinite_elements": nonfinite,
            "max_absolute_error": Double(maxAbsolute), "max_error_over_tolerance": maxNormalized,
            "absolute_tolerance": Double(absoluteTolerance), "relative_tolerance": Double(relativeTolerance),
            "criterion": "abs(actual-reference) <= absolute_tolerance + relative_tolerance*abs(reference)",
            "reference": "Accelerate SGEMM with binary32 accumulation over stored input values",
            "reference_output_rounding": isF16 ? "binary16" : "binary32",
            "coverage": "full_output_at_measured_size"]
}

final class MetalBench {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let measurement: MeasurementConfig

    init(measurement: MeasurementConfig) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkFailure(description: "no Metal device")
        }
        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkFailure(description: "cannot create Metal command queue")
        }
        self.device = device
        self.queue = queue
        self.measurement = measurement
    }

    func baseRecord(name: String, size: Int) -> [String: Any] {
        ["schema_version": 1, "workload": name, "matrix_shape": [size, size, size],
         "bench_size": size, "timestamp_utc": ISO8601DateFormatter().string(from: Date()),
         "device": device.name, "device_registry_id": String(device.registryID),
         "unified_memory": device.hasUnifiedMemory,
         "os": ProcessInfo.processInfo.operatingSystemVersionString,
         "measurement": measurement.json]
    }

    func buffer(length: Int) throws -> MTLBuffer {
        guard let result = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw BenchmarkFailure(description: "cannot allocate \(length) bytes on \(device.name)")
        }
        return result
    }

    func checkSize(_ size: Int, elementSize: Int, hasBias: Bool) throws {
        guard size > 0, size <= 16_384 else {
            throw BenchmarkFailure(description: "matrix size must be between 1 and 16384")
        }
        let bytes = 3 * size * size * elementSize + (hasBias ? size * 4 : 0)
        guard bytes <= Int(UInt32.max), bytes <= device.maxBufferLength else {
            throw BenchmarkFailure(description: "matrix buffers exceed the address or device allocation limit")
        }
    }

    func measure(encode: (MTLCommandBuffer) throws -> Void) throws -> [Double] {
        var samples: [Double] = []
        do {
            for iteration in 0..<(measurement.warmup + measurement.iterations) {
                guard let command = queue.makeCommandBuffer() else {
                    throw BenchmarkFailure(description: "cannot create command buffer")
                }
                try encode(command)
                command.commit()
                command.waitUntilCompleted()
                guard command.status == .completed, command.error == nil else {
                    throw BenchmarkFailure(description: "GPU command failed at iteration \(iteration): \(String(describing: command.error))")
                }
                if iteration >= measurement.warmup {
                    let seconds = command.gpuEndTime - command.gpuStartTime
                    guard seconds.isFinite, seconds > 0 else {
                        throw BenchmarkFailure(description: "invalid GPU duration at iteration \(iteration): \(seconds)")
                    }
                    samples.append(seconds)
                }
            }
        } catch {
            throw BenchmarkFailure(description: String(describing: error), samples: samples)
        }
        return samples
    }

    func inputs(size: Int, isF16: Bool) -> ([Float], [Float]) {
        var random = SplitMix64(state: measurement.seed)
        func values() -> [Float] {
            (0..<(size * size)).map { _ in
                let value = random.nextFloat()
                return isF16 ? Float(Float16(value)) : value
            }
        }
        return (values(), values())
    }

    func finish(_ record: inout [String: Any], samples: [Double], actual: [Float],
                reference: [Float], size: Int, isF16: Bool) {
        let validation = validateOutput(actual, reference, isF16: isF16)
        let correct = validation["correct"] as? Bool == true
        let seconds = median(samples)
        record["gpu_seconds"] = samples
        record["median_gpu_seconds"] = seconds
        record["gflops"] = 2 * pow(Double(size), 3) / seconds / 1e9
        record["gflops_operation_count"] = "2*M*N*K; epilogue operations excluded"
        record["validation"] = validation
        record["correct"] = correct
        record["status"] = correct ? "ok" : "validation_failed"
    }

    func compile(_ kernel: KernelConfig) throws -> (MTLComputePipelineState, String, Double) {
        let data = try Data(contentsOf: URL(fileURLWithPath: kernel.path))
        guard let source = String(data: data, encoding: .utf8) else {
            throw BenchmarkFailure(description: "Metal source is not UTF-8: \(kernel.path)")
        }
        let signature = "device uint8_t* device_mem [[buffer(0)]],\n    uint3 tid"
        let marker = "float _mma_c[16] = {};"
        guard source.components(separatedBy: signature).count == 2,
              source.components(separatedBy: marker).count == 2 else {
            throw BenchmarkFailure(description: "source does not contain a unique WAVE parameter injection site")
        }
        let loads = (0..<6).map { "    r\($0) = wave_params[\($0)];" }.joined(separator: "\n")
        let patched = source.replacingOccurrences(of: signature, with:
            "device uint8_t* device_mem [[buffer(0)]],\n    constant uint32_t* wave_params [[buffer(1)]],\n    uint3 tid")
            .replacingOccurrences(of: marker, with: marker + "\n" + loads)
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let start = ProcessInfo.processInfo.systemUptime
        let library = try device.makeLibrary(source: patched, options: options)
        guard let function = library.makeFunction(name: kernel.entry) else {
            throw BenchmarkFailure(description: "kernel entry not found: \(kernel.entry)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return (pipeline, digest, elapsed)
    }

    func runWave(_ kernel: KernelConfig, size: Int, pipeline: MTLComputePipelineState,
                 sourceHash: String, compileSeconds: Double) throws -> [String: Any] {
        let elementSize = kernel.isF16 ? 2 : 4
        try checkSize(size, elementSize: elementSize, hasBias: kernel.hasBias)
        guard size.isMultiple(of: kernel.tileM), size.isMultiple(of: kernel.tileN),
              size.isMultiple(of: kernel.tileK) else {
            throw BenchmarkFailure(description: "size \(size) must be divisible by tile dimensions \(kernel.tileM), \(kernel.tileN), \(kernel.tileK)")
        }
        guard kernel.groupX <= device.maxThreadsPerThreadgroup.width,
              kernel.groupY <= device.maxThreadsPerThreadgroup.height,
              kernel.groupX * kernel.groupY <= pipeline.maxTotalThreadsPerThreadgroup else {
            throw BenchmarkFailure(description: "workgroup dimensions exceed pipeline limits")
        }
        let count = size * size
        let bOffset = count * elementSize
        let biasOffset = bOffset * 2
        let cOffset = biasOffset + (kernel.hasBias ? size * 4 : 0)
        let storage = try buffer(length: cOffset + count * elementSize)
        let (a, b) = inputs(size: size, isF16: kernel.isF16)
        for (values, offset) in [(a, 0), (b, bOffset)] {
            if kernel.isF16 {
                let ptr = storage.contents().advanced(by: offset).bindMemory(to: UInt16.self, capacity: count)
                for index in values.indices { ptr[index] = Float16(values[index]).bitPattern }
            } else {
                let ptr = storage.contents().advanced(by: offset).bindMemory(to: Float.self, capacity: count)
                for index in values.indices { ptr[index] = values[index] }
            }
        }
        if kernel.hasBias {
            let ptr = storage.contents().advanced(by: biasOffset).bindMemory(to: Float.self, capacity: size)
            for index in 0..<size { ptr[index] = 0.5 }
        }
        // Unwritten output elements remain nonfinite and fail validation.
        memset(storage.contents().advanced(by: cOffset), 0xff, count * elementSize)
        var parameters = kernel.hasBias
            ? [UInt32(0), UInt32(bOffset), UInt32(biasOffset), UInt32(cOffset), UInt32(size), UInt32(size)]
            : [UInt32(0), UInt32(bOffset), UInt32(cOffset), UInt32(size), UInt32(size), UInt32(size)]
        let parameterBuffer = try buffer(length: parameters.count * 4)
        parameters.withUnsafeMutableBytes { bytes in
            parameterBuffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
        }
        let grid = MTLSize(width: size / kernel.tileN, height: size / kernel.tileM, depth: 1)
        let group = MTLSize(width: kernel.groupX, height: kernel.groupY, depth: 1)
        let samples = try measure { command in
            guard let encoder = command.makeComputeCommandEncoder() else {
                throw BenchmarkFailure(description: "cannot create compute encoder")
            }
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(storage, offset: 0, index: 0)
            encoder.setBuffer(parameterBuffer, offset: 0, index: 1)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
            encoder.endEncoding()
        }
        let actual: [Float]
        if kernel.isF16 {
            let ptr = storage.contents().advanced(by: cOffset).bindMemory(to: UInt16.self, capacity: count)
            actual = (0..<count).map { Float(Float16(bitPattern: ptr[$0])) }
        } else {
            let ptr = storage.contents().advanced(by: cOffset).bindMemory(to: Float.self, capacity: count)
            actual = Array(UnsafeBufferPointer(start: ptr, count: count))
        }
        let reference = referenceGemm(a, b, size: size, epilogue: kernel.epilogue, isF16: kernel.isF16)
        var record = baseRecord(name: kernel.name, size: size)
        record["implementation"] = "wave_metal"
        record["entry"] = kernel.entry
        record["source_path"] = kernel.path
        record["source_sha256"] = sourceHash
        record["metal_compile_seconds"] = compileSeconds
        record["input_type"] = kernel.isF16 ? "binary16" : "binary32"
        record["accumulation_type"] = "binary32"
        record["output_type"] = kernel.isF16 ? "binary16" : "binary32"
        record["epilogue"] = kernel.epilogue
        if kernel.epilogue == "bias_sigmoid_gelu" {
            record["epilogue_formula"] = "x / (1 + exp2(-Float32(0x401D1463)*x)); x = GEMM + 0.5"
        }
        record["tile"] = [kernel.tileM, kernel.tileN, kernel.tileK]
        record["threadgroup"] = [kernel.groupX, kernel.groupY, 1]
        record["grid_threadgroups"] = [grid.width, grid.height, grid.depth]
        finish(&record, samples: samples, actual: actual, reference: reference, size: size, isF16: kernel.isF16)
        return record
    }

    func runMPS(size: Int) throws -> [String: Any] {
        try checkSize(size, elementSize: 4, hasBias: false)
        let count = size * size
        let (a, b) = inputs(size: size, isF16: false)
        let aBuffer = try buffer(length: count * 4)
        let bBuffer = try buffer(length: count * 4)
        let cBuffer = try buffer(length: count * 4)
        for (values, target) in [(a, aBuffer), (b, bBuffer)] {
            values.withUnsafeBytes { bytes in
                target.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        }
        memset(cBuffer.contents(), 0xff, count * 4)
        let descriptor = MPSMatrixDescriptor(rows: size, columns: size, rowBytes: size * 4, dataType: .float32)
        let matrixA = MPSMatrix(buffer: aBuffer, descriptor: descriptor)
        let matrixB = MPSMatrix(buffer: bBuffer, descriptor: descriptor)
        let matrixC = MPSMatrix(buffer: cBuffer, descriptor: descriptor)
        let multiply = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                               resultRows: size, resultColumns: size, interiorColumns: size,
                                               alpha: 1, beta: 0)
        let samples = try measure { command in
            multiply.encode(commandBuffer: command, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)
        }
        let pointer = cBuffer.contents().bindMemory(to: Float.self, capacity: count)
        let actual = Array(UnsafeBufferPointer(start: pointer, count: count))
        let reference = referenceGemm(a, b, size: size, epilogue: "none", isF16: false)
        var record = baseRecord(name: "mps_f32", size: size)
        record["implementation"] = "MPSMatrixMultiplication"
        record["input_type"] = "binary32"
        record["output_type"] = "binary32"
        record["epilogue"] = "none"
        record["measurement"] = measurement.json.merging(["fast_math_enabled": "controlled_by_MPS",
                                                        "math_mode": "controlled_by_MPS",
                                                        "math_floating_point_functions": "controlled_by_MPS"]) { _, new in new }
        finish(&record, samples: samples, actual: actual, reference: reference, size: size, isF16: false)
        return record
    }
}
