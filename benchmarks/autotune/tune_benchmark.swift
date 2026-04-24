// SPDX-License-Identifier: Apache-2.0
//
// Lightweight Metal benchmark harness for WAVE auto-tuning.
//
// Accepts a .metal source file, kernel entry point, and tuning parameters via
// CLI arguments. Runs the kernel at a single problem size with warmup and timed
// iterations, optionally checks correctness at a small size, and outputs JSON
// results. Designed to be called repeatedly by the autotune.sh orchestrator,
// once per candidate configuration.

import Metal
import Foundation

let warmup = 5
let benchIters = 10

func patchSource(_ src: String) -> String {
    var s = src.replacingOccurrences(
        of: "device uint8_t* device_mem [[buffer(0)]],\n    uint3 tid",
        with: "device uint8_t* device_mem [[buffer(0)]],\n    constant uint32_t* wave_params [[buffer(1)]],\n    uint3 tid"
    )
    let marker = "float _mma_c[16] = {};"
    var loads = "\n"
    for i in 0..<6 { loads += "    r\(i) = wave_params[\(i)];\n" }
    s = s.replacingOccurrences(of: marker, with: marker + loads)
    return s
}

func gflops(_ M: Int, _ N: Int, _ K: Int, _ sec: Double) -> Double {
    2.0 * Double(M) * Double(N) * Double(K) / (sec * 1e9)
}

func cpuGemm(_ A: [Float], _ B: [Float], _ M: Int, _ N: Int, _ K: Int) -> [Float] {
    var C = [Float](repeating: 0, count: M*N)
    for i in 0..<M {
        for k in 0..<K {
            let a = A[i*K+k]
            for j in 0..<N { C[i*N+j] += a * B[k*N+j] }
        }
    }
    return C
}

guard CommandLine.arguments.count >= 8 else {
    fputs("Usage: tune_benchmark <metal_path> <entry> <tile_m> <tile_n> <wg_x> <wg_y> <has_bias> [bench_size]\n", stderr)
    exit(1)
}

let metalPath = CommandLine.arguments[1]
let entry = CommandLine.arguments[2]
let tileM = Int(CommandLine.arguments[3])!
let tileN = Int(CommandLine.arguments[4])!
let wgX = Int(CommandLine.arguments[5])!
let wgY = Int(CommandLine.arguments[6])!
let hasBias = CommandLine.arguments[7] == "1"
let benchSize = CommandLine.arguments.count > 8 ? Int(CommandLine.arguments[8])! : 4096

guard let device = MTLCreateSystemDefaultDevice() else {
    print("{\"error\": \"no_metal_device\"}")
    exit(1)
}
let queue = device.makeCommandQueue()!

guard let data = FileManager.default.contents(atPath: metalPath),
      let src = String(data: data, encoding: .utf8) else {
    print("{\"error\": \"cannot_read_file\"}")
    exit(1)
}

let patched = patchSource(src)
let lib: MTLLibrary
do { lib = try device.makeLibrary(source: patched, options: nil) }
catch {
    print("{\"error\": \"compile_error\", \"detail\": \"\(error)\"}")
    exit(1)
}

guard let fn = lib.makeFunction(name: entry) else {
    print("{\"error\": \"function_not_found\"}")
    exit(1)
}

let pso: MTLComputePipelineState
do { pso = try device.makeComputePipelineState(function: fn) }
catch {
    print("{\"error\": \"pipeline_error\"}")
    exit(1)
}

let isF16 = entry.contains("f16")
let elemSize = isF16 ? 2 : 4

func runBenchmark(_ N: Int) -> (Double, Bool, Float) {
    let M = N, K = N
    let aOff: Int, bOff: Int, biasOff: Int, cOff: Int, total: Int

    if hasBias {
        aOff = 0; bOff = M*K*elemSize; biasOff = bOff+K*N*elemSize; cOff = biasOff+N*4
        total = cOff+M*N*elemSize
    } else {
        aOff = 0; bOff = M*K*elemSize; biasOff = 0; cOff = bOff+K*N*elemSize
        total = cOff+M*N*elemSize
    }

    guard let buf = device.makeBuffer(length: total, options: .storageModeShared) else {
        return (0, false, 999)
    }

    srand48(42)
    if isF16 {
        let pA = buf.contents().advanced(by: aOff).bindMemory(to: UInt16.self, capacity: M*K)
        let pB = buf.contents().advanced(by: bOff).bindMemory(to: UInt16.self, capacity: K*N)
        for i in 0..<M*K { pA[i] = Float16(Float(drand48()*2.0-1.0)).bitPattern }
        for i in 0..<K*N { pB[i] = Float16(Float(drand48()*2.0-1.0)).bitPattern }
    } else {
        let pA = buf.contents().advanced(by: aOff).bindMemory(to: Float.self, capacity: M*K)
        let pB = buf.contents().advanced(by: bOff).bindMemory(to: Float.self, capacity: K*N)
        for i in 0..<M*K { pA[i] = Float(drand48()*2.0-1.0) }
        for i in 0..<K*N { pB[i] = Float(drand48()*2.0-1.0) }
    }
    if hasBias {
        let pBi = buf.contents().advanced(by: biasOff).bindMemory(to: Float.self, capacity: N)
        for i in 0..<N { pBi[i] = 0.5 }
    }
    memset(buf.contents().advanced(by: cOff), 0, M*N*elemSize)

    var paramVals: [UInt32]
    if hasBias {
        paramVals = [UInt32(aOff), UInt32(bOff), UInt32(biasOff), UInt32(cOff), UInt32(N), UInt32(K)]
    } else {
        paramVals = [UInt32(aOff), UInt32(bOff), UInt32(cOff), UInt32(M), UInt32(N), UInt32(K)]
    }
    let paramBuf = device.makeBuffer(bytes: &paramVals, length: paramVals.count*4, options: .storageModeShared)!

    let grid = MTLSize(width: N/tileN, height: M/tileM, depth: 1)
    let wg = MTLSize(width: wgX, height: wgY, depth: 1)

    for _ in 0..<warmup {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(paramBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: wg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }

    var times: [Double] = []
    for _ in 0..<benchIters {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(paramBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: wg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        let t = cb.gpuEndTime - cb.gpuStartTime
        if t > 0 { times.append(t) }
    }

    guard !times.isEmpty else { return (0, false, 999) }
    times.sort()
    let median = times[times.count / 2]
    let gf = gflops(M, N, K, median)

    var correct = true
    var maxErr: Float = 0

    if N <= 512 {
        memset(buf.contents().advanced(by: cOff), 0, M*N*elemSize)
        srand48(42)
        var refA = [Float](repeating: 0, count: M*K)
        var refB = [Float](repeating: 0, count: K*N)
        if isF16 {
            for i in 0..<M*K { refA[i] = Float(Float16(Float(drand48()*2.0-1.0))) }
            for i in 0..<K*N { refB[i] = Float(Float16(Float(drand48()*2.0-1.0))) }
        } else {
            for i in 0..<M*K { refA[i] = Float(drand48()*2.0-1.0) }
            for i in 0..<K*N { refB[i] = Float(drand48()*2.0-1.0) }
        }

        srand48(42)
        if isF16 {
            let pA = buf.contents().advanced(by: aOff).bindMemory(to: UInt16.self, capacity: M*K)
            let pB = buf.contents().advanced(by: bOff).bindMemory(to: UInt16.self, capacity: K*N)
            for i in 0..<M*K { pA[i] = Float16(Float(drand48()*2.0-1.0)).bitPattern }
            for i in 0..<K*N { pB[i] = Float16(Float(drand48()*2.0-1.0)).bitPattern }
        } else {
            let pA = buf.contents().advanced(by: aOff).bindMemory(to: Float.self, capacity: M*K)
            let pB = buf.contents().advanced(by: bOff).bindMemory(to: Float.self, capacity: K*N)
            for i in 0..<M*K { pA[i] = Float(drand48()*2.0-1.0) }
            for i in 0..<K*N { pB[i] = Float(drand48()*2.0-1.0) }
        }
        if hasBias {
            let pBi = buf.contents().advanced(by: biasOff).bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { pBi[i] = 0.5 }
        }
        memset(buf.contents().advanced(by: cOff), 0, M*N*elemSize)

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(paramBuf, offset: 0, index: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: wg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()

        var refC = cpuGemm(refA, refB, M, N, K)

        if entry.contains("relu") {
            for i in 0..<M { for j in 0..<N { refC[i*N+j] = max(0, refC[i*N+j] + 0.5) } }
        } else if entry.contains("gelu") {
            for i in 0..<M {
                for j in 0..<N {
                    let x = refC[i*N+j] + 0.5
                    refC[i*N+j] = x / (1.0 + exp(-1.702 * x))
                }
            }
        }

        if isF16 {
            let refCf16 = refC.map { Float(Float16($0)) }
            let pC = buf.contents().advanced(by: cOff).bindMemory(to: UInt16.self, capacity: M*N)
            for i in 0..<M*N {
                maxErr = max(maxErr, abs(Float(Float16(bitPattern: pC[i])) - refCf16[i]))
            }
        } else {
            let pC = buf.contents().advanced(by: cOff).bindMemory(to: Float.self, capacity: M*N)
            for i in 0..<M*N { maxErr = max(maxErr, abs(pC[i] - refC[i])) }
        }

        let tol: Float = isF16 ? 0.05 : 1e-3
        correct = maxErr <= tol
    }

    return (gf, correct, maxErr)
}

let checkSize = max(tileM, tileN)
let (checkGf, correct, maxErr) = runBenchmark(checkSize)

if !correct {
    print("{\"gflops\": 0, \"correct\": false, \"max_err\": \(maxErr), \"check_size\": \(checkSize)}")
    exit(0)
}

let (gf, _, _) = runBenchmark(benchSize)

print("{\"gflops\": \(String(format: "%.1f", gf)), \"correct\": true, \"max_err\": \(String(format: "%.2e", maxErr)), \"bench_size\": \(benchSize)}")
