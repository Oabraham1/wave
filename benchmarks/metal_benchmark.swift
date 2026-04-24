// SPDX-License-Identifier: Apache-2.0
//
// Metal GPU benchmark harness for WAVE kernels on Apple Silicon.
//
// Loads generated .metal source files, injects kernel parameters (buffer byte offsets
// and matrix dimensions) via a constant buffer [[buffer(1)]], compiles at runtime via
// MTLDevice.makeLibrary(source:), allocates unified-memory buffers, dispatches compute
// kernels, and measures GPU execution time using MTLCommandBuffer GPU timestamps.
// Reports GFLOPS for each kernel at multiple matrix sizes. Includes MPS baseline
// comparison and correctness verification against CPU reference matmul.

import Metal
import Foundation
import MetalPerformanceShaders

let warmupIters = 5
let benchIters = 20
let sizes = [128, 256, 512, 1024, 2048, 4096]
let f32Tol: Float = 1e-3
let f16Tol: Float = 0.05

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

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
let queue = device.makeCommandQueue()!

print("WAVE Kernel Benchmarks - \(device.name)")
print(String(repeating: "=", count: 70))
print("Warmup: \(warmupIters), Benchmark: \(benchIters) iterations")
print("Timing: MTLCommandBuffer gpuStartTime/gpuEndTime")
print()

struct KernelInfo {
    let name: String
    let display: String
    let metalPath: String
    let entry: String
    let hasBias: Bool
    let isF16: Bool
    var gflopsPerSize: [Int: Double] = [:]
    var correctnessOk: Bool? = nil
    var maxErr: Float = 0
}

var kernels = [
    KernelInfo(name: "blocked_f32", display: "Blocked GEMM (F32, 8x8)",
               metalPath: "/tmp/gemm_blocked.metal", entry: "gemm_register_blocked_8x8",
               hasBias: false, isF16: false),
    KernelInfo(name: "blocked_f16", display: "F16 GEMM (mixed prec)",
               metalPath: "/tmp/gemm_blocked_f16.metal", entry: "gemm_register_blocked_8x8_f16",
               hasBias: false, isF16: true),
    KernelInfo(name: "fused_relu", display: "Fused GEMM+bias+ReLU",
               metalPath: "/tmp/gemm_bias_relu.metal", entry: "gemm_bias_relu_fused",
               hasBias: true, isF16: false),
    KernelInfo(name: "fused_gelu", display: "Fused GEMM+bias+GELU",
               metalPath: "/tmp/gemm_bias_gelu.metal", entry: "gemm_bias_gelu_fused",
               hasBias: true, isF16: false),
    KernelInfo(name: "fused_relu_f16", display: "Fused F16+bias+ReLU",
               metalPath: "/tmp/gemm_bias_relu_f16.metal", entry: "gemm_bias_relu_f16_fused",
               hasBias: true, isF16: true),
]

for ki in 0..<kernels.count {
    let kern = kernels[ki]
    print("--- \(kern.display) ---")

    guard let data = FileManager.default.contents(atPath: kern.metalPath),
          let src = String(data: data, encoding: .utf8) else {
        print("  ERROR: cannot read \(kern.metalPath)"); continue
    }

    let patched = patchSource(src)
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: patched, options: nil) }
    catch { print("  COMPILE ERROR: \(error)"); continue }

    guard let fn = lib.makeFunction(name: kern.entry) else {
        print("  ERROR: function not found"); continue
    }

    let pso: MTLComputePipelineState
    do { pso = try device.makeComputePipelineState(function: fn) }
    catch { print("  PIPELINE ERROR: \(error)"); continue }

    print("  Compiled OK (max threads/group: \(pso.maxTotalThreadsPerThreadgroup))")

    let elemSize = kern.isF16 ? 2 : 4

    for N in sizes {
        let M = N, K = N
        let aOff: Int, bOff: Int, biasOff: Int, cOff: Int, total: Int

        if kern.hasBias {
            aOff = 0; bOff = M*K*elemSize; biasOff = bOff+K*N*elemSize; cOff = biasOff+N*4
            total = cOff+M*N*elemSize
        } else {
            aOff = 0; bOff = M*K*elemSize; biasOff = 0; cOff = bOff+K*N*elemSize
            total = cOff+M*N*elemSize
        }

        guard let buf = device.makeBuffer(length: total, options: .storageModeShared) else {
            print("  \(N)x\(N): alloc failed"); continue
        }

        srand48(42)
        if kern.isF16 {
            let pA = buf.contents().advanced(by: aOff).bindMemory(to: UInt16.self, capacity: M*K)
            let pB = buf.contents().advanced(by: bOff).bindMemory(to: UInt16.self, capacity: K*N)
            for i in 0..<M*K { pA[i] = Float16(Float(drand48() * 2.0 - 1.0)).bitPattern }
            for i in 0..<K*N { pB[i] = Float16(Float(drand48() * 2.0 - 1.0)).bitPattern }
        } else {
            let pA = buf.contents().advanced(by: aOff).bindMemory(to: Float.self, capacity: M*K)
            let pB = buf.contents().advanced(by: bOff).bindMemory(to: Float.self, capacity: K*N)
            for i in 0..<M*K { pA[i] = Float(drand48() * 2.0 - 1.0) }
            for i in 0..<K*N { pB[i] = Float(drand48() * 2.0 - 1.0) }
        }
        if kern.hasBias {
            let pBi = buf.contents().advanced(by: biasOff).bindMemory(to: Float.self, capacity: N)
            for i in 0..<N { pBi[i] = 0.5 }
        }
        memset(buf.contents().advanced(by: cOff), 0, M*N*elemSize)

        var paramVals: [UInt32]
        if kern.hasBias {
            paramVals = [UInt32(aOff), UInt32(bOff), UInt32(biasOff), UInt32(cOff), UInt32(N), UInt32(K)]
        } else {
            paramVals = [UInt32(aOff), UInt32(bOff), UInt32(cOff), UInt32(M), UInt32(N), UInt32(K)]
        }
        let paramBuf = device.makeBuffer(bytes: &paramVals, length: paramVals.count*4, options: .storageModeShared)!

        let grid = MTLSize(width: N/128, height: M/128, depth: 1)
        let wg = MTLSize(width: 16, height: 16, depth: 1)

        for _ in 0..<warmupIters {
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

        guard !times.isEmpty else { print("  \(N)x\(N): no timestamps"); continue }
        times.sort()
        let median = times[times.count / 2]
        let gf = gflops(M, N, K, median)
        kernels[ki].gflopsPerSize[N] = gf
        print(String(format: "  %4dx%-4d  median %.4f ms  %8.1f GFLOPS", N, N, median*1000, gf))

        if N == 128 {
            memset(buf.contents().advanced(by: cOff), 0, M*N*elemSize)
            srand48(42)
            if kern.isF16 {
                let pA = buf.contents().advanced(by: aOff).bindMemory(to: UInt16.self, capacity: M*K)
                let pB = buf.contents().advanced(by: bOff).bindMemory(to: UInt16.self, capacity: K*N)
                for i in 0..<M*K { pA[i] = Float16(Float(drand48() * 2.0 - 1.0)).bitPattern }
                for i in 0..<K*N { pB[i] = Float16(Float(drand48() * 2.0 - 1.0)).bitPattern }
            } else {
                let pA = buf.contents().advanced(by: aOff).bindMemory(to: Float.self, capacity: M*K)
                let pB = buf.contents().advanced(by: bOff).bindMemory(to: Float.self, capacity: K*N)
                for i in 0..<M*K { pA[i] = Float(drand48() * 2.0 - 1.0) }
                for i in 0..<K*N { pB[i] = Float(drand48() * 2.0 - 1.0) }
            }
            if kern.hasBias {
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

            srand48(42)
            var refA = [Float](repeating: 0, count: M*K)
            var refB = [Float](repeating: 0, count: K*N)
            if kern.isF16 {
                for i in 0..<M*K { refA[i] = Float(Float16(Float(drand48()*2.0-1.0))) }
                for i in 0..<K*N { refB[i] = Float(Float16(Float(drand48()*2.0-1.0))) }
            } else {
                for i in 0..<M*K { refA[i] = Float(drand48()*2.0-1.0) }
                for i in 0..<K*N { refB[i] = Float(drand48()*2.0-1.0) }
            }
            var refC = cpuGemm(refA, refB, M, N, K)

            if kern.name == "fused_relu" || kern.name == "fused_relu_f16" {
                for i in 0..<M { for j in 0..<N { refC[i*N+j] = max(0, refC[i*N+j] + 0.5) } }
            } else if kern.name == "fused_gelu" {
                for i in 0..<M {
                    for j in 0..<N {
                        let x = refC[i*N+j] + 0.5
                        refC[i*N+j] = x / (1.0 + exp(-1.702 * x))
                    }
                }
            }

            var maxErr: Float = 0
            if kern.isF16 {
                let refCf16 = refC.map { Float(Float16($0)) }
                let pC = buf.contents().advanced(by: cOff).bindMemory(to: UInt16.self, capacity: M*N)
                for i in 0..<M*N {
                    maxErr = max(maxErr, abs(Float(Float16(bitPattern: pC[i])) - refCf16[i]))
                }
            } else {
                let pC = buf.contents().advanced(by: cOff).bindMemory(to: Float.self, capacity: M*N)
                for i in 0..<M*N { maxErr = max(maxErr, abs(pC[i] - refC[i])) }
            }

            let tol: Float = kern.isF16 ? f16Tol : f32Tol
            kernels[ki].correctnessOk = maxErr <= tol
            kernels[ki].maxErr = maxErr
            print(String(format: "  Correctness (128x128): %@ (max_err=%.2e)",
                         maxErr <= tol ? "PASS" : "FAIL", maxErr))
        }
    }
    print()
}

print("--- MPS Native (Metal Performance Shaders) ---")
var mpsGflops: [Int: Double] = [:]

for N in sizes {
    let M = N, K = N
    let descA = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K*4, dataType: .float32)
    let descB = MPSMatrixDescriptor(rows: K, columns: N, rowBytes: N*4, dataType: .float32)
    let descC = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N*4, dataType: .float32)

    guard let bA = device.makeBuffer(length: M*K*4, options: .storageModeShared),
          let bB = device.makeBuffer(length: K*N*4, options: .storageModeShared),
          let bC = device.makeBuffer(length: M*N*4, options: .storageModeShared) else {
        print("  \(N)x\(N): alloc failed"); continue
    }

    srand48(42)
    let pA = bA.contents().bindMemory(to: Float.self, capacity: M*K)
    let pB = bB.contents().bindMemory(to: Float.self, capacity: K*N)
    for i in 0..<M*K { pA[i] = Float(drand48() * 2.0 - 1.0) }
    for i in 0..<K*N { pB[i] = Float(drand48() * 2.0 - 1.0) }

    let matA = MPSMatrix(buffer: bA, descriptor: descA)
    let matB = MPSMatrix(buffer: bB, descriptor: descB)
    let matC = MPSMatrix(buffer: bC, descriptor: descC)

    let mpsGemm = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                           resultRows: M, resultColumns: N, interiorColumns: K,
                                           alpha: 1.0, beta: 0.0)

    for _ in 0..<warmupIters {
        let cb = queue.makeCommandBuffer()!
        mpsGemm.encode(commandBuffer: cb, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
        cb.commit(); cb.waitUntilCompleted()
    }

    var times: [Double] = []
    for _ in 0..<benchIters {
        let cb = queue.makeCommandBuffer()!
        mpsGemm.encode(commandBuffer: cb, leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
        cb.commit(); cb.waitUntilCompleted()
        let t = cb.gpuEndTime - cb.gpuStartTime
        if t > 0 { times.append(t) }
    }

    guard !times.isEmpty else { print("  \(N)x\(N): no timestamps"); continue }
    times.sort()
    let median = times[times.count / 2]
    let gf = gflops(M, N, K, median)
    mpsGflops[N] = gf
    print(String(format: "  %4dx%-4d  median %.4f ms  %8.1f GFLOPS", N, N, median*1000, gf))
}

print()
print()

let colW = 10
print("WAVE Kernel Benchmarks - \(device.name)")
print(String(repeating: "=", count: 110))

var hdr = "Kernel".padding(toLength: 28, withPad: " ", startingAt: 0)
for s in sizes { hdr += " | " + "\(s)".padding(toLength: colW, withPad: " ", startingAt: 0) }
hdr += " | Unit   | Check"
print(hdr)
print(String(repeating: "-", count: 110))

for kern in kernels {
    var line = kern.display.padding(toLength: 28, withPad: " ", startingAt: 0)
    for s in sizes {
        if let gf = kern.gflopsPerSize[s] {
            line += String(format: " | %8.1f  ", gf)
        } else {
            line += " |      N/A  "
        }
    }
    line += " | GFLOPS"
    if let ok = kern.correctnessOk {
        line += String(format: " | %@ (%.0e)", ok ? "PASS" : "FAIL", kern.maxErr)
    }
    print(line)
}

var mLine = "MPS (native Metal)".padding(toLength: 28, withPad: " ", startingAt: 0)
for s in sizes {
    if let gf = mpsGflops[s] { mLine += String(format: " | %8.1f  ", gf) }
    else { mLine += " |      N/A  " }
}
mLine += " | GFLOPS |"
print(mLine)

print()
if let mps4k = mpsGflops[4096], mps4k > 0 {
    print("WAVE / MPS ratio at 4096x4096:")
    for kern in kernels {
        if let gf = kern.gflopsPerSize[4096] {
            let pct = gf / mps4k * 100.0
            print("  \(kern.display.padding(toLength: 28, withPad: " ", startingAt: 0))  \(String(format: "%5.1f", pct))%")
        }
    }
}

print()
print("--- TSV ---")
print("kernel\tsize\tgflops")
for kern in kernels {
    for s in sizes {
        if let gf = kern.gflopsPerSize[s] {
            print("\(kern.name)\t\(s)\t\(String(format: "%.2f", gf))")
        }
    }
}
for s in sizes {
    if let gf = mpsGflops[s] { print("mps_f32\t\(s)\t\(String(format: "%.2f", gf))") }
}
