// SPDX-License-Identifier: Apache-2.0
// Verify that numerical validation rejects missing, nonfinite, and wrong output.

import Foundation

@main
struct BenchmarkSupportTests {
    static func main() throws {
        let finite = validateOutput([1, 2], [1, 2], isF16: false)
        precondition(finite["correct"] as? Bool == true)
        for invalid: Float in [.nan, .infinity, -.infinity] {
            let result = validateOutput([1, invalid], [1, 2], isF16: false)
            precondition(result["correct"] as? Bool == false)
            precondition(result["nonfinite_elements"] as? Int == 1)
            _ = try jsonData(result)
        }
        precondition(validateOutput([1], [1, 2], isF16: false)["correct"] as? Bool == false)
        precondition(validateOutput([1, 5], [1, 2], isF16: false)["correct"] as? Bool == false)
        precondition(referenceGemm([1, 2, 3, 4], [5, 6, 7, 8], size: 2,
                                  epilogue: "none", isF16: false) == [19, 22, 43, 50])
        let halfInput = Float(Float16(0.3333))
        let halfResult = referenceGemm([halfInput], [halfInput], size: 1, epilogue: "none", isF16: true)
        precondition(halfResult == [Float(Float16(halfInput * halfInput))])
        precondition(median([1, 8, 3, 2]) == 2.5)
        let encoded = try jsonData(failureRecord(BenchmarkFailure(description: "quoted \"message\"\nnew line")))
        let decoded = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        precondition(decoded?["error"] as? String == "quoted \"message\"\nnew line")
        print("Benchmark validation checks passed")
    }
}
