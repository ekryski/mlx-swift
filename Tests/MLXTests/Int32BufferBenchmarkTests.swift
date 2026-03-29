// Copyright © 2026 Eric Kryski
//
// Benchmarks comparing heap-allocated .asInt32 vs stack-allocated
// withInt32Buffer for Int→Int32 conversion in hot-path MLX operations.
//
// The standard MLX ops pass shape/axes arrays as Int32 to C functions.
// Previously this used .asInt32 which heap-allocates a new [Int32] array
// on every call. The withInt32Buffer helper uses stack allocation for
// arrays up to 8 elements (covering nearly all ML shape/axes operations).
//
// We benchmark operations that take axes/shape parameters since those
// are the ones affected by this change. No eval() is called — we measure
// pure dispatch overhead including the Int→Int32 conversion cost.

import Foundation
import XCTest

@testable import MLX

class Int32BufferBenchmarkTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    let iterations = 10_000

    // Typical ML shapes and axes
    static let mat2D = MLXArray(converting: [1.0, 2.0, 3.0, 4.0], [2, 2])
    static let mat3D = MLXArray(
        converting: Array(repeating: 1.0, count: 24), [2, 3, 4])
    static let mat4D = MLXArray(
        converting: Array(repeating: 1.0, count: 120), [2, 3, 4, 5])

    // MARK: - Reshape (shape array conversion)

    func testReshape2D() {
        let a = Self.mat3D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.reshaped([6, 4])
            }
        }
    }

    func testReshape4D() {
        let a = Self.mat4D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.reshaped([2, 3, 20])
            }
        }
    }

    // MARK: - Transpose (axes array conversion)

    func testTranspose2D() {
        let a = Self.mat2D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.transposed(axes: [1, 0])
            }
        }
    }

    func testTranspose4D() {
        let a = Self.mat4D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.transposed(axes: [0, 2, 1, 3])
            }
        }
    }

    // MARK: - Sum with axes (axes array conversion)

    func testSumAxis() {
        let a = Self.mat3D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.sum(axes: [1])
            }
        }
    }

    func testSumMultipleAxes() {
        let a = Self.mat4D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.sum(axes: [1, 2])
            }
        }
    }

    // MARK: - Mean with axes

    func testMeanAxis() {
        let a = Self.mat3D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.mean(axes: [1])
            }
        }
    }

    // MARK: - Max with axes

    func testMaxAxis() {
        let a = Self.mat3D
        measure {
            for _ in 0 ..< iterations {
                let _ = a.max(axes: [1])
            }
        }
    }

    // MARK: - Softmax (axes conversion)

    func testSoftmax() {
        let a = Self.mat3D
        measure {
            for _ in 0 ..< iterations {
                let _ = softmax(a, axes: [-1])
            }
        }
    }

    // MARK: - Squeeze (axes conversion)

    func testSqueeze() {
        let a = MLXArray(converting: [1.0, 2.0, 3.0], [1, 3, 1])
        measure {
            for _ in 0 ..< iterations {
                let _ = a.squeezed(axes: [0, 2])
            }
        }
    }

    // MARK: - Simulated decoder layer dispatch
    // Tests the cumulative effect across a realistic sequence of ops
    // that all use axes/shape conversion.

    func testDecoderLayerShapeOps() {
        let hidden = 256
        let heads = 4
        let headDim = hidden / heads
        let x = MLXArray(
            converting: Array(repeating: 0.5, count: hidden), [1, 1, hidden])
        let w = MLXArray(
            converting: Array(repeating: 0.01, count: hidden * hidden),
            [hidden, hidden])

        measure {
            for _ in 0 ..< 1000 {
                // Simulates the shape manipulation in one attention layer:
                // matmul → reshape → transpose → matmul → softmax → matmul → transpose → reshape
                let q = matmul(x, w)
                let qr = q.reshaped([1, 1, heads, headDim])
                let qt = qr.transposed(axes: [0, 2, 1, 3])
                let scores = matmul(qt, qt.transposed(axes: [0, 1, 3, 2]))
                let weights = softmax(scores, axes: [-1])
                let context = matmul(weights, qt)
                let out = context.transposed(axes: [0, 2, 1, 3])
                    .reshaped([1, 1, hidden])
                let _ = matmul(out, w)
            }
        }
    }
}
