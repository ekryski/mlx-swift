// Copyright © 2026 Eric Kryski.

import Foundation
import MLX
import XCTest

/// Tests for GatedDeltaNet kernel bindings and memory management.
///
/// The standard variant uses pre-computed q, k, g, beta.
/// The fused variant computes rmsNorm, g, beta inside the kernel.
///
/// Both variants had a memory leak where the `mlx_vector_array` result
/// container was never freed (missing `defer { mlx_vector_array_free }`).
class GatedDeltaTests: XCTestCase {

    // Use the smallest instantiated kernel: Dk=64, Dv=64, Hk=8, Hv=8
    let Dk = 64
    let Dv = 64
    let Hk = 8
    let Hv = 8
    let B = 1
    let T = 1

    // MARK: - Standard variant

    func testGatedDeltaStepOutputShape() throws {
        try skipIfNeedGPU()

        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
        let g = full([B, T, Hv], values: MLXArray(0.99), dtype: .float16)
        let beta = full([B, T, Hv], values: MLXArray(0.1), dtype: .float16)
        let state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)

        let results = MLXFast.gatedDeltaStep(
            q: q, k: k, v: v, g: g, beta: beta, state: state,
            T: T, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)

        XCTAssertEqual(results.count, 2, "Should return [y, state_out]")
        eval(results[0], results[1])
        XCTAssertEqual(results[0].shape, [B, T, Hv, Dv])
        XCTAssertEqual(results[1].shape, [B, Hv, Dv, Dk])
    }

    func testGatedDeltaStepStateUpdate() throws {
        try skipIfNeedGPU()

        // With beta > 0 and non-zero v, state should change from zero
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
        let g = full([B, T, Hv], values: MLXArray(0.99), dtype: .float16)
        let beta = full([B, T, Hv], values: MLXArray(0.5), dtype: .float16)
        let state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)

        let results = MLXFast.gatedDeltaStep(
            q: q, k: k, v: v, g: g, beta: beta, state: state,
            T: T, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)

        eval(results[0], results[1])

        // State should be non-zero after processing non-zero inputs
        let stateNorm = results[1].square().sum().sqrt().item(Float.self)
        XCTAssertGreaterThan(stateNorm, 0, "State should be updated from zero")
    }

    // MARK: - Fused variant

    func testGatedDeltaStepFusedOutputShape() throws {
        try skipIfNeedGPU()

        let qRaw = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let kRaw = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
        let a = MLXRandom.normal([B, T, Hv]).asType(.float16)
        let bInput = MLXRandom.normal([B, T, Hv]).asType(.float16)
        let aLog = full([Hv], values: MLXArray(-5.0), dtype: .float16)
        let dtBias = MLXArray.zeros([Hv]).asType(.float16)
        let state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)

        let results = MLXFast.gatedDeltaStepFused(
            qRaw: qRaw, kRaw: kRaw, v: v, a: a, bInput: bInput,
            aLog: aLog, dtBias: dtBias, state: state,
            T: T, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)

        XCTAssertEqual(results.count, 2, "Should return [y, state_out]")
        eval(results[0], results[1])
        XCTAssertEqual(results[0].shape, [B, T, Hv, Dv])
        XCTAssertEqual(results[1].shape, [B, Hv, Dv, Dk])
    }

    // MARK: - Memory leak regression tests

    /// Verifies that repeated calls to gatedDeltaStep do not leak memory.
    ///
    /// Before the fix, each call leaked the `mlx_vector_array` container
    /// (which also held shared_ptr refs to output arrays, keeping GPU buffers alive).
    /// With the fix (`defer { mlx_vector_array_free(result) }`), the container
    /// is freed when the function returns.
    func testGatedDeltaStepNoMemoryLeak() throws {
        try skipIfNeedGPU()

        let iterations = 200

        // Warm up - first call may allocate caches
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
        let g = full([B, T, Hv], values: MLXArray(0.99), dtype: .float16)
        let beta = full([B, T, Hv], values: MLXArray(0.1), dtype: .float16)
        var state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)

        for _ in 0..<5 {
            let results = MLXFast.gatedDeltaStep(
                q: q, k: k, v: v, g: g, beta: beta, state: state,
                T: T, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            eval(results[0], results[1])
            state = results[1]
        }
        eval(state)
        Memory.clearCache()

        let baselineMemory = Memory.activeMemory

        // Run many iterations, discarding outputs each time
        for _ in 0..<iterations {
            state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)
            let results = MLXFast.gatedDeltaStep(
                q: q, k: k, v: v, g: g, beta: beta, state: state,
                T: T, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            eval(results[0], results[1])
        }
        Memory.clearCache()

        let finalMemory = Memory.activeMemory

        // With the leak, each iteration leaks ~32KB (state_out) + ~512B (y)
        // worth of GPU buffer refs. 200 iterations would leak ~6.5MB.
        // With the fix, memory should be roughly stable (within ~1MB tolerance).
        let growth = finalMemory - baselineMemory
        XCTAssertLessThan(
            growth, 2 * 1024 * 1024,
            "Memory grew by \(growth) bytes over \(iterations) iterations — possible leak")
    }

    /// Same leak test for the fused variant.
    func testGatedDeltaStepFusedNoMemoryLeak() throws {
        try skipIfNeedGPU()

        let iterations = 200

        let qRaw = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let kRaw = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
        let a = MLXRandom.normal([B, T, Hv]).asType(.float16)
        let bInput = MLXRandom.normal([B, T, Hv]).asType(.float16)
        let aLog = full([Hv], values: MLXArray(-5.0), dtype: .float16)
        let dtBias = MLXArray.zeros([Hv]).asType(.float16)
        var state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)

        // Warm up
        for _ in 0..<5 {
            let results = MLXFast.gatedDeltaStepFused(
                qRaw: qRaw, kRaw: kRaw, v: v, a: a, bInput: bInput,
                aLog: aLog, dtBias: dtBias, state: state,
                T: T, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            eval(results[0], results[1])
            state = results[1]
        }
        eval(state)
        Memory.clearCache()

        let baselineMemory = Memory.activeMemory

        for _ in 0..<iterations {
            state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)
            let results = MLXFast.gatedDeltaStepFused(
                qRaw: qRaw, kRaw: kRaw, v: v, a: a, bInput: bInput,
                aLog: aLog, dtBias: dtBias, state: state,
                T: T, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
            eval(results[0], results[1])
        }
        Memory.clearCache()

        let finalMemory = Memory.activeMemory
        let growth = finalMemory - baselineMemory
        XCTAssertLessThan(
            growth, 2 * 1024 * 1024,
            "Memory grew by \(growth) bytes over \(iterations) iterations — possible leak")
    }

    // MARK: - Peak memory scaling tests

    /// Verifies that peak GPU memory does not scale linearly with context length.
    ///
    /// Before the async_eval fix, make_arrays created a sibling bond between
    /// y [B,T,Hv,Dv] and state_out [B,Hv,Dv,Dk].  Holding state_out (needed
    /// across tokens) pinned the T-proportional y buffer.  With N layers this
    /// caused peak ∝ layers × context.
    ///
    /// With the fix, async_eval detaches outputs immediately, so y is freed
    /// as soon as downstream consumers are done with it.
    func testPeakMemoryDoesNotScaleWithContext() throws {
        try skipIfNeedGPU()

        // Simulate multiple GDN layers processing different context lengths.
        // Without the fix, peak grows ~linearly with T × numLayers.
        let numLayers = 8

        func runLayers(T: Int) {
            let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
            let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
            let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
            let g = full([B, T, Hv], values: MLXArray(0.99), dtype: .float16)
            let beta = full([B, T, Hv], values: MLXArray(0.1), dtype: .float16)
            var state = MLXArray.zeros([B, Hv, Dv, Dk]).asType(.float16)

            for _ in 0..<numLayers {
                let results = MLXFast.gatedDeltaStep(
                    q: q, k: k, v: v, g: g, beta: beta, state: state,
                    T: T, fused: false, Dk: Dk, Dv: Dv, Hk: Hk, Hv: Hv)
                eval(results[0], results[1])
                state = results[1]
            }
            eval(state)
        }

        // Warm up
        runLayers(T: 10)
        Memory.clearCache()
        Memory.peakMemory = 0

        // Measure peak with small context
        runLayers(T: 100)
        let peakSmall = Memory.peakMemory
        Memory.clearCache()
        Memory.peakMemory = 0

        // Measure peak with large context
        runLayers(T: 1000)
        let peakLarge = Memory.peakMemory
        Memory.clearCache()

        // Without the fix, peakLarge ≈ 10× peakSmall (linear in T).
        // With the fix, peakLarge should be < 5× peakSmall (sub-linear,
        // dominated by input buffers not the accumulated y outputs).
        let ratio = Double(peakLarge) / Double(max(peakSmall, 1))
        XCTAssertLessThan(
            ratio, 5.0,
            "Peak scaled \(String(format: "%.1f", ratio))x for 10x context — sibling pinning likely")
    }

    // MARK: - Helpers

    private func skipIfNeedGPU() throws {
        // GDN kernels only run on GPU (Metal). Skip on CPU-only environments.
        guard MLX.Device.defaultDevice().deviceType == .gpu else {
            throw XCTSkip("GatedDelta kernels require GPU")
        }
    }
}
