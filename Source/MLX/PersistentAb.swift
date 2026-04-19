// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// A caller-owned argument buffer whose MTLBuffer address is stable
/// across calls. Used with `IndirectCommandBuffer` recording so a
/// recorded dispatch of an AB-participating primitive (RMSNorm today;
/// other primitives land as Option A rolls out — see
/// mlx-swift-lm/benchmarks/notes/persistent-ab-pilot-design-2026-04-18.md)
/// can be replayed at the next decode step after the caller rewrites
/// its static scalars.
///
/// ## Typical decode-loop use (RMSNorm)
///
/// ```swift
/// let handle = PersistentRmsAbHandle(stream: stream)
/// handle.setScalar32(slot: .axisSize, value: UInt32(hidden))
/// handle.setScalar32(slot: .wStride, value: 1)
/// handle.setFloat32(slot: .eps, value: eps)
///
/// // On every decode step:
/// let y = MLXFast.rmsNormAb(x, weight: w, eps: eps, handle: handle,
///                           stream: stream)
/// ```
///
/// `axisSize`, `wStride`, and `eps` don't change across decode steps
/// for a given layer, so they're written once at handle construction.
/// Buffer pointers (x/w/out) are written per-call by mlx C++ in
/// `RMSNorm::eval_gpu`.
///
/// ## Lifetime
///
/// The underlying C handle owns an `MTL::Buffer` pooled from mlx's
/// shared AB pool. `deinit` returns it to the pool. Not `Sendable`:
/// the caller should keep one handle per layer-module instance and
/// access it from the model's owning queue.
public final class PersistentRmsAbHandle {
    /// Slot layout for `mlx_metal_persistent_ab_new_rmsnorm`. Matches
    /// the C++ `RMSNorm::eval_gpu` AB branch. Buffer-ptr slots are
    /// written by mlx C++ per call; caller writes only the scalar
    /// slots (usually once at construction).
    public enum Slot: Int32 {
        case x = 0
        case w = 1
        case out = 2
        case eps = 3
        case axisSize = 4
        case wStride = 5
    }

    /// Opaque handle as held by mlx-c. Private — do not reach through
    /// it directly; use the typed setters.
    internal var ctx: mlx_metal_persistent_ab

    public init(stream: StreamOrDevice = .default) {
        var handle = mlx_metal_persistent_ab(ctx: nil)
        let rc = mlx_metal_persistent_ab_new_rmsnorm(&handle, stream.ctx)
        precondition(
            rc == 0 && handle.ctx != nil,
            "mlx_metal_persistent_ab_new_rmsnorm failed (see mlx error log)"
        )
        self.ctx = handle
    }

    deinit {
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    /// Write a Float32 slot. Slot kind must match (Float32 slots only
    /// — e.g. `.eps`). Throws at the C++ layer on kind mismatch.
    public func setFloat32(slot: Slot, value: Float) {
        _ = mlx_metal_persistent_ab_set_float32(ctx, slot.rawValue, value)
    }

    /// Write a Scalar32 slot. Slot kind must match (Scalar32 slots
    /// only — e.g. `.axisSize`, `.wStride`).
    public func setScalar32(slot: Slot, value: UInt32) {
        _ = mlx_metal_persistent_ab_set_scalar32(ctx, slot.rawValue, value)
    }
}

/// Persistent argument buffer for the unified vector SDPA kernel.
/// 18-slot layout matching `SdpaUnifiedArgs` in
/// `kernels/sdpa_unified.h`. Per-decode-step the caller writes `.N`
/// (T_k) to reflect the current K-sequence length; mlx C++ handles
/// buffer pointers and most other scalars internally per call.
public final class PersistentSdpaAbHandle {
    public enum Slot: Int32 {
        case queries = 0
        case keys = 1
        case values = 2
        case out = 3
        case mask = 4
        case sinks = 5
        case kHeadStride = 6
        case kSeqStride = 7
        case vHeadStride = 8
        case vSeqStride = 9
        case scale = 10
        case gqaFactor = 11
        case N = 12
        case blocks = 13
        case maskKvSeqStride = 14
        case maskQSeqStride = 15
        case maskHeadStride = 16
        case numQHeads = 17
    }

    internal var ctx: mlx_metal_persistent_ab

    public init(stream: StreamOrDevice = .default) {
        var handle = mlx_metal_persistent_ab(ctx: nil)
        let rc = mlx_metal_persistent_ab_new_sdpa(&handle, stream.ctx)
        precondition(
            rc == 0 && handle.ctx != nil,
            "mlx_metal_persistent_ab_new_sdpa failed (see mlx error log)"
        )
        self.ctx = handle
        Self.register(self)
    }

    deinit {
        Self.unregister(self)
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    public func setScalar32(slot: Slot, value: UInt32) {
        _ = mlx_metal_persistent_ab_set_scalar32(ctx, slot.rawValue, value)
    }

    public func setFloat32(slot: Slot, value: Float) {
        _ = mlx_metal_persistent_ab_set_float32(ctx, slot.rawValue, value)
    }

    // MARK: - Decode-loop registry
    //
    // Every live `PersistentSdpaAbHandle` registers itself here so the
    // decode-loop ICB orchestrator in `TokenIterator` can update the
    // per-step `N` (T_k) scalar on every handle without needing
    // per-model plumbing to reach each attention block's handle.
    //
    // The registry holds weak refs; deinit removes the entry. Thread-
    // safe via a simple lock — decode is single-threaded, so contention
    // is only with model-init on concurrent queues.

    private static let _registryLock = NSLock()
    nonisolated(unsafe) private static var _registry: [WeakHandle] = []

    private struct WeakHandle {
        weak var ref: PersistentSdpaAbHandle?
    }

    private static func register(_ h: PersistentSdpaAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref == nil }
        _registry.append(WeakHandle(ref: h))
    }

    private static func unregister(_ h: PersistentSdpaAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref === h || $0.ref == nil }
    }

    /// Update the `N` (T_k) slot on every live SDPA handle. Called by
    /// the decode-loop ICB orchestrator each replay step so all
    /// attention layers attend to the current K-sequence length.
    public static func updateNOnAll(_ n: UInt32) {
        _registryLock.lock()
        let handles = _registry.compactMap { $0.ref }
        _registryLock.unlock()
        for h in handles {
            h.setScalar32(slot: .N, value: n)
        }
    }

    /// Number of live handles — diagnostic.
    public static var liveHandleCount: Int {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        return _registry.compactMap { $0.ref }.count
    }
}

/// Persistent argument buffer for the single-token RoPE base-path
/// kernel (6 slots). Use this for layers where `freqs` is not
/// supplied — the offset is consumed from a device buffer at dispatch
/// time, so per-step decoding only requires updating that buffer's
/// contents (mlx C++ rewrites the `offset` pointer per call).
///
/// Layout (matches `RoPE::eval_gpu` AB branch, base path):
///   0: BufferPtrOffset  in
///   1: BufferPtrOffset  out
///   2: BufferPtrOffset  offset
///   3: Float32          scale
///   4: Scalar64         stride
///   5: Float32          base   (= log2(theta_base))
public final class PersistentRopeAbHandle {
    public enum Slot: Int32 {
        case `in` = 0
        case out = 1
        case offset = 2
        case scale = 3
        case stride = 4
        case base = 5
    }

    internal var ctx: mlx_metal_persistent_ab

    public init(stream: StreamOrDevice = .default) {
        var handle = mlx_metal_persistent_ab(ctx: nil)
        let rc = mlx_metal_persistent_ab_new_rope(&handle, stream.ctx)
        precondition(
            rc == 0 && handle.ctx != nil,
            "mlx_metal_persistent_ab_new_rope failed (see mlx error log)"
        )
        self.ctx = handle
    }

    deinit {
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    public func setFloat32(slot: Slot, value: Float) {
        _ = mlx_metal_persistent_ab_set_float32(ctx, slot.rawValue, value)
    }

    public func setScalar64(slot: Slot, value: UInt64) {
        _ = mlx_metal_persistent_ab_set_scalar64(ctx, slot.rawValue, value)
    }
}

/// Persistent argument buffer for the single-token RoPE freqs-path
/// kernel (7 slots). Use this when the model supplies precomputed
/// `freqs` (typical for YarnRoPE / scaled RoPE variants).
///
/// Layout:
///   0: BufferPtrOffset  in
///   1: BufferPtrOffset  out
///   2: BufferPtrOffset  offset
///   3: Float32          scale
///   4: Scalar64         stride
///   5: BufferPtrOffset  freqs
///   6: Scalar64         freq_stride
public final class PersistentRopeFreqsAbHandle {
    public enum Slot: Int32 {
        case `in` = 0
        case out = 1
        case offset = 2
        case scale = 3
        case stride = 4
        case freqs = 5
        case freqStride = 6
    }

    internal var ctx: mlx_metal_persistent_ab

    public init(stream: StreamOrDevice = .default) {
        var handle = mlx_metal_persistent_ab(ctx: nil)
        let rc = mlx_metal_persistent_ab_new_rope_freqs(&handle, stream.ctx)
        precondition(
            rc == 0 && handle.ctx != nil,
            "mlx_metal_persistent_ab_new_rope_freqs failed (see mlx error log)"
        )
        self.ctx = handle
    }

    deinit {
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    public func setFloat32(slot: Slot, value: Float) {
        _ = mlx_metal_persistent_ab_set_float32(ctx, slot.rawValue, value)
    }

    public func setScalar64(slot: Slot, value: UInt64) {
        _ = mlx_metal_persistent_ab_set_scalar64(ctx, slot.rawValue, value)
    }
}
