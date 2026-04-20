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
/// 17-slot layout matching `SdpaUnifiedArgs` in
/// `kernels/sdpa_unified.h`. Per-step N (T_k) is NOT a slot in this
/// AB — it's bound as a direct kernel buffer at slot 1, tagged via a
/// `BindingName`, and overridden per replay step through
/// `IndirectCommandBuffer.replay(overrides:)`. Call
/// `registerNBinding` once at construction to enable auto-tagging.
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
        case blocks = 12
        case maskKvSeqStride = 13
        case maskQSeqStride = 14
        case maskHeadStride = 15
        case numQHeads = 16
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

    /// One-shot register the `BindingName` under which this handle's
    /// N-buffer (the side-buffer bound at kernel slot 1) should be
    /// tagged during ICB recording. After this call, every ICB
    /// recording of an SDPA dispatch using this handle emits a
    /// tag-binding at `name` on the handle's N-buffer — so the
    /// orchestrator can rewrite slot 1 per replay step via
    /// `IndirectCommandBuffer.replay(overrides: [name: freshN])`,
    /// race-free with any in-flight GPU work on prior steps.
    ///
    /// Call once at handle construction (or at least before the
    /// record step). Not thread-safe — should not race with the
    /// eval_gpu that reads it.
    public func registerNBinding(_ name: BindingName) {
        _ = mlx_metal_persistent_ab_set_scalar_binding_name(
            ctx, name.rawValue)
    }

    // MARK: - Decode-loop registry
    //
    // Every live `PersistentSdpaAbHandle` registers itself here so the
    // decode-loop ICB orchestrator can iterate handles in layer order
    // to install per-step N-binding overrides.
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

    /// Register a per-layer N-binding name on every live SDPA handle
    /// in registration order. `names[i]` is applied to handle `i`;
    /// entries past `names.count` are left unchanged.
    public static func registerNBindings(_ names: [BindingName]) {
        _registryLock.lock()
        let handles = _registry.compactMap { $0.ref }
        _registryLock.unlock()
        for (i, h) in handles.enumerated() where i < names.count {
            h.registerNBinding(names[i])
        }
    }

    /// Snapshot of the live handles in registration (layer) order.
    /// Holds strong refs; don't retain the result across decode steps.
    public static var liveHandles: [PersistentSdpaAbHandle] {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        return _registry.compactMap { $0.ref }
    }

    /// Number of live handles — diagnostic.
    public static var liveHandleCount: Int {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        return _registry.compactMap { $0.ref }.count
    }
}

/// Persistent argument buffer for the single-token RoPE base-path
/// kernel (5 slots). Per-step `offset` is NOT a slot in this AB —
/// it's bound as a direct kernel buffer at slot 1 and overridden per
/// replay step via `IndirectCommandBuffer.replay(overrides:)`.
/// Call `registerOffsetBinding` once at construction to enable
/// auto-tagging during record.
///
/// Layout (matches `RoPE::eval_gpu` AB branch, base path):
///   0: BufferPtrOffset  in
///   1: BufferPtrOffset  out
///   2: Float32          scale
///   3: Scalar64         stride
///   4: Float32          base   (= log2(theta_base))
public final class PersistentRopeAbHandle {
    public enum Slot: Int32 {
        case `in` = 0
        case out = 1
        case scale = 2
        case stride = 3
        case base = 4
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
        Self.register(self)
    }

    deinit {
        Self.unregister(self)
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    public func setFloat32(slot: Slot, value: Float) {
        _ = mlx_metal_persistent_ab_set_float32(ctx, slot.rawValue, value)
    }

    public func setScalar64(slot: Slot, value: UInt64) {
        _ = mlx_metal_persistent_ab_set_scalar64(ctx, slot.rawValue, value)
    }

    /// One-shot register the `BindingName` under which the per-step
    /// offset buffer (bound at kernel slot 1) should be tagged
    /// during ICB recording. Same mechanism as
    /// `PersistentSdpaAbHandle.registerNBinding`.
    public func registerOffsetBinding(_ name: BindingName) {
        _ = mlx_metal_persistent_ab_set_scalar_binding_name(
            ctx, name.rawValue)
    }

    // MARK: - Decode-loop registry

    private static let _registryLock = NSLock()
    nonisolated(unsafe) private static var _registry: [WeakHandle] = []

    private struct WeakHandle {
        weak var ref: PersistentRopeAbHandle?
    }

    private static func register(_ h: PersistentRopeAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref == nil }
        _registry.append(WeakHandle(ref: h))
    }

    private static func unregister(_ h: PersistentRopeAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref === h || $0.ref == nil }
    }

    /// Register the supplied binding name on every live base-path
    /// RoPE handle. All layers share the same name, so a single
    /// `overrides[name] = newOffset` rewrites every recorded slot-1
    /// bind in one pass.
    public static func registerOffsetBindingOnAll(_ name: BindingName) {
        _registryLock.lock()
        let handles = _registry.compactMap { $0.ref }
        _registryLock.unlock()
        for h in handles {
            h.registerOffsetBinding(name)
        }
    }

    public static var liveHandleCount: Int {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        return _registry.compactMap { $0.ref }.count
    }
}

/// Persistent argument buffer for the single-token RoPE freqs-path
/// kernel (6 slots). Per-step `offset` is NOT a slot — bound at
/// kernel slot 1 with override-based retargeting. Use this when
/// the model supplies precomputed `freqs` (YarnRoPE / scaled RoPE).
///
/// Layout:
///   0: BufferPtrOffset  in
///   1: BufferPtrOffset  out
///   2: Float32          scale
///   3: Scalar64         stride
///   4: BufferPtrOffset  freqs
///   5: Scalar64         freq_stride
public final class PersistentRopeFreqsAbHandle {
    public enum Slot: Int32 {
        case `in` = 0
        case out = 1
        case scale = 2
        case stride = 3
        case freqs = 4
        case freqStride = 5
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
        Self.register(self)
    }

    deinit {
        Self.unregister(self)
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    public func setFloat32(slot: Slot, value: Float) {
        _ = mlx_metal_persistent_ab_set_float32(ctx, slot.rawValue, value)
    }

    public func setScalar64(slot: Slot, value: UInt64) {
        _ = mlx_metal_persistent_ab_set_scalar64(ctx, slot.rawValue, value)
    }

    public func setBufferPtr(slot: Slot, array: MLXArray) {
        _ = mlx_metal_persistent_ab_set_buffer_ptr(ctx, slot.rawValue, array.ctx)
    }

    /// One-shot register the `BindingName` under which the per-step
    /// offset buffer (bound at kernel slot 1) should be tagged
    /// during ICB recording.
    public func registerOffsetBinding(_ name: BindingName) {
        _ = mlx_metal_persistent_ab_set_scalar_binding_name(
            ctx, name.rawValue)
    }

    // MARK: - Decode-loop registry

    private static let _registryLock = NSLock()
    nonisolated(unsafe) private static var _registry: [WeakHandle] = []

    private struct WeakHandle {
        weak var ref: PersistentRopeFreqsAbHandle?
    }

    private static func register(_ h: PersistentRopeFreqsAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref == nil }
        _registry.append(WeakHandle(ref: h))
    }

    private static func unregister(_ h: PersistentRopeFreqsAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref === h || $0.ref == nil }
    }

    public static func registerOffsetBindingOnAll(_ name: BindingName) {
        _registryLock.lock()
        let handles = _registry.compactMap { $0.ref }
        _registryLock.unlock()
        for h in handles {
            h.registerOffsetBinding(name)
        }
    }

    public static var liveHandleCount: Int {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        return _registry.compactMap { $0.ref }.count
    }
}

/// Persistent argument buffer for the `gather_front_ab` kernel (4
/// slots). Per-step `indices` is NOT a slot — bound at kernel slot 1
/// and overridden per replay step via the tag-binding path.
///
/// Layout:
///   0: BufferPtrOffset  src      (weight table; stable across steps)
///   1: BufferPtrOffset  out      (destination; stable across steps)
///   2: Scalar64         stride
///   3: Scalar32         size
public final class PersistentGatherFrontAbHandle {
    public enum Slot: Int32 {
        case src = 0
        case out = 1
        case stride = 2
        case size = 3
    }

    internal var ctx: mlx_metal_persistent_ab

    public init(stream: StreamOrDevice = .default) {
        var handle = mlx_metal_persistent_ab(ctx: nil)
        let rc = mlx_metal_persistent_ab_new_gather_front(&handle, stream.ctx)
        precondition(
            rc == 0 && handle.ctx != nil,
            "mlx_metal_persistent_ab_new_gather_front failed (see mlx error log)"
        )
        self.ctx = handle
        Self.register(self)
    }

    deinit {
        Self.unregister(self)
        _ = mlx_metal_persistent_ab_free(ctx)
    }

    public func setBufferPtr(slot: Slot, array: MLXArray) {
        _ = mlx_metal_persistent_ab_set_buffer_ptr(ctx, slot.rawValue, array.ctx)
    }

    /// Push this handle onto the thread-local gather_front_ab handoff
    /// queue. FIFO: the next matching gather consumes this handle;
    /// further gathers consume additional handles. Push once per
    /// gather the caller wants to override. For a QuantizedEmbedding
    /// lookup that's three handles in a row (weight/scales/biases).
    public func pushAsNextGatherFront() {
        _ = mlx_metal_push_next_gather_front_persistent_ab(ctx)
    }

    /// Drain the thread-local gather_front_ab handoff queue. Safe
    /// even when the queue is empty.
    public static func clearPendingGatherFront() {
        _ = mlx_metal_clear_next_gather_front_persistent_abs()
    }

    /// One-shot register the `BindingName` under which the per-step
    /// indices buffer (bound at kernel slot 1) should be tagged
    /// during ICB recording.
    public func registerIndicesBinding(_ name: BindingName) {
        _ = mlx_metal_persistent_ab_set_scalar_binding_name(
            ctx, name.rawValue)
    }

    // MARK: - Decode-loop registry

    private static let _registryLock = NSLock()
    nonisolated(unsafe) private static var _registry: [WeakHandle] = []

    private struct WeakHandle {
        weak var ref: PersistentGatherFrontAbHandle?
    }

    private static func register(_ h: PersistentGatherFrontAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref == nil }
        _registry.append(WeakHandle(ref: h))
    }

    private static func unregister(_ h: PersistentGatherFrontAbHandle) {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        _registry.removeAll { $0.ref === h || $0.ref == nil }
    }

    /// Register the supplied binding name on every live gather_front
    /// handle. All three gathers in a QuantizedEmbedding lookup share
    /// the same name, so one override entry rewrites every slot-1
    /// bind.
    public static func registerIndicesBindingOnAll(_ name: BindingName) {
        _registryLock.lock()
        let handles = _registry.compactMap { $0.ref }
        _registryLock.unlock()
        for h in handles {
            h.registerIndicesBinding(name)
        }
    }

    public static var liveHandleCount: Int {
        _registryLock.lock()
        defer { _registryLock.unlock() }
        return _registry.compactMap { $0.ref }.count
    }
}
