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
