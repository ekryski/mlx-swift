// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// A caller-defined name used to tag an `MLXArray` during ICB recording
/// so that it can later be substituted by `IndirectCommandBuffer.replay(overrides:)`.
///
/// Names are opaque to MLX — they only need to be consistent between
/// the `tag` call and the matching override lookup. A string-literal
/// init is provided for ergonomic call sites:
///
/// ```swift
/// tagger.tag(hiddenState, as: "input")
/// icb.replay(overrides: ["input": nextHiddenState])
/// ```
///
/// The underlying identity passed to the C layer is a 32-bit hash of
/// the string, computed once at init time. Within a process, two
/// `BindingName` values constructed from the same string compare equal
/// and hash to the same `rawValue`. Construct via `rawValue:` if you
/// need a specific numeric ID (e.g. when bridging to an external
/// identifier registry).
public struct BindingName: Hashable, Sendable, ExpressibleByStringLiteral {
    /// The 32-bit ID passed to the C API for this name.
    public let rawValue: UInt32

    public init(_ name: String) {
        self.rawValue = Self.fnv1a32(name)
    }

    public init(stringLiteral value: String) {
        self.init(value)
    }

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    /// FNV-1a 32-bit. Deterministic across runs and architectures;
    /// collisions are possible but overwhelmingly unlikely for the small
    /// symbol sets typical of model-integration code.
    private static func fnv1a32(_ s: String) -> UInt32 {
        var hash: UInt32 = 2_166_136_261
        for byte in s.utf8 {
            hash ^= UInt32(byte)
            hash &*= 16_777_619
        }
        return hash
    }
}

/// A handle passed to the `recordWithBindings` closure so the caller
/// can associate `BindingName`s with `MLXArray`s as their backing
/// MTLBuffers flow through the recorded dispatches.
///
/// See `IndirectCommandBuffer.recordWithBindings` for the typical
/// record-and-replay pattern.
public final class BindingTagger: @unchecked Sendable {
    private let streamCtx: mlx_stream

    fileprivate init(streamCtx: mlx_stream) {
        self.streamCtx = streamCtx
    }

    /// Associate `name` with the `MLXArray`'s underlying MTLBuffer
    /// during the active recording. Any already-recorded dispatch
    /// that bound that buffer is tagged immediately; any subsequent
    /// bind of the same buffer in this recording is also tagged.
    ///
    /// Call only from inside the `recordWithBindings` closure —
    /// calling outside an active recording throws at the C++ layer.
    public func tag(_ array: MLXArray, as name: BindingName) {
        _ = mlx_metal_icb_tag_binding(streamCtx, name.rawValue, array.ctx)
    }
}

/// Captures a stable sequence of MLX compute dispatches into an
/// `MTLIndirectCommandBuffer` and replays it on subsequent calls with
/// substantially lower CPU-side encoding cost.
///
/// Typical usage (at the decoder-layer level in mlx-swift-lm):
///
/// ```swift
/// // First decode step — capture.
/// let icb = try IndirectCommandBuffer.record {
///     _ = model.decoderLayer(hidden, cache: kvCache)
/// }
///
/// // Subsequent decode steps — replay.
/// icb.replay()
/// ```
///
/// The recorded sequence must be dispatch-stable: same kernels, same
/// argument layouts, same dispatch counts. Inputs that change per step
/// (KV cache pointers, the current hidden state) must live in device
/// buffers whose contents can be mutated between replays — the ICB
/// re-runs the same dispatches on the current buffer contents.
///
/// Memory barriers are handled transparently: whenever MLX's existing
/// dependency tracker would insert a barrier, the recorder splits into
/// a new ICB segment and replay inserts a `memoryBarrier` between
/// segments on the live encoder.
public final class IndirectCommandBuffer: @unchecked Sendable {

    /// Whether the current GPU supports `MTLIndirectCommandBuffer` for
    /// compute (argument-buffer tier 2). Effectively true on every
    /// M-series Apple Silicon device.
    public static var isSupported: Bool {
        var out: Bool = false
        mlx_metal_icb_is_supported(&out)
        return out
    }

    /// Record dispatches emitted by `block` into an ICB.
    ///
    /// - Parameters:
    ///   - maxCommandsPerSegment: capacity of each ICB segment. A
    ///     segment is closed and a new one opened whenever MLX would
    ///     emit a memory barrier. If your recorded block emits more
    ///     than this many dispatches between barriers, recording throws.
    ///     Default is 2048, which comfortably covers a full decoder
    ///     layer worth of dispatches on any model in the benchmark
    ///     suite.
    ///   - bytesArenaCapacity: shared pool across all segments for
    ///     spilled `setBytes` payloads (shape info, scale constants,
    ///     axis sizes). Default 64 KB.
    ///   - stream: the MLX stream whose CommandEncoder drives recording.
    ///   - block: the compute work to capture.
    public static func record(
        maxCommandsPerSegment: Int = 2048,
        bytesArenaCapacity: Int = 64 * 1024,
        stream: StreamOrDevice = .default,
        _ block: () throws -> Void
    ) rethrows -> IndirectCommandBuffer {
        mlx_metal_icb_begin_recording(
            stream.ctx,
            maxCommandsPerSegment,
            bytesArenaCapacity)

        var threw: Error? = nil
        do {
            try block()
        } catch {
            threw = error
        }

        // If the user block threw, the C++ side may have already aborted
        // recording (e.g. on arena exhaustion) — but it also may not have.
        // Force-abort unconditionally so we never leak recording state.
        if threw != nil {
            _ = mlx_metal_icb_abort_recording(stream.ctx)
            try { throw threw! }()
        }

        var recorder = mlx_metal_icb_recorder(ctx: nil)
        let rc = mlx_metal_icb_end_recording(stream.ctx, &recorder)
        if rc != 0 {
            _ = mlx_metal_icb_recorder_free(recorder)
            // end_recording's own errors (e.g. pending command) also clear
            // recording state, but call abort anyway as a belt-and-braces
            // guard.
            _ = mlx_metal_icb_abort_recording(stream.ctx)
            preconditionFailure(
                "mlx_metal_icb_end_recording failed (see mlx error log)")
        }
        return IndirectCommandBuffer(ctx: recorder)
    }

    /// Replay the captured sequence on `stream`. Each segment's
    /// referenced buffers are made resident via `useResource` and its
    /// commands are issued via `executeCommandsInBuffer`, with a
    /// memory barrier between consecutive segments.
    public func replay(stream: StreamOrDevice = .default) {
        mlx_metal_icb_replay(stream.ctx, ctx)
    }

    /// Record dispatches with named MTLBuffer bindings that can be
    /// substituted on subsequent replays.
    ///
    /// Within `block`, tag the `MLXArray`s whose storage will change
    /// between replays (typical examples: the hidden-state input to a
    /// decoder layer, the KV cache key/value arrays, the slot into
    /// which the layer writes its output). On replay, pass the new
    /// values via `replay(overrides:)` and the ICB's dispatches are
    /// re-bound to the new buffers before execution.
    ///
    /// ```swift
    /// let icb = try IndirectCommandBuffer.recordWithBindings(stream: s) { tagger in
    ///     tagger.tag(hidden,          as: "input")
    ///     tagger.tag(cache.keys,      as: "k_cache")
    ///     tagger.tag(cache.values,    as: "v_cache")
    ///     hidden = layer(hidden, mask: mask, cache: cache)
    ///     tagger.tag(hidden,          as: "output")
    /// }
    /// // per subsequent step:
    /// icb.replay(overrides: [
    ///     "input":   nextHidden,
    ///     "k_cache": cache.keys,
    ///     "v_cache": cache.values,
    ///     "output":  outputSlot,
    /// ])
    /// ```
    ///
    /// Semantically equivalent to `record` — this overload only adds
    /// the tagger handle; all other parameters match.
    public static func recordWithBindings(
        maxCommandsPerSegment: Int = 2048,
        bytesArenaCapacity: Int = 64 * 1024,
        stream: StreamOrDevice = .default,
        _ block: (BindingTagger) throws -> Void
    ) rethrows -> IndirectCommandBuffer {
        try record(
            maxCommandsPerSegment: maxCommandsPerSegment,
            bytesArenaCapacity: bytesArenaCapacity,
            stream: stream
        ) {
            let tagger = BindingTagger(streamCtx: stream.ctx)
            try block(tagger)
        }
    }

    /// Replay the captured sequence, substituting the MTLBuffer
    /// binding for each tagged name with the buffer underlying the
    /// override `MLXArray` (offset included from the array's storage).
    /// Names not present in `overrides` keep their recorded binding;
    /// names never tagged during recording are silently skipped.
    ///
    /// Override writes mutate the ICB's indirect compute commands in
    /// place. A subsequent plain `replay()` observes the last
    /// overrides written, so callers that mix the two paths should
    /// either always replay with overrides, or re-establish the
    /// originals explicitly before a plain replay.
    public func replay(
        overrides: [BindingName: MLXArray],
        stream: StreamOrDevice = .default
    ) {
        if overrides.isEmpty {
            replay(stream: stream)
            return
        }
        // Parallel arrays — names[i] matches arrays[i]. A single pass
        // over the dictionary guarantees matching indices.
        var names = [UInt32]()
        var arrays = [mlx_array]()
        names.reserveCapacity(overrides.count)
        arrays.reserveCapacity(overrides.count)
        for (name, array) in overrides {
            names.append(name.rawValue)
            arrays.append(array.ctx)
        }
        names.withUnsafeBufferPointer { namesPtr in
            arrays.withUnsafeBufferPointer { arraysPtr in
                _ = mlx_metal_icb_replay_with_overrides(
                    stream.ctx,
                    ctx,
                    namesPtr.baseAddress,
                    arraysPtr.baseAddress,
                    overrides.count)
            }
        }
    }

    /// Number of ICB segments (barrier-separated blocks) in this recording.
    public var numSegments: Int {
        var out: Int = 0
        mlx_metal_icb_recorder_num_segments(ctx, &out)
        return out
    }

    /// Total number of compute dispatches across all segments.
    public var totalCommands: Int {
        var out: Int = 0
        mlx_metal_icb_recorder_size(ctx, &out)
        return out
    }

    // MARK: - Internals

    private var ctx: mlx_metal_icb_recorder

    private init(ctx: mlx_metal_icb_recorder) {
        self.ctx = ctx
    }

    deinit {
        _ = mlx_metal_icb_recorder_free(ctx)
    }
}
