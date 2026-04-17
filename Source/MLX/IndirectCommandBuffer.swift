// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

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
        var recorder = mlx_metal_icb_recorder(ctx: nil)
        let rc = mlx_metal_icb_end_recording(stream.ctx, &recorder)
        if rc != 0 {
            // end_recording failed (e.g. partial command pending). Free
            // any partial recorder and rethrow.
            _ = mlx_metal_icb_recorder_free(recorder)
            if let threw { try { throw threw }() }
            preconditionFailure(
                "mlx_metal_icb_end_recording failed while recording had no user error")
        }
        if let threw { try { throw threw }() }
        return IndirectCommandBuffer(ctx: recorder)
    }

    /// Replay the captured sequence on `stream`. Each segment's
    /// referenced buffers are made resident via `useResource` and its
    /// commands are issued via `executeCommandsInBuffer`, with a
    /// memory barrier between consecutive segments.
    public func replay(stream: StreamOrDevice = .default) {
        mlx_metal_icb_replay(stream.ctx, ctx)
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
