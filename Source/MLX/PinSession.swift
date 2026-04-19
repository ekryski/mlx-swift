// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation

/// Stable-address allocator reuse for decode-loop Indirect Command
/// Buffer replay (Option "b").
///
/// During a `PinSession`'s **record** phase, every `MLXArray.set_data`
/// call on the current thread captures its underlying storage into a
/// session-owned slot list. During a subsequent **replay** phase, the
/// same set_data calls reuse the captured slots in order — so every
/// output MLXArray ends up pinned at the same `MTL::Buffer` address
/// it had on the recording pass.
///
/// Combined with a recorded `IndirectCommandBuffer`, this eliminates
/// the need for per-binding overrides: every recorded dispatch's
/// bindings remain valid because the allocator's reuse pattern
/// guarantees stable addresses across replays. Non-AB primitives
/// (those whose outputs aren't tagged individually) just-work.
///
/// Typical usage:
///
/// ```swift
/// // Record pass — runs once, after live warmup.
/// let session = PinSession()
/// try session.record {
///     eval(previous)                                 // input stable
///     let icb = try IndirectCommandBuffer.record { model(...); eval(...) }
/// }
///
/// // Each subsequent step — re-run the forward under replay so every
/// // allocation pins at record-time addresses, then replay the ICB.
/// try session.replay { model(...); eval(...) }
/// icb.replay()
/// eval(outputs)
/// ```
///
/// The session must outlive the replays that use it. Free is automatic
/// on deinit; if the iterator drops it, the next replay will fail with
/// dangling MTLBuffers.
///
/// **Thread affinity**: a session is bound to the thread that invoked
/// `record(_:)` / `replay(_:)` for the duration of that call. Different
/// threads can hold separate sessions concurrently.
public final class PinSession: @unchecked Sendable {
    public enum Phase: Sendable {
        case idle
        case record
        case replay
    }

    private var ctx: mlx_pin_session

    /// Slots captured during record. Non-zero only after `record(_:)`
    /// returns.
    public var slotCount: Int {
        var out: Int = 0
        _ = mlx_pin_session_slot_count(ctx, &out)
        return out
    }

    public init() {
        self.ctx = mlx_pin_session(ctx: nil)
    }

    deinit {
        if ctx.ctx != nil {
            _ = mlx_pin_session_free(ctx)
        }
    }

    /// Run `block` with the session in Record phase. Every MLXArray
    /// allocation on the current thread during `block` is captured
    /// into the session.
    ///
    /// Single-shot: calling `record(_:)` a second time replaces the
    /// captured slots (freeing the previous session). Typical
    /// decode-loop use records once then replays many times.
    public func record(_ block: () throws -> Void) rethrows {
        if ctx.ctx != nil {
            _ = mlx_pin_session_free(ctx)
            ctx = mlx_pin_session(ctx: nil)
        }
        var new = mlx_pin_session(ctx: nil)
        _ = mlx_pin_session_begin_record(&new)
        ctx = new

        var threw: Error? = nil
        do {
            try block()
        } catch {
            threw = error
        }

        var slotCount: Int = 0
        _ = mlx_pin_session_end_record(ctx, &slotCount)

        if let threw {
            try { throw threw }()
        }
    }

    /// Run `block` with the session in Replay phase. Each MLXArray
    /// `set_data` call reuses the correspondingly-indexed slot
    /// captured during `record(_:)`.
    public func replay(_ block: () throws -> Void) rethrows {
        _ = mlx_pin_session_begin_replay(ctx)

        var threw: Error? = nil
        do {
            try block()
        } catch {
            threw = error
        }

        var consumed: Int = 0
        _ = mlx_pin_session_end_replay(&consumed)

        if let threw {
            try { throw threw }()
        }
    }
}
