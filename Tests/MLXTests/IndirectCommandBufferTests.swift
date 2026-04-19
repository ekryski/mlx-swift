// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import XCTest

final class IndirectCommandBufferTests: XCTestCase {

    func testSupportReportedTrueOnAppleSilicon() {
        #if !canImport(Metal)
        throw XCTSkip("requires Metal")
        #endif
        XCTAssertTrue(IndirectCommandBuffer.isSupported)
    }

    func testRecordReplayEmptyBlockProducesZeroCommands() throws {
        let icb = try IndirectCommandBuffer.record {
            // no dispatches
        }
        XCTAssertEqual(icb.totalCommands, 0)
    }

    /// Records a trivial mlx computation via the encoder and replays it.
    /// Since mlx's own `eval` doesn't cooperate with recording mid-graph,
    /// this test only validates that `record`/`replay` succeed — the
    /// numerical parity check against a real primitive lives in
    /// mlx-swift-lm (Phase 4).
    func testRecordReplayRoundTrip() throws {
        let icb = try IndirectCommandBuffer.record {
            // Intentional no-op; a real user records a decoder layer's
            // worth of dispatches here.
        }
        // Replay is a no-op when no commands were recorded. Should not
        // throw or affect state.
        icb.replay()
        XCTAssertEqual(icb.numSegments, 1)  // one empty segment
        XCTAssertEqual(icb.totalCommands, 0)
    }

    // MARK: - Named bindings

    func testBindingNameFNV1aDeterministic() {
        // Same string → same rawValue, across instances and invocations.
        XCTAssertEqual(BindingName("input").rawValue, BindingName("input").rawValue)
        XCTAssertNotEqual(BindingName("input").rawValue,
                          BindingName("output").rawValue)
        // String-literal init produces the same ID as the explicit init.
        let literal: BindingName = "k_cache"
        XCTAssertEqual(literal.rawValue, BindingName("k_cache").rawValue)
    }

    /// Recording with a tagger plus replaying with overrides on an
    /// empty block must succeed without touching the C++ tag tables
    /// incorrectly. Exercises the Swift-side parallel-array packing
    /// and the C FFI signature.
    func testRecordWithBindingsAndReplayOverridesEmptyBlock() throws {
        let x = MLXArray([1.0, 2.0, 3.0] as [Float])
        let y = MLXArray([4.0, 5.0, 6.0] as [Float])

        let icb = try IndirectCommandBuffer.recordWithBindings { tagger in
            // Tagging before any dispatch registers a pending tag — no
            // immediate matches, no subsequent binds (empty block), so
            // the tag never resolves. The replay override with an
            // unknown name must be a no-op.
            tagger.tag(x, as: "input")
        }
        XCTAssertEqual(icb.totalCommands, 0)

        // Override mapping with the same name — the recorder's tag list
        // may be empty (pending never fired), so this must be a no-op,
        // not a crash.
        icb.replay(overrides: ["input": y])
        icb.replay(overrides: [:])  // empty overrides → plain replay path
    }
}
