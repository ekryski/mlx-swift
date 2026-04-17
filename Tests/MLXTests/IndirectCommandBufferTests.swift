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
}
