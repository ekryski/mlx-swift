// Copyright © 2024 Apple Inc.

import Foundation

/// Maximum number of elements for stack-allocated Int32 conversion buffers.
/// Shapes and axes in ML models rarely exceed 8 dimensions.
@usableFromInline
let _maxStackInt32 = 8

extension [Int] {

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt64: [Int64] {
        self.map { Int64($0) }
    }

    /// Call `body` with a temporary `UnsafeBufferPointer<Int32>` containing this array's
    /// elements converted to Int32. For small arrays (up to 8 elements), uses stack allocation
    /// to avoid heap overhead. The pointer is only valid for the duration of `body`.
    @inlinable
    func withInt32Buffer<R>(_ body: (UnsafeBufferPointer<Int32>) throws -> R) rethrows -> R {
        let n = count
        if n <= _maxStackInt32 {
            return try withUnsafeTemporaryAllocation(of: Int32.self, capacity: _maxStackInt32) { buf in
                for i in 0..<n {
                    buf[i] = Int32(self[i])
                }
                return try body(UnsafeBufferPointer(start: buf.baseAddress, count: n))
            }
        } else {
            let converted = self.asInt32
            return try converted.withUnsafeBufferPointer(body)
        }
    }
}

extension Sequence<Int> {

    /// Convenience to coerce  sequence of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }

    @inlinable
    var asInt64: [Int64] {
        self.map { Int64($0) }
    }
}

extension Int {

    /// Convenience to convert `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var int32: Int32 { Int32(self) }

    @inlinable
    var int64: Int64 { Int64(self) }
}
