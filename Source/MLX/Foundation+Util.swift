// Copyright © 2024 Apple Inc.

import Foundation

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

/// Maximum number of Int32 elements to stack-allocate for shape/axes conversion.
/// ML models rarely exceed 8 dimensions.
@usableFromInline
let _maxStackInt32 = 8

extension [Int] {

    /// Convert `[Int]` to a temporary `[Int32]` buffer, using stack allocation
    /// for small arrays (up to 8 elements) and heap allocation otherwise.
    /// This avoids the heap allocation overhead of `asInt32` for the common case
    /// of shape and axes arrays in ML operations.
    @inlinable
    func withInt32Buffer<R>(_ body: (UnsafeBufferPointer<Int32>) throws -> R) rethrows -> R {
        let n = count
        if n <= _maxStackInt32 {
            return try withUnsafeTemporaryAllocation(of: Int32.self, capacity: _maxStackInt32) {
                buf in
                for i in 0 ..< n {
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

extension Int {

    /// Convenience to convert `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var int32: Int32 { Int32(self) }

    @inlinable
    var int64: Int64 { Int64(self) }
}
