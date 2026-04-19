// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Applies instance normalization [1] on the inputs.
///
/// ### References
/// 1. [https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022)
///
/// ### See also
/// - <doc:normalization>
open class InstanceNorm: Module, UnaryLayer {

    public let dimensions: Int
    public let eps: Float

    public let weight: MLXArray?
    public let bias: MLXArray?

    /// Applies instance normalization [1] on the inputs.
    ///
    /// See [InstanceNorm python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.InstanceNorm.html) for more information.
    ///
    /// ### References
    /// 1. [https://arxiv.org/abs/1607.08022](https://arxiv.org/abs/1607.08022)
    ///
    /// - Parameters:
    ///   - dimensions: number of features in the input
    ///   - eps: value added to the denominator for numerical stability
    ///   - affine: if `true` adds a trainable `weight` and `bias`
    public init(dimensions: Int, eps: Float = 1e-5, affine: Bool = false) {
        self.dimensions = dimensions
        self.eps = eps

        if affine {
            self.weight = MLXArray.ones([dimensions])
            self.bias = MLXArray.zeros([dimensions])
        } else {
            self.weight = nil
            self.bias = nil
        }
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let reductionAxes = Array(1 ..< x.ndim - 1)

        // compute stats
        let mean = mean(x, axes: reductionAxes, keepDims: true)
        let variance = variance(x, axes: reductionAxes, keepDims: true)

        // normalize
        let x = (x - mean) * rsqrt(variance + eps)

        // scale and shift if necessary
        if let weight, let bias {
            return weight * x + bias
        } else {
            return x
        }
    }
}

/// Applies layer normalization [1] on the inputs.
///
/// ### References
/// 1. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
///
/// ### See also
/// - <doc:normalization>
open class LayerNorm: Module, UnaryLayer {

    public let dimensions: Int
    public let eps: Float

    public let weight: MLXArray?
    public let bias: MLXArray?

    /// Applies layer normalization [1] on the inputs.
    ///
    /// See [LayerNorm python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.LayerNorm.html) for more information.
    ///
    /// ### References
    /// 1. [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
    ///
    /// - Parameters:
    ///   - dimensions: number of features in the input
    ///   - eps: value added to the denominator for numerical stability
    ///   - affine: if `true` adds a trainable `weight`
    ///   - bias: if `true` adds a trainable `bias`
    public init(
        dimensions: Int, eps: Float = 1e-5, affine: Bool = true, bias: Bool = true
    ) {
        self.dimensions = dimensions
        self.eps = eps

        if affine {
            self.weight = MLXArray.ones([dimensions])
            self.bias = bias ? MLXArray.zeros([dimensions]) : nil
        } else {
            self.weight = nil
            self.bias = nil
        }
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

/// Applies Root Mean Square normalization [1] to the inputs.
///
/// Concretely:
///
/// ```swift
/// weight * x * MLX.rsqrt(x.square().mean() + eps)
/// ```
///
/// where `weight` is initialized with ones and `eps` is a small float to
/// ensure the numerical stability of inverse square root.
///
/// ### References
/// 1. [https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
///
/// ### See also
/// - <doc:normalization>
open class RMSNorm: Module, UnaryLayer {

    public let weight: MLXArray
    public let eps: Float

    // Optional persistent argument buffer for decode-loop ICB replay.
    // Lazy-initialized on the first call when MLX_PERSISTENT_AB=1 is
    // set. One handle per module instance (so one per layer). Static
    // scalars (axis_size, w_stride, eps) are written once at init;
    // buffer pointers are updated per call by mlx C++.
    //
    // Not Sendable and not part of Module's parameter set — purely
    // eval-state for the AB path.
    private var persistentAb: PersistentRmsAbHandle?

    public init(dimensions: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    /// Describe `dimensions` and `eps`.
    public override func describeExtra(_ indent: Int) -> String {
        "(dimensions=\(weight.dim(0)), eps=\(self.eps))"
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Self.persistentAbEnabled {
            let handle = persistentAbHandle(hidden: weight.dim(0))
            return MLXFast.rmsNormAb(
                x, weight: weight, eps: eps, handle: handle)
        }
        return MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }

    /// Cached env-var check — MLX_PERSISTENT_AB=1 activates the
    /// handle-backed RMSNorm path. Gated off by default so existing
    /// callers see no behavior change.
    private static let persistentAbEnabled: Bool = {
        let env = ProcessInfo.processInfo.environment["MLX_PERSISTENT_AB"]
        return env == "1"
    }()

    /// Lazy-initialize the handle on first call and write the static
    /// scalars. `hidden` = last dimension of `weight` (axis_size).
    /// `w_stride` is always 1 for this Module (weight is 1D, unit
    /// stride).
    private func persistentAbHandle(hidden: Int) -> PersistentRmsAbHandle {
        if let existing = persistentAb {
            return existing
        }
        let handle = PersistentRmsAbHandle()
        handle.setScalar32(slot: .axisSize, value: UInt32(hidden))
        handle.setScalar32(slot: .wStride, value: 1)
        handle.setFloat32(slot: .eps, value: eps)
        persistentAb = handle
        return handle
    }
}

/// Applies Group Normalization [1] on the inputs.
///
/// ### References
/// 1. [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
///
/// ### See also
/// - <doc:normalization>
open class GroupNorm: Module, UnaryLayer {

    public let groupCount: Int
    public let dimensions: Int
    public let eps: Float
    public let pytorchCompatible: Bool

    public let weight: MLXArray?
    public let bias: MLXArray?

    /// Applies Group Normalization [1] on the inputs.
    ///
    /// See [GroupNorm python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.GroupNorm.html) for more information.
    ///
    /// The feature dimension is assumed to be the last dimension and the dimensions
    /// that precede it (except the first) are considered the spatial dimensions.
    ///
    /// ### References
    /// 1. [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)
    ///
    /// - Parameters:
    ///   - groupCount: number of groups to separate the features into
    ///   - dimensions: number of features in the input
    ///   - eps: value added to the denominator for numerical stability
    ///   - affine: if `true` adds a trainable `weight` and `bias`
    ///   - pytorchCompatible: if `true` perform the group normalization in the same
    ///     order/grouping as PyTorch
    public init(
        groupCount: Int, dimensions: Int, eps: Float = 1e-5, affine: Bool = true,
        pytorchCompatible: Bool = false
    ) {
        self.groupCount = groupCount
        self.dimensions = dimensions
        self.eps = eps
        self.pytorchCompatible = pytorchCompatible

        if affine {
            self.weight = MLXArray.ones([dimensions])
            self.bias = MLXArray.zeros([dimensions])
        } else {
            self.weight = nil
            self.bias = nil
        }
    }

    func pytorchGroupNorm(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let dims = x.dim(-1)
        let rest = x.shape.dropFirst().dropLast()
        let groupSize = dims / groupCount

        // split into groups
        var x = x.reshaped(batch, -1, groupCount, groupSize)
        x = x.transposed(0, 2, 1, 3).reshaped(batch, groupCount, -1)

        // normalize
        x = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)

        x = x.reshaped(batch, groupCount, -1, groupSize)
        x = x.transposed(0, 2, 1, 3).reshaped([batch] + rest + [dims])

        return x
    }

    func groupNorm(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let dims = x.dim(-1)
        let rest = x.shape.dropFirst().dropLast()

        // split into groups
        var x = x.reshaped(batch, -1, groupCount)

        // normalize
        let means = mean(x, axis: 1, keepDims: true)
        let variance = variance(x, axis: 1, keepDims: true)
        x = (x - means) * rsqrt(variance + eps)
        x = x.reshaped([batch] + rest + [dims])

        return x
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = pytorchCompatible ? pytorchGroupNorm(x) : groupNorm(x)
        if let weight, let bias {
            return weight * x + bias
        } else {
            return x
        }
    }
}

/// Applies batch normalization [1] on the inputs.
///
/// ### References
/// 1. [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
///
/// ### See also
/// - <doc:normalization>
open class BatchNorm: Module, UnaryLayer {

    public let featureCount: Int
    public let eps: Float
    public let momentum: Float

    public let weight: MLXArray?
    public let bias: MLXArray?

    @ParameterInfo(key: "running_mean") var runningMean: MLXArray?
    @ParameterInfo(key: "running_var") var runningVar: MLXArray?

    /// Applies batch normalization [1] on the inputs.
    ///
    /// See [BatchNorm python docs](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.BatchNorm.html) for more information.
    ///
    /// The input shape is specified as `NC` or `NLC`, where `N` is the
    /// batch, `C` is the number of features or channels, and `L` is the
    /// sequence length. The output has the same shape as the input. For
    /// four-dimensional arrays, the shape is `NHWC`, where `H` and `W` are
    /// the height and width respectively.
    ///
    /// For more information on Batch Normalization, see the original paper "Batch
    /// Normalization: Accelerating Deep Network Training by Reducing Internal
    /// Covariate Shift" <https://arxiv.org/abs/1502.03167>.
    ///
    /// ### References
    /// 1. [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
    ///
    /// - Parameters:
    ///   - featureCount: number of features in the input
    ///   - eps: value added to the denominator for numerical stability
    ///   - momentum: momentum for updating the running mean and variance
    ///   - affine: if `true` adds a trainable `weight` and `bias`
    ///   - trackRunningStats: if `true` track the running mean and variance
    public init(
        featureCount: Int, eps: Float = 1e-5, momentum: Float = 0.1, affine: Bool = true,
        trackRunningStats: Bool = true
    ) {
        self.featureCount = featureCount
        self.eps = eps
        self.momentum = momentum

        if affine {
            self.weight = MLXArray.ones([featureCount])
            self.bias = MLXArray.zeros([featureCount])
        } else {
            self.weight = nil
            self.bias = nil
        }

        if trackRunningStats {
            self._runningMean.wrappedValue = MLXArray.zeros([featureCount])
            self._runningVar.wrappedValue = MLXArray.ones([featureCount])
        }

        super.init()

        if trackRunningStats {
            self.freeze(recursive: false, keys: ["running_mean", "running_var"])
        }
    }

    public override func unfreeze(
        recursive: Bool = true, keys: [String]? = nil, strict: Bool = false
    ) throws {
        try super.unfreeze(recursive: recursive, keys: keys, strict: strict)
        self.freeze(recursive: false, keys: ["running_mean", "running_var"])
    }

    func stats(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let reductionAxes = Array(0 ..< x.ndim - 1)

        let mean = mean(x, axes: reductionAxes)
        let variance = variance(x, axes: reductionAxes)

        return (mean, variance)
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition((2 ... 4).contains(x.ndim))

        // Calculate the mean and variance used to normalize the input x. If we
        // are in training mode update the running stats if needed.

        var (mean, variance) = stats(x)

        if self.training, let runningMean, let runningVar {
            let mu = momentum
            runningMean._updateInternal((1 - mu) * runningMean + mu * mean)
            runningVar._updateInternal((1 - mu) * runningVar + mu * variance)

        } else if let runningMean, let runningVar {
            mean = runningMean
            variance = runningVar
        }

        let x = (x - mean) * rsqrt(variance + eps)

        if let weight, let bias {
            return weight * x + bias
        } else {
            return x
        }
    }
}
