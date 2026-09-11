import Foundation
import CoreGraphics
import simd

/// Finds the maimai cabinet in a frame.
///
/// No machine learning here: the cabinet is a large lit disc in a dim arcade,
/// which is exactly the case Otsu's method plus a connected component handles
/// well. The disc's centre and semi-axes come straight out of the second
/// moments of the region, so there is nothing iterative to go wrong.
///
/// What the centre is used for is alignment, not distortion correction: a wide
/// lens stretches whatever sits away from the middle of the frame, so a cabinet
/// that is off-centre looks distorted no matter how round its screen is. Putting
/// its centre on the optical axis is what removes that.
enum CabinetDetector {

    struct Result {
        var found = false
        /// Centre offset from the frame centre, normalised by the short side.
        var centerX: Float = 0
        var centerY: Float = 0
        /// The same centre in source pixels, ready for the lens model.
        var centerPixel = SIMD2<Float>(0, 0)
        /// Semi-axes normalised by the short side.
        var radiusX: Float = 0
        var radiusY: Float = 0
        var angle: Float = 0
        /// Region area over the area of a matching ellipse: 1 means a clean disc.
        var fill: Float = 0
        var areaFraction: Float = 0
        var summary = ""
    }

    static func detect(grid: LensCircleMeasurer.LumaGrid) -> Result {
        var result = Result()
        let width = grid.width
        let height = grid.height
        guard width > 24, height > 24 else {
            result.summary = "画面太小，读不出数据"
            return result
        }
        let shortSide = Float(min(width, height))

        let threshold = otsuThreshold(grid.lum)
        guard threshold > 4 else {
            result.summary = "画面太暗或没有明暗差异，找不到机台"
            return result
        }

        // Largest bright region.
        var visited = [Bool](repeating: false, count: width * height)
        var best: [Int] = []
        var stack: [Int] = []
        for start in 0..<(width * height) where !visited[start] && grid.lum[start] >= threshold {
            stack.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true
            var region: [Int] = []
            while let index = stack.popLast() {
                region.append(index)
                let x = index % width
                let y = index / width
                // 4-connected: diagonal leaks would join the cabinet to a
                // neighbouring bright panel.
                if x > 0 { push(index - 1, &stack, &visited, grid.lum, threshold) }
                if x < width - 1 { push(index + 1, &stack, &visited, grid.lum, threshold) }
                if y > 0 { push(index - width, &stack, &visited, grid.lum, threshold) }
                if y < height - 1 { push(index + width, &stack, &visited, grid.lum, threshold) }
            }
            if region.count > best.count { best = region }
        }

        let total = Float(width * height)
        guard !best.isEmpty else {
            result.summary = "没找到发亮的区域，对着机台再试"
            return result
        }
        let areaFraction = Float(best.count) / total
        guard areaFraction > 0.02 else {
            result.summary = "亮区太小，可能是远处的灯而不是机台"
            return result
        }
        guard areaFraction < 0.60 else {
            result.summary = "亮区几乎占满画面，判断不出机台边界"
            return result
        }

        // First and second moments.
        var sumX: Float = 0, sumY: Float = 0
        for index in best {
            sumX += Float(index % width)
            sumY += Float(index / width)
        }
        let count = Float(best.count)
        let meanX = sumX / count, meanY = sumY / count
        var mu20: Float = 0, mu02: Float = 0, mu11: Float = 0
        for index in best {
            let dx = Float(index % width) - meanX
            let dy = Float(index / width) - meanY
            mu20 += dx * dx
            mu02 += dy * dy
            mu11 += dx * dy
        }
        mu20 /= count
        mu02 /= count
        mu11 /= count

        // Eigen-decomposition of the 2x2 covariance: for a filled ellipse the
        // variance along an axis is (semi-axis)^2 / 4.
        let trace = mu20 + mu02
        let delta = ((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11).squareRoot()
        let lambda1 = max((trace + delta) * 0.5, 0.0001)
        let lambda2 = max((trace - delta) * 0.5, 0.0001)
        let semiMajor = 2 * lambda1.squareRoot()
        let semiMinor = 2 * lambda2.squareRoot()
        let angle = 0.5 * atan2(2 * mu11, mu20 - mu02)

        let ellipseArea = Float.pi * semiMajor * semiMinor
        let fill = ellipseArea > 0 ? count / ellipseArea : 0

        result.centerX = (meanX - Float(width) * 0.5) / shortSide
        result.centerY = (meanY - Float(height) * 0.5) / shortSide
        let scaleX = Float(grid.sourceSize.width) / Float(width)
        let scaleY = Float(grid.sourceSize.height) / Float(height)
        result.centerPixel = SIMD2<Float>(meanX * scaleX, meanY * scaleY)
        result.radiusX = semiMajor / shortSide
        result.radiusY = semiMinor / shortSide
        result.angle = angle
        result.fill = fill
        result.areaFraction = areaFraction

        guard fill > 0.72 else {
            result.summary = String(format: "亮区形状不规整（填充率 %.2f），不像机台圆盘", Double(fill))
            return result
        }
        guard semiMinor / max(semiMajor, 1) > 0.45 else {
            result.summary = "亮区太扁，看看是不是斜着拍的"
            return result
        }

        result.found = true
        result.summary = String(format: "机台圆心偏移 (%+.3f, %+.3f) · 半径 %.2f · 填充 %.2f",
                                Double(result.centerX), Double(result.centerY),
                                Double(result.radiusX), Double(fill))
        return result
    }

    private static func push(_ index: Int,
                             _ stack: inout [Int],
                             _ visited: inout [Bool],
                             _ lum: [Float],
                             _ threshold: Float) {
        guard !visited[index], lum[index] >= threshold else { return }
        visited[index] = true
        stack.append(index)
    }

    /// Otsu's method: the threshold that best separates the histogram into two
    /// groups, which suits a lit screen against a dim room.
    private static func otsuThreshold(_ lum: [Float]) -> Float {
        var histogram = [Int](repeating: 0, count: 256)
        var minimum: Float = 255, maximum: Float = 0
        for value in lum {
            let clamped = min(max(value, 0), 255)
            histogram[Int(clamped)] += 1
            minimum = min(minimum, clamped)
            maximum = max(maximum, clamped)
        }
        let total = lum.count
        guard total > 0, maximum - minimum > 8 else { return 0 }

        var sum: Double = 0
        for index in 0..<256 { sum += Double(index * histogram[index]) }

        var sumBackground: Double = 0
        var weightBackground = 0
        var bestVariance: Double = 0
        var bestThreshold = Int(minimum)

        for index in 0..<256 {
            weightBackground += histogram[index]
            if weightBackground == 0 { continue }
            let weightForeground = total - weightBackground
            if weightForeground == 0 { break }
            sumBackground += Double(index * histogram[index])
            let meanBackground = sumBackground / Double(weightBackground)
            let meanForeground = (sum - sumBackground) / Double(weightForeground)
            let difference = meanBackground - meanForeground
            let variance = Double(weightBackground) * Double(weightForeground) * difference * difference
            if variance > bestVariance {
                bestVariance = variance
                bestThreshold = index
            }
        }
        return Float(bestThreshold)
    }
}
