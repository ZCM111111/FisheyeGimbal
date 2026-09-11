import Foundation
import CoreGraphics
import simd

/// Finds the maimai cabinet by its lit button ring.
///
/// The first attempt here looked for the brightest blob in the frame, and real
/// footage killed it outright: in one shot a lit wall won, in another the white
/// cabinet shell did. What survives across idle and mid-song frames is the ring
/// of eight lit buttons — saturated, always on, and arranged in a circle. Fitting
/// a circle through their centres both finds the cabinet and proves it was found,
/// because clutter that happens to be violet does not sit on a ring.
///
/// Thresholds below are the ones measured against real photos; the violet test
/// is deliberately loose and the shape gate is what rejects the noise.
enum CabinetDetector {

    struct Result {
        var found = false
        /// Centre offset from the frame centre, normalised by the short side.
        var centerX: Float = 0
        var centerY: Float = 0
        /// The same centre in source pixels, ready for the lens model.
        var centerPixel = SIMD2<Float>(0, 0)
        /// Ring radius, normalised by the short side.
        var radius: Float = 0
        var blobs = 0
        var spread: Float = 0
        var coverage: Float = 0
        var summary = ""
    }

    // Measured on real footage: min(Cr-128, Cb-128) > 4 and Y > 90 found the
    // buttons in every frame tested.
    private static let violetThreshold: Float = 4
    private static let lumaFloor: Float = 90
    private static let minimumBlobArea = 4
    private static let maximumBlobArea = 1400
    private static let minimumBlobs = 5
    private static let maximumSpread: Float = 0.28
    private static let minimumCoverage: Float = 200

    static func detect(grid: FrameGrid, spreadLimit: Float = maximumSpread) -> Result {
        var result = Result()
        let width = grid.width
        let height = grid.height
        guard width > 24, height > 24 else {
            result.summary = "画面太小，读不出数据"
            return result
        }
        let shortSide = Float(min(width, height))

        // Violet mask.
        var mask = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                mask[index] = grid.violetness(x, y) > violetThreshold
                    && grid.lum[index] > lumaFloor
            }
        }
        mask = open(mask, width: width, height: height)

        // Blob centroids, 8-connected: the buttons are solid patches.
        var visited = [Bool](repeating: false, count: width * height)
        var stack: [Int] = []
        var centers: [SIMD2<Float>] = []
        for start in 0..<(width * height) where mask[start] && !visited[start] {
            stack.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true
            var sumX: Float = 0
            var sumY: Float = 0
            var count = 0
            while let index = stack.popLast() {
                let x = index % width
                let y = index / width
                sumX += Float(x)
                sumY += Float(y)
                count += 1
                for dy in -1...1 {
                    for dx in -1...1 where !(dx == 0 && dy == 0) {
                        let nx = x + dx
                        let ny = y + dy
                        guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                        let neighbour = ny * width + nx
                        guard mask[neighbour], !visited[neighbour] else { continue }
                        visited[neighbour] = true
                        stack.append(neighbour)
                    }
                }
            }
            if count >= minimumBlobArea, count <= maximumBlobArea {
                centers.append(SIMD2<Float>(sumX / Float(count), sumY / Float(count)))
            }
        }

        result.blobs = centers.count
        guard centers.count >= minimumBlobs else {
            result.summary = "只找到 \(centers.count) 个按键色块，对着机台再试"
            return result
        }

        // Circle fit with iterative outlier rejection: one stray violet object
        // is enough to spoil a plain least-squares fit, and rejecting it is what
        // turns 10/12 frames into 12/12.
        var points = centers
        var center = SIMD2<Float>(Float(width) * 0.5, Float(height) * 0.5)
        var radius: Float = 0
        for _ in 0..<3 {
            guard let fitted = fitCircle(points) else {
                result.summary = "圆环拟合失败"
                return result
            }
            center = fitted.center
            radius = fitted.radius
            let distances = points.map { simd_distance($0, center) }
            let mean = distances.reduce(0, +) / Float(distances.count)
            let tolerance = max(0.28 * mean, 3)
            let kept = zip(points, distances)
                .filter { abs($0.1 - mean) <= tolerance }
                .map { $0.0 }
            if kept.count == points.count || kept.count < minimumBlobs { break }
            points = kept
        }

        let distances = points.map { simd_distance($0, center) }
        let mean = distances.reduce(0, +) / Float(distances.count)
        let variance = distances.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
            / Float(distances.count)
        let spread = mean > 0 ? variance.squareRoot() / mean : 1

        // Angular coverage: violet clutter all in one corner must not pass.
        var angles = points.map { atan2($0.y - center.y, $0.x - center.x) }
        angles.sort()
        var biggestGap: Float = 0
        for index in 0..<angles.count {
            let next = angles[(index + 1) % angles.count]
            let gap = index == angles.count - 1
                ? (next + 2 * .pi) - angles[index]
                : next - angles[index]
            biggestGap = max(biggestGap, gap)
        }
        let coverage = 360 - biggestGap * 180 / .pi

        result.centerX = (center.x - Float(width) * 0.5) / shortSide
        result.centerY = (center.y - Float(height) * 0.5) / shortSide
        let scaleX = Float(grid.sourceSize.width) / Float(width)
        let scaleY = Float(grid.sourceSize.height) / Float(height)
        result.centerPixel = SIMD2<Float>(center.x * scaleX, center.y * scaleY)
        result.radius = radius / shortSide
        result.spread = spread
        result.coverage = coverage
        result.blobs = points.count

        guard spread <= spreadLimit else {
            result.summary = String(format: "按键半径离散度 %.2f 太大，不像圆环", Double(spread))
            return result
        }
        guard coverage >= minimumCoverage else {
            result.summary = String(format: "按键只覆盖 %.0f° 的圆环", Double(coverage))
            return result
        }

        result.found = true
        result.summary = String(format: "按键 %d 个 · 偏移 (%+.3f, %+.3f) · 环半径 %.2f · 离散 %.2f",
                                points.count,
                                Double(result.centerX), Double(result.centerY),
                                Double(result.radius), Double(spread))
        return result
    }

    /// 3x3 open: drops single-pixel chroma noise before blob analysis.
    private static func open(_ mask: [Bool], width: Int, height: Int) -> [Bool] {
        var eroded = mask
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let index = y * width + x
                guard mask[index] else { continue }
                var all = true
                for dy in -1...1 {
                    for dx in -1...1 where !mask[(y + dy) * width + (x + dx)] {
                        all = false
                    }
                }
                eroded[index] = all
            }
        }
        var dilated = eroded
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let index = y * width + x
                guard !eroded[index] else { continue }
                var any = false
                for dy in -1...1 {
                    for dx in -1...1 where eroded[(y + dy) * width + (x + dx)] {
                        any = true
                    }
                }
                dilated[index] = any
            }
        }
        return dilated
    }

    private static func fitCircle(_ points: [SIMD2<Float>])
        -> (center: SIMD2<Float>, radius: Float)? {
        guard points.count >= 4 else { return nil }
        var sx: Float = 0, sy: Float = 0
        var sxx: Float = 0, syy: Float = 0, sxy: Float = 0
        var sxz: Float = 0, syz: Float = 0, sz: Float = 0
        for p in points {
            let z = p.x * p.x + p.y * p.y
            sx += p.x
            sy += p.y
            sxx += p.x * p.x
            syy += p.y * p.y
            sxy += p.x * p.y
            sxz += p.x * z
            syz += p.y * z
            sz += z
        }
        let n = Float(points.count)
        let m = [[2 * sxx, 2 * sxy, sx],
                 [2 * sxy, 2 * syy, sy],
                 [2 * sx, 2 * sy, n]]
        let rhs: [Float] = [sxz, syz, sz]
        let det = determinant(m)
        guard abs(det) > 1e-6 else { return nil }
        let a = determinant([[rhs[0], m[0][1], m[0][2]],
                             [rhs[1], m[1][1], m[1][2]],
                             [rhs[2], m[2][1], m[2][2]]]) / det
        let b = determinant([[m[0][0], rhs[0], m[0][2]],
                             [m[1][0], rhs[1], m[1][2]],
                             [m[2][0], rhs[2], m[2][2]]]) / det
        let c = determinant([[m[0][0], m[0][1], rhs[0]],
                             [m[1][0], m[1][1], rhs[1]],
                             [m[2][0], m[2][1], rhs[2]]]) / det
        let value = c + a * a + b * b
        guard value > 0, value.isFinite else { return nil }
        return (SIMD2<Float>(a, b), value.squareRoot())
    }

    private static func determinant(_ m: [[Float]]) -> Float {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}
