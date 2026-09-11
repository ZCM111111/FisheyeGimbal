import Foundation
import CoreGraphics
import simd

/// Finds the maimai cabinet by the circular boundary of its screen.
///
/// Colour is deliberately not used anywhere here. The buttons are RGB LEDs that
/// change with the song, so any hue test works only by luck; the screen's round
/// bezel and the ring of button shapes around it care nothing about colour.
///
/// Gradients find the bezel, a Hough pass proposes circles, and each proposal is
/// scored by how much real edge actually lies along it — a circle with weak
/// support is a guess, not a detection. Measured on 62 real frames from a dim
/// room and an arcade, the accepted circles carried 0.82-0.99 edge support.
enum CabinetDetector {

    struct Result {
        var found = false
        /// Centre offset from the frame centre, normalised by the short side.
        var centerX: Float = 0
        var centerY: Float = 0
        /// The same centre in source pixels, ready for the lens model.
        var centerPixel = SIMD2<Float>(0, 0)
        /// Screen radius in source pixels, for the framing lock.
        var radiusPixel: Float = 0
        /// Screen radius, normalised by the short side.
        var radius: Float = 0
        var support: Float = 0
        var candidates = 0
        var summary = ""
    }

    private static let minRadiusFraction: Float = 0.12
    private static let maxRadiusFraction: Float = 0.78
    private static let radiusStep: Float = 2
    /// Fraction of the circle that must sit on a real edge.
    private static let supportThreshold: Float = 0.55
    /// Radial slack when sampling, because a wide lens bends a circle slightly.
    private static let supportTolerance = 3

    static func detect(grid: FrameGrid, supportLimit: Float = supportThreshold) -> Result {
        var result = Result()
        let width = grid.width
        let height = grid.height
        guard width > 32, height > 32 else {
            result.summary = "画面太小，读不出数据"
            return result
        }

        // Sobel gradients on luminance only.
        var gradientX = [Float](repeating: 0, count: width * height)
        var gradientY = [Float](repeating: 0, count: width * height)
        var magnitude = [Float](repeating: 0, count: width * height)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let tl = grid.lum[(y - 1) * width + (x - 1)]
                let tc = grid.lum[(y - 1) * width + x]
                let tr = grid.lum[(y - 1) * width + (x + 1)]
                let ml = grid.lum[y * width + (x - 1)]
                let mr = grid.lum[y * width + (x + 1)]
                let bl = grid.lum[(y + 1) * width + (x - 1)]
                let bc = grid.lum[(y + 1) * width + x]
                let br = grid.lum[(y + 1) * width + (x + 1)]
                let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
                let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
                let index = y * width + x
                gradientX[index] = gx
                gradientY[index] = gy
                magnitude[index] = (gx * gx + gy * gy).squareRoot()
            }
        }

        let strongEdge = 0.55 * percentile(magnitude, every: 7, fraction: 0.97)
        guard strongEdge > 4 else {
            result.summary = "画面太平，没有可用的边缘"
            return result
        }

        // Edge pixel list, so the Hough pass does not walk the whole frame once
        // per radius.
        var edgePixels: [Int] = []
        edgePixels.reserveCapacity(width * height / 8)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let index = y * width + x
                if magnitude[index] >= strongEdge {
                    edgePixels.append(index)
                }
            }
        }
        guard edgePixels.count > 200 else {
            result.summary = "边缘太少，找不到屏幕边界"
            return result
        }

        // Hough: one radius at a time, voting along the gradient direction.
        let shortSide = Float(min(width, height))
        var accumulator = [Int](repeating: 0, count: width * height)
        var candidates: [(x: Float, y: Float, r: Float)] = []
        var radius = shortSide * minRadiusFraction
        let radiusLimit = shortSide * maxRadiusFraction
        while radius <= radiusLimit {
            for index in 0..<accumulator.count { accumulator[index] = 0 }
            var bestVotes = 0
            var bestIndex = -1
            for index in edgePixels {
                let m = magnitude[index]
                let ux = gradientX[index] / m
                let uy = gradientY[index] / m
                let px = Float(index % width)
                let py = Float(index / width)
                for sign in [Float(1), Float(-1)] {
                    let cx = Int((px + sign * ux * radius).rounded())
                    let cy = Int((py + sign * uy * radius).rounded())
                    guard cx >= 0, cx < width, cy >= 0, cy < height else { continue }
                    let vote = cy * width + cx
                    accumulator[vote] += 1
                    if accumulator[vote] > bestVotes {
                        bestVotes = accumulator[vote]
                        bestIndex = vote
                    }
                }
            }
            if bestIndex >= 0 {
                candidates.append((Float(bestIndex % width), Float(bestIndex / width), radius))
            }
            radius += radiusStep
        }

        result.candidates = candidates.count
        guard !candidates.isEmpty else {
            result.summary = "没有候选圆"
            return result
        }

        // Score every proposal by real edge support and keep the best.
        var bestSupport: Float = 0
        var best: (x: Float, y: Float, r: Float)?
        for candidate in candidates {
            let score = support(magnitude: magnitude,
                                width: width,
                                height: height,
                                cx: candidate.x,
                                cy: candidate.y,
                                r: candidate.r,
                                threshold: strongEdge)
            if score > bestSupport {
                bestSupport = score
                best = candidate
            }
        }
        guard let chosen = best else {
            result.summary = "候选圆都不可用"
            return result
        }
        guard bestSupport >= supportLimit else {
            result.summary = String(format: "屏幕圆边界支持度只有 %.2f，没认出机台",
                                    Double(bestSupport))
            return result
        }

        // Refine the centre on the edge points that actually supported the
        // circle, which also tolerates the slight ellipticity of a wide lens.
        var supported: [SIMD2<Float>] = []
        let samples = 180
        for step in 0..<samples {
            let angle = Float(step) / Float(samples) * 2 * .pi
            let dx = cos(angle)
            let dy = sin(angle)
            var bestValue: Float = 0
            var bestPoint: SIMD2<Float>?
            for offset in -supportTolerance...supportTolerance {
                let x = Int((chosen.x + (chosen.r + Float(offset)) * dx).rounded())
                let y = Int((chosen.y + (chosen.r + Float(offset)) * dy).rounded())
                guard x >= 1, x < width - 1, y >= 1, y < height - 1 else { continue }
                let value = magnitude[y * width + x]
                if value > bestValue {
                    bestValue = value
                    bestPoint = SIMD2<Float>(Float(x), Float(y))
                }
            }
            if let point = bestPoint, bestValue >= strongEdge {
                supported.append(point)
            }
        }

        var center = SIMD2<Float>(chosen.x, chosen.y)
        var ringRadius = chosen.r
        if supported.count >= 24, let fitted = fitCircle(supported) {
            center = fitted.center
            ringRadius = fitted.radius
        }

        result.centerX = (center.x - Float(width) * 0.5) / shortSide
        result.centerY = (center.y - Float(height) * 0.5) / shortSide
        let scaleX = Float(grid.sourceSize.width) / Float(width)
        let scaleY = Float(grid.sourceSize.height) / Float(height)
        result.centerPixel = SIMD2<Float>(center.x * scaleX, center.y * scaleY)
        result.radiusPixel = ringRadius * (scaleX + scaleY) * 0.5
        result.radius = ringRadius / shortSide
        result.support = bestSupport
        result.found = true
        result.summary = String(format: "屏幕圆 · 偏移 (%+.3f, %+.3f) · 半径 %.2f · 边缘支持度 %.2f",
                                Double(result.centerX), Double(result.centerY),
                                Double(result.radius), Double(bestSupport))
        return result
    }

    /// Fraction of the circle where a real edge is present.
    private static func support(magnitude: [Float],
                                width: Int,
                                height: Int,
                                cx: Float,
                                cy: Float,
                                r: Float,
                                threshold: Float) -> Float {
        let samples = 180
        var strong = 0
        var valid = 0
        for step in 0..<samples {
            let angle = Float(step) / Float(samples) * 2 * .pi
            let dx = cos(angle)
            let dy = sin(angle)
            var best: Float = 0
            var found = false
            for offset in -supportTolerance...supportTolerance {
                let x = Int((cx + (r + Float(offset)) * dx).rounded())
                let y = Int((cy + (r + Float(offset)) * dy).rounded())
                guard x >= 0, x < width, y >= 0, y < height else { continue }
                found = true
                best = max(best, magnitude[y * width + x])
            }
            guard found else { continue }
            valid += 1
            if best >= threshold { strong += 1 }
        }
        guard valid > 0 else { return 0 }
        return Float(strong) / Float(valid)
    }

    private static func percentile(_ values: [Float], every step: Int, fraction: Float) -> Float {
        var sample: [Float] = []
        sample.reserveCapacity(values.count / max(step, 1) + 1)
        var index = 0
        while index < values.count {
            sample.append(values[index])
            index += max(step, 1)
        }
        guard !sample.isEmpty else { return 0 }
        sample.sort()
        let position = Int(Float(sample.count - 1) * fraction)
        return sample[min(max(position, 0), sample.count - 1)]
    }

    private static func fitCircle(_ points: [SIMD2<Float>])
        -> (center: SIMD2<Float>, radius: Float)? {
        guard points.count >= 12 else { return nil }
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
