import Foundation
import CoreVideo
import simd

/// Measures where the fisheye image circle sits inside the frame.
///
/// Only the centre is measured, and only the centre is written back to the
/// settings. Distortion coefficients are deliberately left alone: an earlier
/// version also fitted those from straight edges in the scene, and a bad fit
/// there could quietly spoil a working setup. Where the circle sits is an
/// observation, so automating it is safe.
enum LensCircleMeasurer {

    struct LumaGrid {
        let width: Int
        let height: Int
        let lum: [Float]

        func sample(_ x: Float, _ y: Float) -> Float? {
            guard x >= 0, y >= 0, x <= Float(width - 1), y <= Float(height - 1) else {
                return nil
            }
            let x0 = Int(x), y0 = Int(y)
            let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
            let fx = x - Float(x0), fy = y - Float(y0)
            let top = lum[y0 * width + x0] + (lum[y0 * width + x1] - lum[y0 * width + x0]) * fx
            let bottom = lum[y1 * width + x0] + (lum[y1 * width + x1] - lum[y1 * width + x0]) * fx
            return top + (bottom - top) * fy
        }
    }

    struct Result {
        var found = false
        /// Offsets from the frame centre, normalised by the short side — the
        /// same convention `FisheyeSettings` uses.
        var centerX: Float = 0
        var centerY: Float = 0
        var radiusScale: Float = 1
        var raysUsed = 0
        var spread: Float = 1
        var summary = ""
    }

    /// Decimated luminance copy of one frame.
    static func makeGrid(from pixelBuffer: CVPixelBuffer, targetWidth: Int = 256) -> LumaGrid? {
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let sourceWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let sourceHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        guard sourceWidth > 16, sourceHeight > 16, bytesPerRow > 0 else { return nil }

        let planar = CVPixelBufferIsPlanar(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let isBGRA = !planar || pixelFormat == kCVPixelFormatType_32BGRA

        let step = max(1, sourceWidth / max(targetWidth, 32))
        let width = sourceWidth / step
        let height = sourceHeight / step
        guard width > 16, height > 16 else { return nil }

        var lum = [Float](repeating: 0, count: width * height)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            let sourceRow = y * step
            guard sourceRow < sourceHeight else { break }
            let rowStart = sourceRow * bytesPerRow
            for x in 0..<width {
                let column = x * step
                guard column < sourceWidth else { continue }
                if isBGRA {
                    lum[y * width + x] = Float(bytes[rowStart + column * 4 + 1])
                } else {
                    lum[y * width + x] = Float(bytes[rowStart + column])
                }
            }
        }
        return LumaGrid(width: width, height: height, lum: lum)
    }

    static func measure(grid: LumaGrid) -> Result {
        var result = Result()
        let width = Float(grid.width), height = Float(grid.height)
        let shortSide = Float(min(grid.width, grid.height))

        // Percentiles from a subsample, so one bright object cannot define the
        // whole range.
        var samples: [Float] = []
        let stepX = max(1, grid.width / 48), stepY = max(1, grid.height / 64)
        var y = 0
        while y < grid.height {
            var x = 0
            while x < grid.width {
                samples.append(grid.lum[y * grid.width + x])
                x += stepX
            }
            y += stepY
        }
        guard samples.count > 64 else {
            result.summary = "画面太小，读不出数据"
            return result
        }
        samples.sort()
        let dark = samples[samples.count / 20]
        let bright = samples[samples.count * 19 / 20]
        guard bright - dark > 14 else {
            result.summary = "画面明暗对比不足，测不到成像圈边界"
            return result
        }
        let threshold = dark + 0.40 * (bright - dark)

        var center = SIMD2<Float>(width * 0.5, height * 0.5)
        let searchLimit = (width * width + height * height).squareRoot() * 0.72
        let angles = 180
        var radius: Float = 0
        var used = 0
        var spread: Float = 1

        for _ in 0..<4 {
            var radii = [Float](repeating: 0, count: angles)
            var found = [Bool](repeating: false, count: angles)
            for a in 0..<angles {
                let angle = Float(a) / Float(angles) * 2 * .pi
                let dir = SIMD2<Float>(cos(angle), sin(angle))
                var r = searchLimit
                while r > 8 {
                    let p = center + dir * r
                    if p.x >= 0, p.y >= 0, p.x <= width - 1, p.y <= height - 1,
                       let value = grid.sample(p.x, p.y), value >= threshold {
                        radii[a] = r
                        found[a] = true
                        break
                    }
                    r -= 1.5
                }
            }

            let present = radii.enumerated().compactMap { found[$0.offset] ? $0.element : nil }
            guard present.count >= angles / 2 else {
                result.summary = "成像圈边界不清晰：画面边缘太暗或没有可见的暗角"
                return result
            }

            // Reject rays whose boundary disagrees with the median: a dark
            // object at the rim should not drag the centre with it.
            let sorted = present.sorted()
            let median = sorted[sorted.count / 2]
            let tolerance = max(median * 0.22, 4)
            var points: [SIMD2<Float>] = []
            for a in 0..<angles where found[a] && abs(radii[a] - median) <= tolerance {
                let angle = Float(a) / Float(angles) * 2 * .pi
                points.append(center + SIMD2<Float>(cos(angle), sin(angle)) * radii[a])
            }
            guard points.count >= angles / 2 else {
                result.summary = "成像圈边界不规整，测不出可靠圆心"
                return result
            }

            guard let fitted = fitCircle(points) else {
                result.summary = "圆拟合失败，换个角度再试"
                return result
            }
            // Confidence: how tightly the accepted rays agree on one radius.
            var deviations: [Float] = []
            for a in 0..<angles where found[a] && abs(radii[a] - median) <= tolerance {
                deviations.append(abs(radii[a] - median))
            }
            deviations.sort()
            spread = deviations.isEmpty ? 1 : deviations[deviations.count / 2] / max(median, 1)

            center = fitted.center
            radius = fitted.radius
            used = points.count
        }

        guard radius > shortSide * 0.22, radius < searchLimit * 0.99 else {
            result.summary = "没看到成像圈边界：镜头可能覆盖了整个画面"
            return result
        }
        guard spread < 0.12 else {
            result.summary = String(format: "边界太模糊（离散度 %.0f%%），换个光线好一点的地方再试",
                                    Double(spread * 100))
            return result
        }

        result.found = true
        result.centerX = (center.x - width * 0.5) / shortSide
        result.centerY = (center.y - height * 0.5) / shortSide
        result.radiusScale = radius / (shortSide * 0.5)
        result.raysUsed = used
        result.spread = spread
        result.summary = String(format: "圆心偏移 (%+.3f, %+.3f) · 成像圈 %.2f · %d 条边界一致",
                                Double(result.centerX), Double(result.centerY),
                                Double(result.radiusScale), used)
        return result
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
        guard value > 0 else { return nil }
        return (SIMD2<Float>(a, b), value.squareRoot())
    }

    private static func determinant(_ m: [[Float]]) -> Float {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}
