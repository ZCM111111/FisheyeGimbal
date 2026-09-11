import Foundation
import CoreVideo
import simd

/// One-tap lens calibration.
///
/// Two things are measured straight from a live frame, with no user input:
///
///  1. **The image circle** the clip-on glass projects: its centre and radius
///     are found from the radial luminance profile and a least-squares circle
///     fit. That gives the principal point and the circle scale directly.
///
///  2. **The residual radial distortion**: long straight edges in the scene are
///     located, and the lens parameters are searched for the pair that makes
///     those edges come out straight. Straightness is measured in the
///     corrected (rectilinear) plane, so the fit optimises exactly what the eye
///     judges.
///
/// Everything runs on a decimated luminance grid, so a press costs tens of
/// milliseconds of arithmetic instead of a second of pixel work.
enum AutoCalibrator {

    // MARK: - Input / output

    struct Input {
        var lensHalfFovDegrees: Float
        var circleScale: Float
        var centerX: Float
        var centerY: Float
        var k1: Float
        var k2: Float
        var equidistant: Bool
    }

    struct Outcome {
        var circleFound = false
        var centerX: Float = 0
        var centerY: Float = 0
        var circleScale: Float = 1
        var lensHalfFovDegrees: Float = 90
        var k1: Float = 0
        var lines = 0
        var bowBefore: Float = 1
        var bowAfter: Float = 1
        var applied = false
        var summaryKind: SummaryKind = .alreadyStraight
        var summary = ""
    }

    /// Decimated luminance copy of one frame.
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

    // MARK: - Frame to grid

    /// Copies one frame into a decimated luminance grid. Handles both the
    /// biplanar 4:2:0 the camera normally delivers and a BGRA fallback.
    static func makeGrid(from pixelBuffer: CVPixelBuffer, targetWidth: Int = 240) -> LumaGrid? {
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
                    // Green channel is the closest cheap luma proxy.
                    let offset = rowStart + column * 4 + 1
                    lum[y * width + x] = Float(bytes[offset])
                } else {
                    lum[y * width + x] = Float(bytes[rowStart + column])
                }
            }
        }
        return LumaGrid(width: width, height: height, lum: lum)
    }

    // MARK: - Lens model

    private struct Lens {
        var center: SIMD2<Float>
        var maxRadius: Float
        var focal: Float
        var k1: Float
        var k2: Float
        var equidistant: Bool

        /// Output plane (tan units) -> source pixel. Mirrors the shader.
        func sourcePoint(_ v: SIMD2<Float>) -> SIMD2<Float>? {
            let r = simd_length(v)
            guard r > 1e-6 else { return center }
            let theta = atan(r)
            let radiusModel = equidistant
                ? focal * theta
                : 2 * focal * sin(theta * 0.5)
            let nr = radiusModel / max(maxRadius, 1)
            let radius = radiusModel * (1 + k1 * nr * nr + k2 * nr * nr * nr * nr)
            return center + (v / r) * radius
        }

        /// Source pixel -> output plane (tan units), for a candidate correction.
        func outputPoint(_ p: SIMD2<Float>,
                         focalScale: Float,
                         candidateK1: Float) -> SIMD2<Float>? {
            let u = p - center
            let radiusFinal = simd_length(u)
            guard radiusFinal > 1e-6 else { return SIMD2<Float>(0, 0) }
            let dir = u / radiusFinal

            // Invert radius = rm * (1 + k1 * (rm/R)^2 + k2 * (rm/R)^4).
            var rm = radiusFinal
            for _ in 0..<3 {
                let nr = rm / max(maxRadius, 1)
                let factor = 1 + candidateK1 * nr * nr + k2 * nr * nr * nr * nr
                rm = radiusFinal / max(factor, 0.25)
            }

            let f = max(focal * focalScale, 1)
            let theta: Float
            if equidistant {
                theta = rm / f
            } else {
                let ratio = min(max(rm / (2 * f), -1), 1)
                theta = 2 * asin(ratio)
            }
            guard theta > 0, theta < 1.45 else { return nil }
            return dir * tan(theta)
        }
    }

    // MARK: - Analysis

    static func analyze(grid: LumaGrid, input: Input) -> Outcome? {
        var outcome = Outcome()
        let shortSide = Float(min(grid.width, grid.height))

        // 1. Image circle.
        var center = SIMD2<Float>(Float(grid.width) * 0.5, Float(grid.height) * 0.5)
        var maxRadius = shortSide * 0.5 * input.circleScale
        if let circle = detectCircle(grid) {
            outcome.circleFound = true
            center = circle.center
            maxRadius = circle.radius
            outcome.centerX = (circle.center.x - Float(grid.width) * 0.5) / shortSide
            outcome.centerY = (circle.center.y - Float(grid.height) * 0.5) / shortSide
            outcome.circleScale = circle.radius / (shortSide * 0.5)
        } else {
            outcome.centerX = input.centerX
            outcome.centerY = input.centerY
            outcome.circleScale = input.circleScale
            center = SIMD2<Float>(Float(grid.width) * 0.5 + input.centerX * shortSide,
                                  Float(grid.height) * 0.5 + input.centerY * shortSide)
            maxRadius = shortSide * 0.5 * input.circleScale
        }
        guard maxRadius > 8 else { return nil }

        let baseFocal = maxRadius / max(input.lensHalfFovDegrees * .pi / 180, 0.05)
        // Build the reference view with the parameters in use right now, so the
        // "before" figure describes what the user is actually looking at.
        let base = Lens(center: center,
                        maxRadius: maxRadius,
                        focal: baseFocal,
                        k1: input.k1,
                        k2: input.k2,
                        equidistant: input.equidistant)

        // 2. Straight edges, located in the corrected plane.
        let curves = traceEdges(grid: grid, base: base)
        outcome.lines = curves.count
        guard curves.count >= 5 else {
            outcome.summary = "没找到足够的直线（\(curves.count) 条）。对着门框、桌沿、屏幕边框这类有长直线的场景再按一次。"
            outcome.summaryKind = .notEnoughLines
            return outcome
        }

        // 3. Search the lens parameters that straighten those edges.
        let before = bowing(curves: curves, base: base, focalScale: 1, k1: input.k1)
        guard before.isFinite else {
            outcome.summary = "找到的直线太短或太碎，测不出畸变。把镜头对准一条长直线再按一次。"
            outcome.summaryKind = .notEnoughLines
            return outcome
        }
        var best = (focalScale: Float(1), k1: input.k1, cost: before)
        var coarseFocal: Float = 0.75
        while coarseFocal <= 1.2501 {
            var coarseK: Float = -0.35
            while coarseK <= 0.3501 {
                let cost = bowing(curves: curves, base: base,
                                  focalScale: coarseFocal, k1: coarseK)
                if cost < best.cost {
                    best = (coarseFocal, coarseK, cost)
                }
                coarseK += 0.05
            }
            coarseFocal += 0.0625
        }
        let focalStep = Float(0.0625) / 6
        let kStep = Float(0.05) / 6
        for fi in -4...4 {
            for ki in -4...4 {
                let focalScale = best.focalScale + Float(fi) * focalStep
                let k1 = best.k1 + Float(ki) * kStep
                guard focalScale > 0.4, abs(k1) < 0.6 else { continue }
                let cost = bowing(curves: curves, base: base,
                                  focalScale: focalScale, k1: k1)
                if cost < best.cost {
                    best = (focalScale, k1, cost)
                }
            }
        }

        outcome.k1 = best.k1
        let halfFov = (maxRadius / max(baseFocal * best.focalScale, 1)) * 180 / .pi
        outcome.lensHalfFovDegrees = min(max(halfFov, 45), 120)
        outcome.bowBefore = before
        outcome.bowAfter = best.cost

        let improved = best.cost < before * 0.9
        outcome.applied = improved || before < 0.35
        outcome.summaryKind = improved ? .calibrated : (before < 0.35 ? .alreadyStraight : .noGain)
        outcome.summary = describe(outcome)
        return outcome
    }

    // MARK: - Circle detection

    private static func detectCircle(_ grid: LumaGrid) -> (center: SIMD2<Float>, radius: Float)? {
        let w = Float(grid.width), h = Float(grid.height)
        let stepX = max(1, grid.width / 48), stepY = max(1, grid.height / 64)
        var samples: [Float] = []
        samples.reserveCapacity((grid.width / stepX + 1) * (grid.height / stepY + 1))
        var y = 0
        while y < grid.height {
            var x = 0
            while x < grid.width {
                samples.append(grid.lum[y * grid.width + x])
                x += stepX
            }
            y += stepY
        }
        guard samples.count > 64 else { return nil }
        samples.sort()
        let dark = samples[samples.count / 20]
        let bright = samples[samples.count * 19 / 20]
        guard bright - dark > 10 else { return nil }
        let threshold = dark + 0.35 * (bright - dark)

        var center = SIMD2<Float>(w * 0.5, h * 0.5)
        let searchLimit = (w * w + h * h).squareRoot() * 0.72
        var radius: Float = 0

        for _ in 0..<4 {
            var points: [SIMD2<Float>] = []
            let angles = 144
            for a in 0..<angles {
                let angle = Float(a) / Float(angles) * 2 * .pi
                let dir = SIMD2<Float>(cos(angle), sin(angle))
                var r = searchLimit
                var hit: Float? = nil
                while r > 8 {
                    let p = center + dir * r
                    if p.x >= 0, p.y >= 0, p.x <= w - 1, p.y <= h - 1 {
                        let v = grid.sample(p.x, p.y) ?? 0
                        if v >= threshold {
                            hit = r
                            break
                        }
                    }
                    r -= 1.5
                }
                if let hit = hit {
                    points.append(center + dir * hit)
                }
            }
            guard points.count >= angles / 2 else { return nil }
            guard let fitted = fitCircle(points) else { return nil }
            center = fitted.center
            radius = fitted.radius
        }

        let cornerDistance = (w * w + h * h).squareRoot() * 0.5
        guard radius > Float(min(grid.width, grid.height)) * 0.22,
              radius < cornerDistance * 0.98 else {
            // The rim is outside the frame: the glass covers everything and
            // there is no circle to measure.
            return nil
        }
        return (center, radius)
    }

    private static func fitCircle(_ points: [SIMD2<Float>])
        -> (center: SIMD2<Float>, radius: Float)? {
        guard points.count >= 8 else { return nil }
        var sx: Float = 0, sy: Float = 0
        var sxx: Float = 0, syy: Float = 0, sxy: Float = 0
        var sxz: Float = 0, syz: Float = 0, sz: Float = 0
        for p in points {
            let z = p.x * p.x + p.y * p.y
            sx += p.x; sy += p.y
            sxx += p.x * p.x; syy += p.y * p.y; sxy += p.x * p.y
            sxz += p.x * z; syz += p.y * z; sz += z
        }
        let n = Float(points.count)
        // Solve [2sxx 2sxy sx; 2sxy 2syy sy; 2sx 2sy n] * (a,b,c) = (sxz,syz,sz)
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

    // MARK: - Edge tracing

    private struct Curve {
        var sourcePoints: [SIMD2<Float>]
    }

    private static let outputSize = 224
    private static let tanLimit: Float = 1.7321   // tan(60 degrees)

    /// Builds a corrected view with the base model (k1 = 0), finds long edges
    /// there, and returns each edge together with the source pixels it came
    /// from so any candidate correction can be re-evaluated cheaply.
    private static func traceEdges(grid: LumaGrid, base: Lens) -> [Curve] {
        let size = outputSize
        var image = [Float](repeating: -1, count: size * size)
        for j in 0..<size {
            let ty = tanCoordinate(j, size)
            for i in 0..<size {
                let tx = tanCoordinate(i, size)
                guard let source = base.sourcePoint(SIMD2<Float>(tx, ty)) else { continue }
                if let v = grid.sample(source.x, source.y) {
                    image[j * size + i] = v
                }
            }
        }

        var magnitude = [Float](repeating: 0, count: size * size)
        var sum: Float = 0, sumSquares: Float = 0
        var count = 0
        var gradientX = [Float](repeating: 0, count: size * size)
        var gradientY = [Float](repeating: 0, count: size * size)

        for j in 1..<(size - 1) {
            for i in 1..<(size - 1) {
                var ok = true
                for dj in -1...1 {
                    for di in -1...1 where image[(j + dj) * size + (i + di)] < 0 {
                        ok = false
                    }
                }
                guard ok else { continue }
                let tl = image[(j - 1) * size + (i - 1)]
                let tc = image[(j - 1) * size + i]
                let tr = image[(j - 1) * size + (i + 1)]
                let ml = image[j * size + (i - 1)]
                let mr = image[j * size + (i + 1)]
                let bl = image[(j + 1) * size + (i - 1)]
                let bc = image[(j + 1) * size + i]
                let br = image[(j + 1) * size + (i + 1)]
                let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
                let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
                gradientX[j * size + i] = gx
                gradientY[j * size + i] = gy
                let mag = (gx * gx + gy * gy).squareRoot()
                magnitude[j * size + i] = mag
                sum += mag
                sumSquares += mag * mag
                count += 1
            }
        }
        guard count > 500 else { return [] }
        let mean = sum / Float(count)
        let variance = max(sumSquares / Float(count) - mean * mean, 0)
        let threshold = mean + variance.squareRoot() * 1.1
        guard threshold > 4 else { return [] }

        // Keep near-vertical edge pixels: a line that wanders sideways is what
        // distortion does to a vertical edge, so those are the informative ones.
        var rowPoints = [[(Int, Float)]](repeating: [], count: size)
        for j in 1..<(size - 1) {
            for i in 1..<(size - 1) {
                let mag = magnitude[j * size + i]
                guard mag > threshold else { continue }
                let gx = abs(gradientX[j * size + i])
                let gy = abs(gradientY[j * size + i])
                guard gx > gy * 1.1 else { continue }
                rowPoints[j].append((i, mag))
            }
        }

        var visited = [Bool](repeating: false, count: size * size)
        var curves: [Curve] = []
        let minimumSpan = Int(Float(size) * 0.28)

        for startRow in 1..<(size - 1) {
            for (startColumn, _) in rowPoints[startRow] {
                guard !visited[startRow * size + startColumn] else { continue }
                var chain: [(Int, Int)] = [(startRow, startColumn)]
                visited[startRow * size + startColumn] = true
                var column = startColumn
                var row = startRow
                var gap = 0
                while row + 1 < size {
                    var bestColumn: Int? = nil
                    var bestDistance = Int.max
                    for (candidate, _) in rowPoints[row + 1] {
                        let distance = abs(candidate - column)
                        if distance <= 3, distance < bestDistance, !visited[(row + 1) * size + candidate] {
                            bestDistance = distance
                            bestColumn = candidate
                        }
                    }
                    if let next = bestColumn {
                        column = next
                        row += 1
                        gap = 0
                        chain.append((row, column))
                        visited[row * size + column] = true
                    } else if gap < 2 {
                        // Allow a couple of missing rows so a soft edge still links.
                        gap += 1
                        row += 1
                    } else {
                        break
                    }
                }
                guard chain.count >= 14, row - startRow >= minimumSpan else { continue }
                let horizontal = abs(chain.last!.1 - chain.first!.1)
                guard horizontal < (row - startRow) else { continue }

                var sourcePoints: [SIMD2<Float>] = []
                sourcePoints.reserveCapacity(chain.count)
                for (r, c) in chain {
                    let v = SIMD2<Float>(tanCoordinate(c, size), tanCoordinate(r, size))
                    if let source = base.sourcePoint(v) {
                        sourcePoints.append(source)
                    }
                }
                guard sourcePoints.count >= 14 else { continue }
                curves.append(Curve(sourcePoints: sourcePoints))
            }
        }
        return curves
    }

    private static func tanCoordinate(_ index: Int, _ size: Int) -> Float {
        (Float(index) / Float(size - 1) * 2 - 1) * tanLimit
    }

    // MARK: - Straightness metric

    /// Mean bowing, in corrected-view pixels, of the traced edges. A straight
    /// world edge should score zero; a curved one scores its sagitta.
    private static func bowing(curves: [Curve],
                               base: Lens,
                               focalScale: Float,
                               k1: Float) -> Float {
        var total: Float = 0
        var weight: Float = 0
        for curve in curves {
            var xs: [Float] = []
            var ys: [Float] = []
            xs.reserveCapacity(curve.sourcePoints.count)
            ys.reserveCapacity(curve.sourcePoints.count)
            for p in curve.sourcePoints {
                guard let v = base.outputPoint(p, focalScale: focalScale, candidateK1: k1) else {
                    continue
                }
                xs.append(v.x)
                ys.append(v.y)
            }
            guard xs.count >= 10 else { continue }
            let meanY = ys.reduce(0, +) / Float(ys.count)
            let span = max(ys.max()! - ys.min()!, 1e-4)
            var t: [Float] = []
            t.reserveCapacity(xs.count)
            for y in ys { t.append((y - meanY) / span) }

            // Least squares fit x = A t^2 + B t + C. Bow = |A| / 4.
            var s0: Float = 0, s1: Float = 0, s2: Float = 0, s3: Float = 0, s4: Float = 0
            var x0: Float = 0, x1: Float = 0, x2: Float = 0
            for index in 0..<xs.count {
                let tv = t[index]
                let t2 = tv * tv
                let t3 = t2 * tv
                let t4 = t2 * t2
                s0 += 1; s1 += tv; s2 += t2; s3 += t3; s4 += t4
                x0 += xs[index]
                x1 += xs[index] * tv
                x2 += xs[index] * t2
            }
            let m = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
            let det = determinant(m)
            guard abs(det) > 1e-6 else { continue }
            let a = determinant([[x2, m[0][1], m[0][2]],
                                 [x1, m[1][1], m[1][2]],
                                 [x0, m[2][1], m[2][2]]]) / det
            let lengthWeight = min(1, span / tanLimit)
            total += abs(a) / 4 * lengthWeight
            weight += lengthWeight
        }
        guard weight > 0 else { return .greatestFiniteMagnitude }
        // Convert to output pixels for a readable number.
        return total / weight * (Float(outputSize) * 0.5 / tanLimit)
    }

    // MARK: - Reporting

    enum SummaryKind {
        case calibrated
        case alreadyStraight
        case noGain
        case notEnoughLines
    }

    private static func describe(_ outcome: Outcome) -> String {
        let circle = outcome.circleFound
            ? String(format: "成像圈 r=%.2f 圆心偏移 (%+.2f, %+.2f)",
                     Double(outcome.circleScale),
                     Double(outcome.centerX),
                     Double(outcome.centerY))
            : "成像圈未测到，沿用当前半径"
        switch outcome.summaryKind {
        case .calibrated:
            return String(format: "标定完成：直线 %d 条，弯曲 %.2f → %.2f px，K1 %+.3f，半视场 %.0f°，%@",
                          outcome.lines,
                          Double(outcome.bowBefore), Double(outcome.bowAfter),
                          Double(outcome.k1), Double(outcome.lensHalfFovDegrees), circle)
        case .alreadyStraight:
            return String(format: "画面已经足够直（弯曲 %.2f px），保留当前参数。%@",
                          Double(outcome.bowBefore), circle)
        case .noGain:
            return String(format: "找到 %d 条直线但没找到更好的参数（弯曲 %.2f px），保持原值。%@",
                          outcome.lines, Double(outcome.bowBefore), circle)
        case .notEnoughLines:
            return outcome.summary
        }
    }
}
