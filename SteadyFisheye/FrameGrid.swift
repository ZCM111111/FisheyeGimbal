import Foundation
import CoreVideo
import simd

/// A decimated copy of one camera frame.
///
/// Both detectors work on this so they never touch the 4K buffer directly, and
/// so they see exactly the same data. Only luminance is kept: the cabinet is
/// found from edges, and the buttons' colour changes with the song, so chroma
/// would be misleading rather than useful.
struct FrameGrid {
    let width: Int
    let height: Int
    let lum: [Float]
    /// Pixel size of the frame this was decimated from.
    let sourceSize: CGSize

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

extension FrameGrid {

    /// Decimates one frame. Handles the biplanar 4:2:0 the camera normally
    /// delivers and a BGRA fallback.
    static func make(from pixelBuffer: CVPixelBuffer, targetWidth: Int = 256) -> FrameGrid? {
        guard CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) == kCVReturnSuccess else {
            return nil
        }
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let lumaBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return nil }
        let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let sourceWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let sourceHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        guard sourceWidth > 16, sourceHeight > 16, lumaStride > 0 else { return nil }

        let planar = CVPixelBufferIsPlanar(pixelBuffer)
        let isBGRA = !planar || CVPixelBufferGetPixelFormatType(pixelBuffer)
            == kCVPixelFormatType_32BGRA

        let step = max(1, sourceWidth / max(targetWidth, 32))
        let width = sourceWidth / step
        let height = sourceHeight / step
        guard width > 16, height > 16 else { return nil }

        var lum = [Float](repeating: 0, count: width * height)
        let bytes = lumaBase.assumingMemoryBound(to: UInt8.self)

        for y in 0..<height {
            let sourceRow = y * step
            guard sourceRow < sourceHeight else { break }
            let rowStart = sourceRow * lumaStride
            for x in 0..<width {
                let column = x * step
                guard column < sourceWidth else { continue }
                let index = y * width + x
                if isBGRA {
                    // Green channel is a good enough luma proxy for edge work.
                    lum[index] = Float(bytes[rowStart + column * 4 + 1])
                } else {
                    lum[index] = Float(bytes[rowStart + column])
                }
            }
        }

        return FrameGrid(width: width,
                         height: height,
                         lum: lum,
                         sourceSize: CGSize(width: CGFloat(sourceWidth),
                                            height: CGFloat(sourceHeight)))
    }
}
