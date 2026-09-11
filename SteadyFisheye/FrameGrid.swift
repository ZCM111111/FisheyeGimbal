import Foundation
import CoreVideo
import simd

/// A decimated copy of one camera frame: luminance plus chroma.
///
/// Both detectors work on this so they never touch the 4K buffer directly, and
/// so they see exactly the same data. Chroma is carried as Cb/Cr rather than RGB
/// because that is what the capture pipeline already delivers, and the violet
/// test the button detector needs is a two-channel comparison there.
struct FrameGrid {
    let width: Int
    let height: Int
    let lum: [Float]
    let cb: [Float]
    let cr: [Float]
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

    /// How far towards violet a pixel is: both chroma channels have to be above
    /// neutral, which is what "violet" means in Cb/Cr terms.
    func violetness(_ x: Int, _ y: Int) -> Float {
        let index = y * width + x
        guard index >= 0, index < lum.count else { return -255 }
        return min(cb[index] - 128, cr[index] - 128)
    }

    func luma(_ x: Int, _ y: Int) -> Float {
        let index = y * width + x
        guard index >= 0, index < lum.count else { return 0 }
        return lum[index]
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

        // Biplanar chroma is half resolution in both directions, interleaved Cb
        // then Cr.
        let chromaBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1)
        let chromaStride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)

        let step = max(1, sourceWidth / max(targetWidth, 32))
        let width = sourceWidth / step
        let height = sourceHeight / step
        guard width > 16, height > 16 else { return nil }

        var lum = [Float](repeating: 0, count: width * height)
        var cb = [Float](repeating: 128, count: width * height)
        var cr = [Float](repeating: 128, count: width * height)
        let bytes = lumaBase.assumingMemoryBound(to: UInt8.self)
        let chromaBytes = chromaBase?.assumingMemoryBound(to: UInt8.self)

        for y in 0..<height {
            let sourceRow = y * step
            guard sourceRow < sourceHeight else { break }
            let rowStart = sourceRow * lumaStride
            for x in 0..<width {
                let column = x * step
                guard column < sourceWidth else { continue }
                let index = y * width + x

                if isBGRA {
                    let offset = rowStart + column * 4
                    let blue = Float(bytes[offset])
                    let green = Float(bytes[offset + 1])
                    let red = Float(bytes[offset + 2])
                    lum[index] = 0.299 * red + 0.587 * green + 0.114 * blue
                    cb[index] = 128 - 0.1687 * red - 0.3313 * green + 0.5 * blue
                    cr[index] = 128 + 0.5 * red - 0.4187 * green - 0.0813 * blue
                } else {
                    lum[index] = Float(bytes[rowStart + column])
                    if let chromaBytes = chromaBytes, chromaStride > 0 {
                        let chromaRow = (sourceRow / 2) * chromaStride
                        let chromaColumn = (column / 2) * 2
                        cb[index] = Float(chromaBytes[chromaRow + chromaColumn])
                        cr[index] = Float(chromaBytes[chromaRow + chromaColumn + 1])
                    }
                }
            }
        }

        return FrameGrid(width: width,
                         height: height,
                         lum: lum,
                         cb: cb,
                         cr: cr,
                         sourceSize: CGSize(width: CGFloat(sourceWidth),
                                            height: CGFloat(sourceHeight)))
    }
}
