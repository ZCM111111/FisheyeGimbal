import Foundation
import Combine
import Metal
import MetalKit
import CoreVideo
import QuartzCore
import simd

private struct FEUniforms {
    var rotation0 = SIMD4<Float>(1, 0, 0, 0)
    var rotation1 = SIMD4<Float>(0, 1, 0, 0)
    var rotation2 = SIMD4<Float>(0, 0, 1, 0)
    var sourceView = SIMD4<Float>(1920, 1080, 390, 844)
    var lens0 = SIMD4<Float>(960, 540, 300, 540)
    var lens1 = SIMD4<Float>(1.57, 1.36, 0, 0.02)
    var distortion = SIMD4<Float>(0, 0, 1, 0)
}

private final class SourceFrame {
    let pixelBuffer: CVPixelBuffer
    let textures: [MTLTexture]
    let wrappers: [CVMetalTexture]
    let format: UInt32
    let timestamp: Double
    let size: SIMD2<Float>

    init(pixelBuffer: CVPixelBuffer,
         textures: [MTLTexture],
         wrappers: [CVMetalTexture],
         format: UInt32,
         timestamp: Double,
         size: SIMD2<Float>) {
        self.pixelBuffer = pixelBuffer
        self.textures = textures
        self.wrappers = wrappers
        self.format = format
        self.timestamp = timestamp
        self.size = size
    }
}

final class MetalRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache
    private let settings: FisheyeSettings
    private let motion: MotionStabilizer
    private let frameLock = NSLock()
    private var latestFrame: SourceFrame?

    private weak var view: MTKView?
    private var lastFrameTimestamp: Double = 0
    private var cameraFPS: Double = 0
    private var lastHUDTime: Double = 0

    @Published private(set) var inputSize = CGSize.zero
    @Published private(set) var cameraFPSText = "camera --"

    enum RendererError: Error {
        case noMetal
        case noCommandQueue
        case noLibraryFunction
        case pipelineCreationFailed
        case textureCacheCreationFailed
    }

    init(view: MTKView, settings: FisheyeSettings, motion: MotionStabilizer) throws {
        guard let device = view.device ?? MTLCreateSystemDefaultDevice() else {
            throw RendererError.noMetal
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw RendererError.noCommandQueue
        }
        guard let library = device.makeDefaultLibrary(),
              let vertex = library.makeFunction(name: "FisheyeVertex"),
              let fragment = library.makeFunction(name: "FisheyeFragment") else {
            throw RendererError.noLibraryFunction
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "SteadyFisheye"
        descriptor.vertexFunction = vertex
        descriptor.fragmentFunction = fragment
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        guard let pipeline = try? device.makeRenderPipelineState(descriptor: descriptor) else {
            throw RendererError.pipelineCreationFailed
        }

        var cache: CVMetalTextureCache?
        let cacheStatus = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        guard cacheStatus == kCVReturnSuccess, let cache else {
            throw RendererError.textureCacheCreationFailed
        }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pipeline
        self.textureCache = cache
        self.settings = settings
        self.motion = motion
        super.init()

        view.device = device
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = true
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.delegate = self
        self.view = view
    }

    func enqueue(pixelBuffer: CVPixelBuffer, timestamp: Double) {
        guard let frame = makeFrame(pixelBuffer: pixelBuffer, timestamp: timestamp) else { return }
        frameLock.lock()
        latestFrame = frame
        frameLock.unlock()

        let now = timestamp
        if lastFrameTimestamp > 0 {
            let delta = now - lastFrameTimestamp
            if delta > 0.001 && delta < 1 {
                let measured = 1 / delta
                cameraFPS = cameraFPS == 0 ? measured : cameraFPS * 0.9 + measured * 0.1
            }
        }
        lastFrameTimestamp = now
    }

    private func makeFrame(pixelBuffer: CVPixelBuffer, timestamp: Double) -> SourceFrame? {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0 else { return nil }

        var textures: [MTLTexture] = []
        var wrappers: [CVMetalTexture] = []
        var sourceFormat: UInt32

        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            guard let pair = makeTexture(pixelBuffer: pixelBuffer,
                                          pixelFormat: .bgra8Unorm,
                                          width: width,
                                          height: height,
                                          plane: 0) else { return nil }
            textures = [pair.texture]
            wrappers = [pair.wrapper]
            sourceFormat = 0
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            guard let y = makeTexture(pixelBuffer: pixelBuffer,
                                      pixelFormat: .r8Unorm,
                                      width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
                                      height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 0),
                                      plane: 0),
                  let uv = makeTexture(pixelBuffer: pixelBuffer,
                                       pixelFormat: .rg8Unorm,
                                       width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 1),
                                       height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 1),
                                       plane: 1) else { return nil }
            textures = [y.texture, uv.texture]
            wrappers = [y.wrapper, uv.wrapper]
            sourceFormat = 1
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            guard let y = makeTexture(pixelBuffer: pixelBuffer,
                                      pixelFormat: .r8Unorm,
                                      width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
                                      height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 0),
                                      plane: 0),
                  let uv = makeTexture(pixelBuffer: pixelBuffer,
                                       pixelFormat: .rg8Unorm,
                                       width: CVPixelBufferGetWidthOfPlane(pixelBuffer, 1),
                                       height: CVPixelBufferGetHeightOfPlane(pixelBuffer, 1),
                                       plane: 1) else { return nil }
            textures = [y.texture, uv.texture]
            wrappers = [y.wrapper, uv.wrapper]
            sourceFormat = 2
        default:
            return nil
        }

        return SourceFrame(pixelBuffer: pixelBuffer,
                           textures: textures,
                           wrappers: wrappers,
                           format: sourceFormat,
                           timestamp: timestamp,
                           size: SIMD2<Float>(Float(width), Float(height)))
    }

    private func makeTexture(pixelBuffer: CVPixelBuffer,
                             pixelFormat: MTLPixelFormat,
                             width: Int,
                             height: Int,
                             plane: Int) -> (texture: MTLTexture, wrapper: CVMetalTexture)? {
        guard width > 0, height > 0 else { return nil }
        var wrapper: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pixelBuffer,
            nil,
            pixelFormat,
            width,
            height,
            plane,
            &wrapper
        )
        guard status == kCVReturnSuccess,
              let wrapper,
              let texture = CVMetalTextureGetTexture(wrapper) else { return nil }
        return (texture, wrapper)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPass = view.currentRenderPassDescriptor else { return }

        frameLock.lock()
        let frame = latestFrame
        frameLock.unlock()
        guard let frame else { return }

        let drawableSize = view.drawableSize
        guard drawableSize.width > 0, drawableSize.height > 0,
              let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        var uniforms = makeUniforms(frame: frame, viewSize: drawableSize)

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPass) else {
            return
        }
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBytes(&uniforms,
                               length: MemoryLayout<FEUniforms>.stride,
                               index: 0)
        encoder.setFragmentBytes(&uniforms,
                                 length: MemoryLayout<FEUniforms>.stride,
                                 index: 0)
        encoder.setFragmentTexture(frame.textures[0], index: 0)
        encoder.setFragmentTexture(frame.textures.count > 1 ? frame.textures[1] : frame.textures[0], index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        // Keeping the frame in this completion closure keeps its CVMetalTexture
        // wrappers alive until the GPU has finished sampling them.
        commandBuffer.addCompletedHandler { _ in
            withExtendedLifetime(frame) {}
        }
        commandBuffer.present(drawable)
        commandBuffer.commit()

        updateHUD(sourceSize: CGSize(width: CGFloat(frame.size.x),
                                      height: CGFloat(frame.size.y)))
    }

    private func makeUniforms(frame: SourceFrame, viewSize: CGSize) -> FEUniforms {
        let parameters = settings.parameters(sourceSize: CGSize(width: CGFloat(frame.size.x),
                                                                  height: CGFloat(frame.size.y)))
        let snapshot = motion.renderSnapshot(forFrameAt: frame.timestamp)
        let matrix = snapshot.valid ? snapshot.cameraFromLocked : matrix_identity_float3x3
        let columns = matrix.columns

        return FEUniforms(
            rotation0: SIMD4<Float>(columns.0, 0),
            rotation1: SIMD4<Float>(columns.1, 0),
            rotation2: SIMD4<Float>(columns.2, 0),
            sourceView: SIMD4<Float>(frame.size.x, frame.size.y,
                                     Float(viewSize.width), Float(viewSize.height)),
            lens0: SIMD4<Float>(parameters.center.x, parameters.center.y,
                                parameters.focal, parameters.maxRadius),
            lens1: SIMD4<Float>(parameters.maxTheta, parameters.outputFov,
                                parameters.projection, parameters.edgeFeather),
            distortion: SIMD4<Float>(parameters.k1, parameters.k2,
                                     Float(frame.format), 1)
        )
    }

    private func updateHUD(sourceSize: CGSize) {
        let now = CACurrentMediaTime()
        guard now - lastHUDTime > 0.25 else { return }
        lastHUDTime = now
        let fps = Int(cameraFPS.rounded())
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.inputSize = sourceSize
            self.cameraFPSText = "camera \(fps) fps"
        }
    }
}
