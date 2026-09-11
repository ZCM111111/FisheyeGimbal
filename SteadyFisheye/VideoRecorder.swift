import Foundation
import AVFoundation
import CoreVideo
import Photos
import Combine

/// Records what the app actually shows: the undistorted, stabilised frame.
///
/// The same fragment shader that fills the screen is rendered a second time
/// into IOSurface-backed pixel buffers, which are handed straight to an
/// `AVAssetWriter`. Nothing is read back through the CPU, so recording costs one
/// extra GPU pass and no memory copies.
final class VideoRecorder: ObservableObject {

    @Published private(set) var isRecording = false
    @Published private(set) var duration: TimeInterval = 0
    /// Last thing worth telling the user: saved, failed, permission missing.
    @Published private(set) var message: String?

    private(set) var recordingSize = CGSize.zero

    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var pixelPool: CVPixelBufferPool?
    private var fileURL: URL?
    private var sessionStart: CMTime?
    private var lastAppended = CMTime.invalid
    private var startedAt: Date?
    private var clock: Timer?

    /// Height and width are forced even because the video encoder requires it.
    static func fittedSize(for drawableSize: CGSize, width target: CGFloat = 1080) -> CGSize {
        guard drawableSize.width > 1, drawableSize.height > 1 else {
            return CGSize(width: 1080, height: 1920)
        }
        let scale = target / drawableSize.width
        let width = Int((drawableSize.width * scale).rounded()) & ~1
        let height = Int((drawableSize.height * scale).rounded()) & ~1
        return CGSize(width: max(width, 16), height: max(height, 16))
    }

    @discardableResult
    func start(size: CGSize) -> Bool {
        guard !isRecording else { return false }
        let width = Int(size.width) & ~1
        let height = Int(size.height) & ~1
        guard width > 15, height > 15 else {
            message = "画面尺寸异常，无法录像"
            return false
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("steadyfisheye-\(Int(Date().timeIntervalSince1970)).mp4")
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else {
            message = "无法创建录像文件"
            return false
        }

        // H.264 at high profile is the safe choice: HEVC encoding is not
        // guaranteed on every device, and this is a short-clip tool.
        let bitrate = Int(Double(width * height) * 6)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        guard writer.canAdd(input) else {
            message = "录像格式不被支持"
            return false
        }
        writer.add(input)

        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        var pool: CVPixelBufferPool?
        let poolAttributes: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 6
        ]
        guard CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                      poolAttributes as CFDictionary,
                                      attributes as CFDictionary,
                                      &pool) == kCVReturnSuccess,
              let createdPool = pool else {
            message = "无法创建录像缓冲区"
            return false
        }

        guard writer.startWriting() else {
            message = "录像启动失败"
            return false
        }

        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                       sourcePixelBufferAttributes: attributes)
        self.writer = writer
        videoInput = input
        pixelPool = createdPool
        fileURL = url
        recordingSize = CGSize(width: width, height: height)
        sessionStart = nil
        lastAppended = CMTime.invalid
        startedAt = Date()
        duration = 0
        message = nil
        isRecording = true
        startClock()
        return true
    }

    func makePixelBuffer() -> CVPixelBuffer? {
        guard isRecording, let pool = pixelPool else { return nil }
        var buffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
                == kCVReturnSuccess else { return nil }
        return buffer
    }

    /// Called on the main thread once the GPU has finished drawing into the
    /// buffer, so the encoder only ever sees completed frames.
    ///
    /// Takes plain seconds rather than a `CMTime` so callers outside the
    /// AVFoundation world — the Metal renderer — do not have to import CoreMedia.
    func append(_ buffer: CVPixelBuffer, atSeconds seconds: Double) {
        guard isRecording, let writer = writer, let input = videoInput,
              let adaptor = adaptor else { return }
        guard seconds.isFinite, seconds > 0 else { return }
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        if sessionStart == nil {
            sessionStart = time
            writer.startSession(atSourceTime: .zero)
        }
        guard writer.status == .writing, input.isReadyForMoreMediaData,
              let start = sessionStart else { return }

        let relative = CMTimeSubtract(time, start)
        guard CMTimeCompare(relative, .zero) >= 0 else { return }
        if lastAppended.isValid, CMTimeCompare(relative, lastAppended) <= 0 { return }
        lastAppended = relative
        adaptor.append(buffer, withPresentationTime: relative)
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        stopClock()
        duration = 0

        let finishingWriter = writer
        let finishingInput = videoInput
        let url = fileURL
        writer = nil
        videoInput = nil
        adaptor = nil
        pixelPool = nil
        fileURL = nil
        sessionStart = nil
        recordingSize = .zero

        // Mark the input finished on the captured reference; clearing the
        // properties first would make this a no-op and leave the file open.
        finishingInput?.markAsFinished()
        guard let finishingWriter = finishingWriter, let url = url else { return }
        finishingWriter.finishWriting { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                if finishingWriter.status == .completed {
                    self.saveToPhotos(url: url)
                } else {
                    self.message = "录像保存失败"
                }
            }
        }
    }

    private func saveToPhotos(url: URL) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    self?.message = "相册权限未开启，视频未保存"
                }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                PHAssetCreationRequest.forAsset().addResource(with: .video,
                                                              fileURL: url,
                                                              options: nil)
            } completionHandler: { success, error in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if success {
                        self.message = "已保存到相册"
                    } else {
                        self.message = "保存到相册失败：\(error?.localizedDescription ?? "未知错误")"
                    }
                }
            }
        }
    }

    private func startClock() {
        clock?.invalidate()
        clock = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self, let startedAt = self.startedAt else { return }
            self.duration = Date().timeIntervalSince(startedAt)
        }
    }

    private func stopClock() {
        clock?.invalidate()
        clock = nil
        startedAt = nil
    }
}
