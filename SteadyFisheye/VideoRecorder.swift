import Foundation
import AVFoundation
import CoreVideo
import Photos
import Combine

/// Records what the app actually shows: the undistorted, stabilised frame, with
/// sound.
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
    private var videoAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var audioInput: AVAssetWriterInput?
    private var pixelPool: CVPixelBufferPool?
    private var fileURL: URL?
    private var sessionStart: CMTime?
    private var lastVideoTime = CMTime.invalid
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
    func start(size: CGSize, audioSettings: [String: Any]?) -> Bool {
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
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = true
        guard writer.canAdd(input) else {
            message = "录像格式不被支持"
            return false
        }
        writer.add(input)

        // Audio is optional: when the microphone was unavailable or refused,
        // recording still proceeds silently rather than failing.
        var audio: AVAssetWriterInput?
        if let audioSettings = audioSettings {
            let candidate = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
            candidate.expectsMediaDataInRealTime = true
            if writer.canAdd(candidate) {
                writer.add(candidate)
                audio = candidate
            }
        }

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
            message = "录像启动失败：\(writer.error?.localizedDescription ?? "未知原因")"
            return false
        }
        guard writer.status == .writing else {
            message = "录像无法开始写入"
            return false
        }

        videoAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: attributes)
        self.writer = writer
        videoInput = input
        audioInput = audio
        pixelPool = createdPool
        fileURL = url
        recordingSize = CGSize(width: width, height: height)
        sessionStart = nil
        lastVideoTime = CMTime.invalid
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
        guard seconds.isFinite, seconds > 0 else { return }
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        guard beginSessionIfNeeded(at: time) else { return }
        guard let writer = writer, let input = videoInput,
              let adaptor = videoAdaptor, let start = sessionStart else { return }
        // The writer has to be in the writing state before anything is handed
        // over: startSession and append both raise a hard exception otherwise,
        // which is exactly the crash a failed encoder used to cause.
        guard writer.status == .writing, input.isReadyForMoreMediaData else { return }
        guard CMTimeCompare(time, start) >= 0 else { return }
        if lastVideoTime.isValid, CMTimeCompare(time, lastVideoTime) <= 0 { return }
        lastVideoTime = time
        adaptor.append(buffer, withPresentationTime: time)
    }

    /// Audio samples arrive straight from the capture pipeline.
    ///
    /// The movie session starts on the first video frame's timestamp and both
    /// tracks are appended with their original presentation times, so picture
    /// and sound share one timeline without any re-stamping.
    func appendAudio(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording, let writer = writer, let input = audioInput,
              let start = sessionStart else { return }
        guard writer.status == .writing, input.isReadyForMoreMediaData else { return }
        let presentation = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        guard presentation.isValid, CMTimeCompare(presentation, start) >= 0 else { return }
        guard CMSampleBufferDataIsReady(sampleBuffer) else { return }
        input.append(sampleBuffer)
    }

    /// Starts the movie timeline on the first frame that actually arrives.
    private func beginSessionIfNeeded(at time: CMTime) -> Bool {
        guard let writer = writer, writer.status == .writing else { return false }
        if sessionStart == nil {
            writer.startSession(atSourceTime: time)
            sessionStart = time
        }
        return true
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        stopClock()
        duration = 0

        let finishingWriter = writer
        let finishingVideo = videoInput
        let finishingAudio = audioInput
        let url = fileURL
        writer = nil
        videoInput = nil
        videoAdaptor = nil
        audioInput = nil
        pixelPool = nil
        fileURL = nil
        sessionStart = nil
        lastVideoTime = CMTime.invalid
        recordingSize = .zero

        guard let finishingWriter = finishingWriter, let url = url else { return }

        // finishWriting and markAsFinished both assume the writer is still
        // writing. Calling them on a failed writer is what raises the hard
        // exception here, so the state is checked first and a failure is
        // reported instead of thrown.
        guard finishingWriter.status == .writing else {
            let reason = finishingWriter.error?.localizedDescription ?? "编码器已停止"
            message = "录像失败：\(reason)"
            try? FileManager.default.removeItem(at: url)
            return
        }

        // Mark the inputs finished on the captured references; clearing the
        // properties first would make these no-ops and leave the file open.
        finishingVideo?.markAsFinished()
        finishingAudio?.markAsFinished()

        finishingWriter.finishWriting { [weak self] in
            DispatchQueue.main.async {
                guard let self else { return }
                if finishingWriter.status == .completed {
                    // The file is moved somewhere permanent first, so a refused
                    // or failing photo library can never lose the recording.
                    let kept = self.persist(url: url)
                    self.saveToPhotos(url: kept)
                } else {
                    let reason = finishingWriter.error?.localizedDescription ?? "未知原因"
                    self.message = "录像保存失败：\(reason)"
                    try? FileManager.default.removeItem(at: url)
                }
            }
        }
    }

    /// Moves a finished recording out of the temporary directory and into the
    /// app's own Documents folder, which the Files app can browse.
    private func persist(url: URL) -> URL {
        let manager = FileManager.default
        guard let documents = manager.urls(for: .documentDirectory,
                                           in: .userDomainMask).first else {
            return url
        }
        let folder = documents.appendingPathComponent("SteadyFisheye", isDirectory: true)
        try? manager.createDirectory(at: folder, withIntermediateDirectories: true)
        let destination = folder.appendingPathComponent(url.lastPathComponent)
        try? manager.removeItem(at: destination)
        guard (try? manager.moveItem(at: url, to: destination)) != nil else {
            return url
        }
        return destination
    }

    /// Asks for photo-library access up front.
    ///
    /// Requesting it while a recording finishes means the permission dialog
    /// appears at the worst possible moment, and a refused or unavailable
    /// description there ends the app instead of the save.
    func preparePhotoAccess() {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { _ in }
    }

    private func saveToPhotos(url: URL) {
        let manager = FileManager.default
        let attributes = try? manager.attributesOfItem(atPath: url.path)
        let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
        guard manager.fileExists(atPath: url.path), size > 0 else {
            message = "录像文件为空，已丢弃"
            try? manager.removeItem(at: url)
            return
        }

        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] status in
            DispatchQueue.main.async {
                guard let self else { return }
                guard status == .authorized || status == .limited else {
                    self.message = "相册权限未开启，视频在「文件」App → SteadyFisheye"
                    return
                }
                PHPhotoLibrary.shared().performChanges {
                    let options = PHAssetResourceCreationOptions()
                    // Keep our own copy: the file is also the user's backup.
                    options.shouldMoveFile = false
                    PHAssetCreationRequest.forAsset().addResource(with: .video,
                                                                 fileURL: url,
                                                                 options: options)
                } completionHandler: { success, error in
                    DispatchQueue.main.async {
                        // `self` is already unwrapped and held by the enclosing
                        // closure, so it must not be bound again here.
                        if success {
                            self.message = "已保存到相册"
                        } else {
                            let reason = error?.localizedDescription ?? "未知错误"
                            self.message = "相册保存失败（\(reason)），视频在「文件」App → SteadyFisheye"
                        }
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
