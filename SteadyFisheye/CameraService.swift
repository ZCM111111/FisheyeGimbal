import AVFoundation
import Combine
import CoreMedia
import CoreVideo
import QuartzCore

final class CameraService: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum Lens: String, CaseIterable, Identifiable {
        case wide
        case ultraWide

        var id: String { rawValue }
        var title: String {
            switch self {
            case .wide: return "1x"
            case .ultraWide: return "0.5x"
            }
        }

        var deviceType: AVCaptureDevice.DeviceType {
            switch self {
            case .wide: return .builtInWideAngleCamera
            case .ultraWide: return .builtInUltraWideCamera
            }
        }
    }

    let session = AVCaptureSession()

    @Published private(set) var running = false
    @Published private(set) var lens: Lens = .ultraWide
    @Published private(set) var formatText = "No camera"
    @Published private(set) var activeFPS = 60
    @Published private(set) var measuredFPS = 0
    @Published private(set) var error: String?

    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    private let sessionQueue = DispatchQueue(label: "steadyfisheye.camera.session",
                                              qos: .userInitiated)
    private let videoQueue = DispatchQueue(label: "steadyfisheye.camera.video",
                                           qos: .userInitiated)
    private var selectedLens: Lens = .ultraWide
    private var configured = false
    private var output: AVCaptureVideoDataOutput?
    private var lastFrameTimestamp: Double = 0
    private var fpsEstimate: Double = 0
    private var lastFPSPublishTime: Double = 0

    static func requestAccess(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { completion(true) }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            DispatchQueue.main.async { completion(false) }
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
                self.setError("Camera permission is not granted")
                return
            }
            if !self.configured {
                self.configure()
            }
            guard self.configured else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
            self.publishRunning(self.session.isRunning)
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.publishRunning(false)
        }
    }

    func setLens(_ newLens: Lens) {
        sessionQueue.async { [weak self] in
            guard let self, newLens != self.selectedLens else { return }
            self.selectedLens = newLens
            if self.configured {
                if self.session.isRunning { self.session.stopRunning() }
                self.removeConfiguration()
            }
            self.configure()
            if self.configured { self.session.startRunning() }
            DispatchQueue.main.async { self.lens = newLens }
            self.publishRunning(self.session.isRunning)
        }
    }

    func setError(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.error = message
        }
    }

    private func configure() {
        session.beginConfiguration()
        // inputPriority lets the explicitly selected activeFormat (including
        // its 60 FPS capability) win over a preset's automatic format choice.
        session.sessionPreset = .inputPriority

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [selectedLens.deviceType, .builtInWideAngleCamera],
            mediaType: .video,
            position: .back
        )
        guard let device = discovery.devices.first(where: { $0.deviceType == selectedLens.deviceType })
                ?? discovery.devices.first
                ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            finishConfiguration(with: "No rear camera was found")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                finishConfiguration(with: "The camera input could not be added")
                return
            }
            session.addInput(input)

            try device.lockForConfiguration()
            device.videoZoomFactor = 1.0
            if device.isGeometricDistortionCorrectionSupported {
                device.isGeometricDistortionCorrectionEnabled = false
            }

            // Pick a real 60 FPS format instead of only changing the frame
            // duration on the current format. Prefer 1080p or larger while
            // keeping the format close to the sensor's native aspect ratio.
            if let sixtyFormat = best60FPSFormat(for: device) {
                device.activeFormat = sixtyFormat
            }
            let targetDuration = CMTime(value: 1, timescale: 60)
            let supports60 = device.activeFormat.videoSupportedFrameRateRanges.contains {
                $0.minFrameRate <= 60 && $0.maxFrameRate >= 60
            }
            guard supports60 else {
                device.unlockForConfiguration()
                finishConfiguration(with: "This camera does not provide 60 FPS")
                return
            }
            device.activeVideoMinFrameDuration = targetDuration
            device.activeVideoMaxFrameDuration = targetDuration
            device.unlockForConfiguration()
        } catch {
            finishConfiguration(with: "Camera setup failed: \(error.localizedDescription)")
            return
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.alwaysDiscardsLateVideoFrames = true
        let supported = videoOutput.availableVideoPixelFormatTypes
        let preferred: [OSType] = [
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            kCVPixelFormatType_32BGRA
        ]
        guard let pixelFormat = preferred.first(where: supported.contains) else {
            finishConfiguration(with: "The camera returned no supported pixel format")
            return
        }
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(videoOutput) else {
            finishConfiguration(with: "The video output could not be added")
            return
        }
        session.addOutput(videoOutput)
        output = videoOutput

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
            connection.isVideoMirrored = false
        }

        session.commitConfiguration()
        configured = true
        let dimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let format = "\(dimensions.width)x\(dimensions.height) · " + fourCC(pixelFormat)
        let effectiveLens: Lens = device.deviceType == .builtInUltraWideCamera ? .ultraWide : .wide
        DispatchQueue.main.async { [weak self] in
            self?.lens = effectiveLens
            self?.activeFPS = 60
            self?.formatText = format
            self?.error = nil
        }
    }

    /// Select a practical 60 FPS format.
    ///
    /// Scoring by "distance from 4K" alone was wrong: on lenses whose 60 FPS
    /// formats top out at 1080p, it could prefer an odd 4:3 format over a
    /// proper 16:9 one. Prefer 60 FPS + 16:9 + at least 720p, then take the
    /// largest available; fall back to any 60 FPS format above 720p.
    private func best60FPSFormat(for device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        let sixtyFPS = device.formats.filter { format in
            let supports60 = format.videoSupportedFrameRateRanges.contains {
                $0.minFrameRate <= 60 && $0.maxFrameRate >= 60
            }
            guard supports60 else { return false }
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return d.width >= 1280 && d.height >= 720
        }

        func isWide16x9(_ format: AVCaptureDevice.Format) -> Bool {
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let aspect = Double(d.width) / max(Double(d.height), 1)
            return abs(aspect - (16.0 / 9.0)) < 0.05
        }

        func area(_ format: AVCaptureDevice.Format) -> Double {
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return Double(d.width) * Double(d.height)
        }

        let wide = sixtyFPS.filter(isWide16x9)
        if let best = wide.max(by: { area($0) < area($1) }) {
            return best
        }
        return sixtyFPS.max(by: { area($0) < area($1) })
    }

    private func removeConfiguration() {
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs {
            if let video = output as? AVCaptureVideoDataOutput {
                video.setSampleBufferDelegate(nil, queue: nil)
            }
            session.removeOutput(output)
        }
        session.commitConfiguration()
        output = nil
        lastFrameTimestamp = 0
        fpsEstimate = 0
        lastFPSPublishTime = 0
        configured = false
    }

    private func finishConfiguration(with message: String) {
        session.commitConfiguration()
        configured = false
        setError(message)
    }

    private func publishRunning(_ value: Bool) {
        DispatchQueue.main.async { [weak self] in
            self?.running = value
        }
    }

    private func fourCC(_ code: OSType) -> String {
        let bytes = [UInt8((code >> 24) & 0xff), UInt8((code >> 16) & 0xff),
                     UInt8((code >> 8) & 0xff), UInt8(code & 0xff)]
        return String(bytes: bytes.map { $0 >= 32 && $0 <= 126 ? $0 : 63 }, encoding: .ascii) ?? "????"
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        let captureTime = timestamp.isFinite ? timestamp : CACurrentMediaTime()
        if lastFrameTimestamp > 0 {
            let delta = captureTime - lastFrameTimestamp
            if delta > 0.001 && delta < 1 {
                let measured = 1.0 / delta
                fpsEstimate = fpsEstimate == 0 ? measured : fpsEstimate * 0.9 + measured * 0.1
                let now = CACurrentMediaTime()
                if now - lastFPSPublishTime > 0.25 {
                    lastFPSPublishTime = now
                    let value = Int(fpsEstimate.rounded())
                    DispatchQueue.main.async { [weak self] in
                        self?.measuredFPS = value
                    }
                }
            }
        }
        lastFrameTimestamp = captureTime
        onFrame?(pixelBuffer, captureTime)
    }
}
