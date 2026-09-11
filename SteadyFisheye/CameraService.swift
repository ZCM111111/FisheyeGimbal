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
    @Published private(set) var lens: Lens = .wide
    @Published private(set) var formatText = "No camera"
    @Published private(set) var error: String?

    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    private let sessionQueue = DispatchQueue(label: "steadyfisheye.camera.session",
                                              qos: .userInitiated)
    private let videoQueue = DispatchQueue(label: "steadyfisheye.camera.video",
                                           qos: .userInitiated)
    private var selectedLens: Lens = .wide
    private var configured = false
    private var output: AVCaptureVideoDataOutput?

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
        session.sessionPreset = .high

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
            let targetDuration = CMTime(value: 1, timescale: 30)
            if device.activeFormat.videoSupportedFrameRateRanges.contains(where: {
                $0.minFrameRate <= 30 && $0.maxFrameRate >= 30
            }) {
                device.activeVideoMinFrameDuration = targetDuration
                device.activeVideoMaxFrameDuration = targetDuration
            }
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
        let format = "\(dimensions.width)x\(dimensions.height)  " + fourCC(pixelFormat)
        DispatchQueue.main.async { [weak self] in
            self?.formatText = format
            self?.error = nil
        }
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
        onFrame?(pixelBuffer, timestamp.isFinite ? timestamp : CACurrentMediaTime())
    }
}
