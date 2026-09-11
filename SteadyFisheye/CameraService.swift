import AVFoundation
import Combine
import CoreImage
import CoreMedia
import CoreVideo
import QuartzCore

final class CameraService: NSObject, ObservableObject,
                           AVCaptureVideoDataOutputSampleBufferDelegate,
                           AVCaptureAudioDataOutputSampleBufferDelegate {

    enum Lens: String, CaseIterable, Identifiable {
        case wide
        case ultraWide

        var id: String { rawValue }
        var title: String {
            switch self {
            case .wide: return "1x 主摄"
            case .ultraWide: return "0.5x 超广角"
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

    /// Settings the asset writer needs for the microphone, or nil when there is
    /// no usable audio input. Backed by `audioSettings`, which is written once
    /// during configuration on the session queue.
    var audioWriterSettings: [String: Any]? { audioSettings }

    /// Delivers captured audio to the recorder.
    var onAudioSample: ((CMSampleBuffer) -> Void)?

    @Published private(set) var running = false
    @Published private(set) var lens: Lens = .ultraWide
    @Published private(set) var formatText = "无摄像头"
    /// Percentage of the lens image circle that the corners of the preview
    /// reach. Reported by the renderer so the calibration panel can show the
    /// same number the GPU shader is using.
    @Published private(set) var lensCoverage: Float = 0

    /// Called on the main thread by the renderer's HUD update.
    func reportCoverage(_ percent: Float) {
        lensCoverage = percent
    }
    @Published private(set) var activeFPS = 60
    @Published private(set) var measuredFPS = 0
    @Published private(set) var error: String?

    // Focus and exposure state, mirrored for the UI.
    @Published private(set) var exposureBias: Float = 0
    @Published private(set) var exposureBiasRange: ClosedRange<Float> = -8...8
    @Published private(set) var aeafLocked = false

    /// Read from the capture queue; `aeafLocked` is the main-thread mirror.
    private var aeafLockedSnapshot = false
    private var microphoneInput: AVCaptureDeviceInput?
    private var audioOutput: AVCaptureAudioDataOutput?
    private var audioSettings: [String: Any]?
    /// The device currently feeding the session, kept so focus, exposure and
    /// bias can be driven after configuration has finished.
    private var activeDevice: AVCaptureDevice?

    // Bias updates arrive far faster than the hardware should be reconfigured,
    // so the newest request is coalesced instead of queueing every drag event.
    private let biasLock = NSLock()
    private var pendingBias: Float?
    private var biasApplyScheduled = false
    private var lastBiasPublish: CFTimeInterval = 0

    var onFrame: ((CVPixelBuffer, Double) -> Void)?

    private var pendingGrid: ((LensCircleMeasurer.LumaGrid?) -> Void)?
    private let gridLock = NSLock()

    // Saving one untouched frame, for checking detection and calibration
    // against the real thing instead of against assumptions.
    private var stillRequested = false
    private let stillLock = NSLock()
    private let ciContext = CIContext()
    @Published private(set) var rawFrameMessage: String?

    /// Writes the next camera frame, exactly as the pipeline receives it, into
    /// the app's Documents folder. Visible in the Files app.
    func captureRawFrame() {
        stillLock.lock()
        stillRequested = true
        stillLock.unlock()
    }

    private func saveRawFrame(_ pixelBuffer: CVPixelBuffer) {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let data = ciContext.jpegDataRepresentation(
                of: image,
                colorSpace: CGColorSpaceCreateDeviceRGB()) else {
            DispatchQueue.main.async { [weak self] in
                self?.rawFrameMessage = "原始帧编码失败"
            }
            return
        }
        let manager = FileManager.default
        guard let documents = manager.urls(for: .documentDirectory,
                                           in: .userDomainMask).first else { return }
        let folder = documents.appendingPathComponent("SteadyFisheye/frames",
                                                      isDirectory: true)
        try? manager.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = "frame-\(Int(Date().timeIntervalSince1970)).jpg"
        let url = folder.appendingPathComponent(name)
        let ok = (try? data.write(to: url)) != nil
        DispatchQueue.main.async { [weak self] in
            self?.rawFrameMessage = ok
                ? "原始帧已存到「文件」→ SteadyFisheye/frames/\(name)"
                : "原始帧写入失败"
        }
    }

    /// Asks for one frame, decimated to a luminance grid, for measuring where
    /// the image circle sits. The handler runs off the main thread.
    func requestLumaGrid(_ handler: @escaping (LensCircleMeasurer.LumaGrid?) -> Void) {
        gridLock.lock()
        pendingGrid = handler
        gridLock.unlock()
    }

    private let sessionQueue = DispatchQueue(label: "steadyfisheye.camera.session",
                                              qos: .userInitiated)
    private let videoQueue = DispatchQueue(label: "steadyfisheye.camera.video",
                                           qos: .userInitiated)
    private let audioQueue = DispatchQueue(label: "steadyfisheye.camera.audio",
                                           qos: .userInitiated)
    private var selectedLens: Lens = .ultraWide
    /// Lens to bring up on the first configuration, so the camera starts on the
    /// same lens whose stored calibration was loaded.
    var initialLens: Lens?
    private var configured = false
    private var output: AVCaptureVideoDataOutput?
    private var lastFrameTimestamp: Double = 0
    private var fpsEstimate: Double = 0
    private var lastFPSPublishTime: Double = 0

    static func requestAccess(completion: @escaping (Bool) -> Void) {
        // Ask for the microphone first so the audio input can be part of the
        // very first session configuration. Capture access is the one the app
        // cannot work without, so its answer is what gets reported back.
        AVCaptureDevice.requestAccess(for: .audio) { _ in
            requestVideoAccess(completion: completion)
        }
    }

    private static func requestVideoAccess(completion: @escaping (Bool) -> Void) {
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
                self.setError("未获得相机权限")
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

    // MARK: - Focus and exposure

    /// Runs `body` with the device locked for configuration.
    ///
    /// Every device change goes through here. `unlockForConfiguration()` raises
    /// an Objective-C exception — an immediate crash — when the matching lock
    /// did not succeed, so the lock is always proven before the unlock, and no
    /// caller is allowed to hand-roll the pairing.
    private func withDevice(_ body: (AVCaptureDevice) -> Void) {
        guard let device = activeDevice else { return }
        guard (try? device.lockForConfiguration()) != nil else { return }
        body(device)
        device.unlockForConfiguration()
    }

    /// Focuses and meters at a point in the sensor's normalised coordinate
    /// space — the space `AVCaptureDevice` expects, where (0,0) is the top-left
    /// of the unrotated sensor. Callers must convert from a preview point
    /// through the lens mapping first; a raw screen coordinate would land in
    /// the wrong place once the fisheye correction is applied.
    func focusAndExpose(atDevicePoint point: CGPoint) {
        guard point.x >= 0, point.x <= 1, point.y >= 0, point.y <= 1 else { return }
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.withDevice { device in
                if device.isFocusPointOfInterestSupported,
                   device.isFocusModeSupported(.autoFocus) {
                    device.focusPointOfInterest = point
                    device.focusMode = .autoFocus
                }
                if device.isExposurePointOfInterestSupported,
                   device.isExposureModeSupported(.autoExpose) {
                    device.exposurePointOfInterest = point
                    device.exposureMode = .autoExpose
                }
            }

            // Apple focuses once on the tapped point and then keeps tracking
            // slowly, so panning away does not leave focus stuck at a stale
            // distance. Hand back to the continuous modes after the one-shot
            // has had time to settle.
            self.sessionQueue.asyncAfter(deadline: .now() + 1.6) { [weak self] in
                guard let self, !self.aeafLockedSnapshot else { return }
                self.withDevice { device in
                    if device.isFocusModeSupported(.continuousAutoFocus) {
                        device.focusMode = .continuousAutoFocus
                    }
                    if device.isExposureModeSupported(.continuousAutoExposure) {
                        device.exposureMode = .continuousAutoExposure
                    }
                }
            }
        }
    }

    /// Exposure compensation. The slider emits events far faster than the
    /// capture device should be reconfigured, and the UI only calls this a few
    /// times a second while dragging plus once on release.
    func setExposureBias(_ value: Float) {
        biasLock.lock()
        pendingBias = value
        let shouldSchedule = !biasApplyScheduled
        biasApplyScheduled = true
        biasLock.unlock()
        guard shouldSchedule else { return }
        applyPendingBiasSoon()
    }

    /// Applies the newest requested bias, dropping whatever arrived in the
    /// meantime. The pending slot always holds the latest value, so skipping
    /// the intermediate ones costs nothing and keeps the capture device from
    /// being reconfigured once per drag event.
    private func applyPendingBiasSoon() {
        sessionQueue.asyncAfter(deadline: .now() + 0.04) { [weak self] in
            guard let self else { return }
            self.biasLock.lock()
            let latest = self.pendingBias
            self.pendingBias = nil
            // Cleared before the value is applied: anything arriving from here
            // on schedules its own pass instead of being silently dropped.
            self.biasApplyScheduled = false
            self.biasLock.unlock()

            guard let latest = latest, latest.isFinite else { return }
            var applied: Float?
            self.withDevice { device in
                let low = Self.saneBias(device.minExposureTargetBias, fallback: -8)
                let high = max(Self.saneBias(device.maxExposureTargetBias, fallback: 8), low)
                let clamped = min(max(latest, low), high)
                guard clamped.isFinite else { return }
                device.setExposureTargetBias(clamped, completionHandler: nil)
                applied = clamped
            }

            // Publish at most a few times a second. Each publish rebuilds every
            // view that reads the bias — including the slider the finger is on
            // — and doing that once per drag event is what took the UI down.
            guard let clamped = applied else { return }
            let now = CACurrentMediaTime()
            guard now - self.lastBiasPublish > 0.08 else { return }
            self.lastBiasPublish = now
            DispatchQueue.main.async {
                self.exposureBias = clamped
            }
        }
    }

    /// Device-reported limits are used to clamp the bias, so a NaN or an
    /// absurd value from a quirky lens would otherwise be forwarded straight
    /// into `setExposureTargetBias`, which rejects out-of-range input.
    private static func saneBias(_ value: Float, fallback: Float) -> Float {
        guard value.isFinite, abs(value) < 100 else { return fallback }
        return value
    }

    /// Long-pressing takes the same meaning it has in the system camera: the
    /// current focus and exposure are frozen until the user unlocks them.
    func setAEAFLocked(_ locked: Bool) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.aeafLockedSnapshot = locked
            self.withDevice { device in
                if locked {
                    if device.isFocusModeSupported(.locked) {
                        device.focusMode = .locked
                    }
                    if device.isExposureModeSupported(.locked) {
                        device.exposureMode = .locked
                    }
                } else {
                    if device.isFocusModeSupported(.continuousAutoFocus) {
                        device.focusMode = .continuousAutoFocus
                    }
                    if device.isExposureModeSupported(.continuousAutoExposure) {
                        device.exposureMode = .continuousAutoExposure
                    }
                    // Drop the points of interest so the system meters and
                    // focuses the whole frame again, the way it does when you
                    // dismiss the lock in the system camera.
                    if device.isExposurePointOfInterestSupported {
                        device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
                    }
                    if device.isFocusPointOfInterestSupported {
                        device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
                    }
                }
            }
            let value = locked
            DispatchQueue.main.async { self.aeafLocked = value }
        }
    }

    private func configure() {
        // The very first configuration honours the lens whose calibration was
        // restored, so the app does not start on one lens and load the other
        // lens's profile.
        if let initial = initialLens {
            selectedLens = initial
            initialLens = nil
        }
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
            finishConfiguration(with: "未找到后置摄像头")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                finishConfiguration(with: "无法添加相机输入")
                return
            }
            session.addInput(input)
            activeDevice = device

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
                finishConfiguration(with: "该摄像头不支持 60 帧")
                return
            }
            device.activeVideoMinFrameDuration = targetDuration
            device.activeVideoMaxFrameDuration = targetDuration

            // Publish the exposure compensation range this device actually
            // accepts, so the UI slider can span exactly that. The upper bound
            // is forced to be at least the lower one: building a ClosedRange
            // with upper < lower traps at runtime, and a device that reports
            // inconsistent limits would otherwise take the whole app down.
            let low = Self.saneBias(device.minExposureTargetBias, fallback: -8)
            let high = max(Self.saneBias(device.maxExposureTargetBias, fallback: 8), low)
            let initialBias = min(max(Self.saneBias(device.exposureTargetBias, fallback: 0), low),
                                  high)
            DispatchQueue.main.async { [weak self] in
                self?.exposureBiasRange = low...high
                self?.exposureBias = initialBias
            }
            device.unlockForConfiguration()
        } catch {
            finishConfiguration(with: "相机初始化失败：\(error.localizedDescription)")
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
            finishConfiguration(with: "相机未返回受支持的像素格式")
            return
        }
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(videoOutput) else {
            finishConfiguration(with: "无法添加视频输出")
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

        // Microphone, so recordings carry sound. Added only when access is
        // already granted: an audio input that the system refuses would take
        // the whole session down with it, and silent video beats no video.
        if AVCaptureDevice.authorizationStatus(for: .audio) == .authorized,
           let microphone = AVCaptureDevice.default(for: .audio),
           let audioInput = try? AVCaptureDeviceInput(device: microphone),
           session.canAddInput(audioInput) {
            session.addInput(audioInput)
            let audioOutput = AVCaptureAudioDataOutput()
            audioOutput.setSampleBufferDelegate(self, queue: audioQueue)
            if session.canAddOutput(audioOutput) {
                session.addOutput(audioOutput)
                microphoneInput = audioInput
                self.audioOutput = audioOutput
                audioSettings = audioOutput.recommendedAudioSettingsForAssetWriter(
                    writingTo: .mp4)
            }
        }

        session.commitConfiguration()
        configured = true

        if microphoneInput != nil {
            // The capture session drives the microphone, so the audio session
            // has to allow recording before the session starts running.
            let audioSession = AVAudioSession.sharedInstance()
            try? audioSession.setCategory(.playAndRecord,
                                          mode: .videoRecording,
                                          options: [.defaultToSpeaker, .allowBluetooth])
            try? audioSession.setActive(true)
        }
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
        // Audio arrives on its own output and queue; it only needs to reach the
        // recorder, and it must never fall through into the video path.
        if output is AVCaptureAudioDataOutput {
            onAudioSample?(sampleBuffer)
            return
        }
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

        // One-shot hand-off for measuring the lens circle. The frame is
        // decimated right here while the buffer is still valid, so no pool
        // buffer is held across frames.
        gridLock.lock()
        let pending = pendingGrid
        pendingGrid = nil
        gridLock.unlock()
        if let pending = pending {
            let grid = LensCircleMeasurer.makeGrid(from: pixelBuffer)
            DispatchQueue.global(qos: .userInitiated).async {
                pending(grid)
            }
        }

        stillLock.lock()
        let wantsStill = stillRequested
        stillRequested = false
        stillLock.unlock()
        if wantsStill { saveRawFrame(pixelBuffer) }

        onFrame?(pixelBuffer, captureTime)
    }
}
