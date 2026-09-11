import SwiftUI
import Combine
import QuartzCore

@main
struct SteadyFisheyeApp: App {
    @StateObject private var app = CameraApp()

    var body: some Scene {
        WindowGroup {
            ContentView(app: app,
                        settings: app.settings,
                        motion: app.motion,
                        camera: app.camera,
                        recorder: app.recorder)
                .preferredColorScheme(.dark)
                .statusBarHidden()
                .ignoresSafeArea()
        }
    }
}

final class CameraApp: ObservableObject {
    let settings = FisheyeSettings()
    let motion = MotionStabilizer()
    let camera = CameraService()
    let recorder = VideoRecorder()

    @Published private(set) var started = false
    @Published private(set) var isCentering = false
    @Published private(set) var centerReport: String?

    /// Measures where the fisheye image circle sits and moves it to the middle
    /// of the frame.
    ///
    /// Unlike the removed auto-calibration this only writes `centerX/centerY`.
    /// The circle's position is measured rather than fitted, and hand-tuned
    /// distortion terms are left untouched.
    func centerLens() {
        guard !isCentering, started else { return }
        isCentering = true
        centerReport = nil

        camera.requestFrameGrid { [weak self] grid in
            let result = grid.map { LensCircleMeasurer.measure(grid: $0) }
            DispatchQueue.main.async {
                guard let self else { return }
                self.isCentering = false
                guard let result = result else {
                    self.centerReport = "读不到画面，确认相机在出图"
                    return
                }
                guard result.found else {
                    self.centerReport = result.summary
                    return
                }
                self.settings.applyMeasuredCenter(centerX: result.centerX,
                                                  centerY: result.centerY)
                self.centerReport = result.summary
            }
        }
    }

    @Published private(set) var alignReport: String?
    /// Where the detector thinks the screen is, in preview coordinates, so it
    /// can be shown rather than guessed at.
    @Published private(set) var detectionMarker: CGPoint?

    /// Maps a source pixel to preview coordinates. Set by the view, which owns
    /// the renderer that knows the current pose.
    var mapToPreview: ((SIMD2<Float>, CGSize) -> CGPoint?)?

    /// Never swing the view further than this when re-aiming.
    ///
    /// Beyond roughly this much the frame starts sampling past the edge of the
    /// lens, and the shader pins those samples to the rim, which turns the
    /// corners into a radial smear — the kaleidoscope look. A detection that
    /// asks for more than this is far more likely to be wrong than the camera
    /// is to be pointed that badly.
    private static let maxAimDegrees: Float = 25
    @Published private(set) var isFraming = false

    /// A confident enough model hit; below this the classical detector is asked
    /// instead, because acting on a weak box moves the picture for no reason.
    ///
    /// The model is a weak classifier — 0.15 is already a clear detection for it,
    /// since its correct boxes score 0.13–0.41 — so this sits just above noise.
    private static let minimumModelConfidence: Float = 0.15

    /// What either detector found, in the pixels of the frame it measured.
    struct Aim {
        var found: Bool
        var center: SIMD2<Float>
        var radius: Float
        var summary: String
    }

    /// One detection: the trained model first, the classical circle detector as
    /// the fallback.
    ///
    /// The two run one after the other rather than side by side. A frame holds
    /// only one pending request of each kind, and the fallback is cheap enough
    /// that a duplicated wait is not worth the extra state.
    private func detectAim(lenient: Bool, completion: @escaping (Aim, CGSize) -> Void) {
        guard ScreenDetector.isAvailable else {
            detectAimClassical(lenient: lenient, completion: completion)
            return
        }
        camera.requestFrameImage { [weak self] image in
            let detection = image.flatMap { ScreenDetector.detect(in: $0) }
            DispatchQueue.main.async {
                guard let self else { return }
                if let detection = detection,
                   detection.confidence >= Self.minimumModelConfidence,
                   let image = image {
                    completion(Aim(found: true,
                                   center: SIMD2<Float>(Float(detection.center.x),
                                                        Float(detection.center.y)),
                                   radius: Float(detection.radius),
                                   summary: detection.summary),
                               CGSize(width: image.width, height: image.height))
                } else {
                    self.detectAimClassical(lenient: lenient, completion: completion)
                }
            }
        }
    }

    /// The classical structural detector, which needs no model in the bundle.
    private func detectAimClassical(lenient: Bool,
                                    completion: @escaping (Aim, CGSize) -> Void) {
        // The lenient bar is for the first alignment pass, which runs on the raw
        // fisheye frame where the screen's circle is still bent.
        let limit: Float = lenient ? 0.45 : 0.55
        camera.requestFrameGrid { grid in
            let result = grid.map { CabinetDetector.detect(grid: $0, supportLimit: limit) }
            let sourceSize = grid?.sourceSize
            DispatchQueue.main.async {
                guard let result = result, let sourceSize = sourceSize else {
                    completion(Aim(found: false, center: SIMD2<Float>(0, 0), radius: 0,
                                   summary: "读不到画面，确认相机在出图"), .zero)
                    return
                }
                completion(Aim(found: result.found,
                               center: result.centerPixel,
                               radius: result.radiusPixel,
                               summary: result.summary),
                           sourceSize)
            }
        }
    }

    /// Aligns on the cabinet and then zooms so its screen always fills the same
    /// fraction of the frame. Same size, same place, every recording.
    func lockFraming() {
        guard !isFraming, !autoAimBusy, started else { return }
        isFraming = true
        alignReport = nil

        detectAim(lenient: false) { [weak self] aim, sourceSize in
            guard let self else { return }
            self.isFraming = false
            guard aim.found,
                  let direction = self.settings.cameraDirection(
                    forSourcePixel: aim.center,
                    sourceSize: sourceSize),
                  let angle = self.settings.cameraAngle(
                    forSourcePixel: SIMD2<Float>(aim.center.x + aim.radius, aim.center.y),
                    sourceSize: sourceSize) else {
                self.alignReport = aim.summary
                return
            }
            // Centre first, then zoom, so the measured radius refers to a
            // circle that is no longer bent by the wide lens.
            self.motion.reLock(lookingAlong: direction)
            self.settings.applyFramingLock(screenAngle: angle)
            self.alignReport = String(format: "已对准并锁定构图 · 占比 %.0f%% · %@",
                                      Double(self.settings.cabinetFillTarget * 100),
                                      aim.summary)
        }
    }

    /// Finds the cabinet in the frame and re-aims the lock so it sits in the
    /// middle.
    ///
    /// Finds the cabinet and re-aims the lock so it sits in the middle — and
    /// keeps doing it, so there is no button to press.
    ///
    /// Alignment, not distortion correction: a wide lens stretches whatever is
    /// away from the centre, so an off-centre cabinet looks distorted however
    /// round its screen is. Centring it is what removes that.
    ///
    /// Off is one tap away for anyone who wants to compose the shot themselves.
    @Published var autoAim: Bool =
        (UserDefaults.standard.object(forKey: "autoAim") as? Bool) ?? true {
        didSet {
            UserDefaults.standard.set(autoAim, forKey: "autoAim")
            if !autoAim {
                autoAimStatus = nil
                previousAim = nil
                agreeingAims = 0
            }
        }
    }

    /// What the automatic search is doing right now, for the panel.
    @Published private(set) var autoAimStatus: String?

    /// How often the search runs. Slower than the frame rate on purpose: the
    /// picture barely changes between two checks, and one Vision inference per
    /// frame would spend battery and heat on nothing new.
    private static let autoAimInterval: TimeInterval = 0.45
    /// Two detections this close together are the same cabinet.
    private static let autoAimAgreement: Float = 0.06
    /// Time to let a move settle before measuring again.
    private static let autoAimCooldown: TimeInterval = 2.0
    /// Below this much error, moving the picture is not worth it.
    private static let autoAimToleranceDegrees: Float = 2.5

    private var autoAimTimer: Timer?
    private var autoAimBusy = false
    private var autoAimCooldownUntil: CFTimeInterval = 0
    private var previousAim: Aim?
    private var agreeingAims = 0

    private func startAutoAimTimer() {
        guard autoAimTimer == nil else { return }
        let timer = Timer(timeInterval: Self.autoAimInterval, repeats: true) { [weak self] _ in
            self?.autoAimStep()
        }
        // .common, so a check still happens while the panel is being scrolled.
        RunLoop.main.add(timer, forMode: .common)
        autoAimTimer = timer
    }

    /// One search-and-centre cycle.
    private func autoAimStep() {
        // Nothing while recording: a re-aim mid-take is a jump in the footage,
        // and the mount is holding the framing steady by then anyway.
        guard autoAim, started, !isFraming, !autoAimBusy, !recorder.isRecording,
              CACurrentMediaTime() >= autoAimCooldownUntil else { return }
        autoAimBusy = true

        detectAim(lenient: false) { [weak self] aim, sourceSize in
            guard let self else { return }
            self.autoAimBusy = false

            guard aim.found,
                  let direction = self.settings.cameraDirection(forSourcePixel: aim.center,
                                                                sourceSize: sourceSize) else {
                // Forget where it was: acting on a memory of a cabinet that is
                // no longer in view is how the picture ends up somewhere random.
                self.previousAim = nil
                self.agreeingAims = 0
                self.autoAimStatus = "没看到机台 · \(aim.summary)"
                return
            }

            // Hysteresis: the same cabinet twice in a row. A single frame can
            // land on a key or a lamp, and moving the view for that is worse
            // than not moving it at all.
            if let previous = self.previousAim,
               self.sameCabinet(aim, previous, sourceSize: sourceSize) {
                self.agreeingAims += 1
            } else {
                self.agreeingAims = 1
            }
            self.previousAim = aim
            guard self.agreeingAims >= 2 else {
                self.autoAimStatus = "正在确认 · \(aim.summary)"
                return
            }

            let offAxis = acos(min(max(direction.z, -1), 1)) * 180 / .pi
            guard offAxis <= Self.maxAimDegrees else {
                self.autoAimStatus = String(format: "机台偏离 %.0f°，超过上限 %.0f°",
                                            Double(offAxis), Double(Self.maxAimDegrees))
                return
            }
            guard offAxis > Self.autoAimToleranceDegrees else {
                self.autoAimStatus = String(format: "已居中 · 偏 %.1f° · %@",
                                            Double(offAxis), aim.summary)
                return
            }

            self.autoAimCooldownUntil = CACurrentMediaTime() + Self.autoAimCooldown
            // Show where it found the cabinet *before* the view moves, so a
            // wrong detection is visible instead of mysterious.
            self.detectionMarker = self.mapToPreview?(aim.center, sourceSize)
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
                self?.detectionMarker = nil
            }
            self.motion.reLock(lookingAlong: direction)
            self.autoAimStatus = String(format: "已自动居中（原来偏 %.0f°）· %@",
                                        Double(offAxis), aim.summary)
        }
    }

    /// Same cabinet, not merely "a circle about that size somewhere".
    private func sameCabinet(_ a: Aim, _ b: Aim, sourceSize: CGSize) -> Bool {
        let shortSide = Float(min(sourceSize.width, sourceSize.height))
        guard shortSide > 1 else { return false }
        let dx = a.center.x - b.center.x
        let dy = a.center.y - b.center.y
        let moved = (dx * dx + dy * dy).squareRoot() / shortSide
        let ratio = a.radius / max(b.radius, 1)
        return moved <= Self.autoAimAgreement && (0.6...1.7).contains(ratio)
    }

    func start() {
        guard !started else { return }
        started = true

        // Bring the camera up on the same lens whose calibration was restored.
        camera.initialLens = settings.activeLens
        startAutoAimTimer()

        motion.start()
        CameraService.requestAccess { [weak self] granted in
            guard let self else { return }
            if granted {
                self.camera.start()
            } else {
                self.camera.setError("相机权限被拒绝，请到「设置」中开启。")
            }
        }
    }

    func stop() {
        started = false
        camera.stop()
        motion.stop()
    }
}
