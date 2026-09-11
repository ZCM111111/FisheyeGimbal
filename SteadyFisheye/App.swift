import SwiftUI
import Combine
import QuartzCore
import simd

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
        /// Capture time of the frame behind this answer, when the detector knows
        /// it. Automatic centring uses it to convert the target with the pose of
        /// that frame instead of the pose the phone has now.
        var capturedAt: TimeInterval?
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
        camera.requestFrameImage { [weak self] image, capturedAt in
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
                                   summary: detection.summary,
                                   capturedAt: capturedAt),
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

    /// How often the search runs.
    ///
    /// Fourteen times a second. This is only how often the *aim* is refreshed:
    /// the cabinet is held in place by the world lock between updates, so this
    /// rate sets how quickly the lock notices that the geometry changed (you
    /// walking around the machine), not how quickly it holds on.
    private static let autoAimInterval: TimeInterval = 0.07
    /// Two detections this close together are the same cabinet.
    private static let autoAimAgreement: Float = 0.06
    /// Time constant of the centring, in seconds.
    ///
    /// Almost zero on purpose. The detector reports where the cabinet *is*, and
    /// the lock is set to that direction immediately rather than eased toward it.
    /// Easing means the frame trails the phone during a turn — the cabinet
    /// wanders across the picture until the chase catches up — which is the
    /// opposite of a locked shot. Set the lock and let the world lock do the
    /// holding: that is the orbit-the-car look, where the subject sits still and
    /// everything else moves.
    ///
    /// Not quite zero, because the detector's own position wobbles by a pixel or
    /// two between frames and that is not worth passing on as shake.
    private static let autoAimSmoothing: Double = 0.03
    /// Still fast while recording, just enough to keep the frame from snapping.
    private static let autoAimRecordingSmoothing: Double = 0.08
    /// How far the detector has to disagree with the current lock before the aim
    /// moves at all.
    ///
    /// This is what makes the shot rigid. Rotating the phone does not move the
    /// cabinet in the world, so a lock that was right stays right, and the
    /// detector's frame-to-frame wobble is smaller than this — re-aiming from
    /// every noisy detection is precisely how the picture ends up breathing.
    /// Anything beyond it is real: the cabinet moved, or you walked around it.
    private static let autoAimUpdateDegrees: Float = 0.4
    /// A correction this large is a new answer rather than tracking, and a single
    /// frame can land on a key or a lamp — so it has to repeat before the picture
    /// moves for it. Ordinary tracking needs no such ceremony.
    private static let autoAimSuspiciousDegrees: Float = 4
    /// The search runs several times a second; the panel does not need to
    /// redraw that often.
    private static let autoAimStatusInterval: TimeInterval = 0.5

    private var autoAimTimer: Timer?
    private var autoAimBusy = false
    private var previousAim: Aim?
    private var agreeingAims = 0
    private var lastAutoAimStatus: CFTimeInterval = 0

    private func startAutoAimTimer() {
        guard autoAimTimer == nil else { return }
        let timer = Timer(timeInterval: Self.autoAimInterval, repeats: true) { [weak self] _ in
            self?.autoAimStep()
        }
        // .common, so a check still happens while the panel is being scrolled.
        RunLoop.main.add(timer, forMode: .common)
        autoAimTimer = timer
    }

    /// One search-and-centre pass.
    private func autoAimStep() {
        guard autoAim, started, !isFraming, !autoAimBusy else { return }
        autoAimBusy = true

        detectAim(lenient: false) { [weak self] aim, sourceSize in
            guard let self else { return }
            self.autoAimBusy = false

            guard aim.found,
                  let target = self.settings.cameraDirection(forSourcePixel: aim.center,
                                                             sourceSize: sourceSize) else {
                // Forget where it was: acting on a memory of a cabinet that is
                // no longer in view is how the picture ends up somewhere random.
                self.previousAim = nil
                self.agreeingAims = 0
                self.motion.setAimTarget(cameraDirection: nil)
                self.publishAutoAim("没看到机台 · \(aim.summary)")
                return
            }

            // How far the cabinet is from where the frame is centred. Measured
            // against the lock rather than the optical axis, because that is what
            // the picture will actually move by.
            let current = self.motion.lockedCameraDirection() ?? SIMD3<Float>(0, 0, 1)
            let error = Self.angleDegrees(from: current, to: target)
            let recording = self.recorder.isRecording

            guard error > Self.autoAimUpdateDegrees else {
                // Already on it. Deliberately leaves the lock alone: the world
                // lock is holding the shot, and re-aiming every pass from a noisy
                // detection would only add shake.
                self.previousAim = aim
                self.agreeingAims = 0
                self.publishAutoAim(String(format: "已锁定 · 偏 %.2f° · %@",
                                           Double(error), aim.summary))
                return
            }
            // While the phone is being turned, the detector's answer is old
            // enough to be wrong about the present pose, so it is no longer used
            // to correct from — the target is converted with the pose of its own
            // frame instead, and the lock follows immediately. That is what keeps
            // the subject still in the middle while the picture moves around it.
            guard error <= Self.maxAimDegrees else {
                self.previousAim = nil
                self.agreeingAims = 0
                self.publishAutoAim(String(format: "机台偏离画面 %.0f°，超过上限 %.0f°",
                                           Double(error), Double(Self.maxAimDegrees)))
                return
            }

            // Hysteresis only for a jump: a single frame can land on a key or a
            // lamp, and being dragged most of the way there on that is worse
            // than waiting one more pass. Ordinary tracking needs no ceremony.
            if error > Self.autoAimSuspiciousDegrees {
                if let previous = self.previousAim,
                   self.sameCabinet(aim, previous, sourceSize: sourceSize) {
                    self.agreeingAims += 1
                } else {
                    self.agreeingAims = 1
                }
                self.previousAim = aim
                guard self.agreeingAims >= 2 else {
                    self.publishAutoAim("正在确认 · \(aim.summary)")
                    return
                }
            } else {
                self.previousAim = aim
                self.agreeingAims = 0
            }

            // Report the target and let the stabiliser chase it per motion
            // sample. Correcting here instead would move the picture in
            // detector-sized steps, five times a second, which reads as a jitter
            // rather than as tracking.
            self.motion.setAimTarget(
                cameraDirection: target,
                smoothing: recording ? Self.autoAimRecordingSmoothing : Self.autoAimSmoothing,
                capturedAt: aim.capturedAt)
            self.previousAim = aim

            // Only for a real move, so the marker does not sit on the preview
            // permanently while tracking.
            if error > 3 {
                self.detectionMarker = self.mapToPreview?(aim.center, sourceSize)
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                    self?.detectionMarker = nil
                }
            }
            self.publishAutoAim(String(format: "正在居中 · 还偏 %.1f° · %@",
                                       Double(error), aim.summary))
        }
    }

    /// The panel does not need the search's full update rate.
    private func publishAutoAim(_ text: String) {
        let now = CACurrentMediaTime()
        guard now - lastAutoAimStatus >= Self.autoAimStatusInterval else { return }
        lastAutoAimStatus = now
        autoAimStatus = text
    }

    private static func angleDegrees(from a: SIMD3<Float>, to b: SIMD3<Float>) -> Float {
        let x = simd_normalize(a)
        let y = simd_normalize(b)
        return acos(min(max(simd_dot(x, y), -1), 1)) * 180 / .pi
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
