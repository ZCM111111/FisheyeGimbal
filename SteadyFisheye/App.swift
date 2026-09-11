import SwiftUI
import Combine

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

    @Published private(set) var isAligning = false
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
        guard !isFraming, !isAligning, started else { return }
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
    /// Alignment, not distortion correction: a wide lens stretches whatever is
    /// away from the centre, so an off-centre cabinet looks distorted however
    /// round its screen is. Centring it is what removes that.
    func alignToCabinet() {
        guard !isAligning, started else { return }
        isAligning = true
        alignReport = nil
        alignPass(remaining: 2)
    }

    /// Two passes on purpose: the first one centres a ring that the wide lens
    /// has distorted into a slightly non-circular shape, and once it is centred
    /// that distortion is gone, so the second measurement lands closer.
    private func alignPass(remaining: Int) {
        // Only the detection runs off the main thread: it is pure CPU work over
        // a local frame, with no shared state.
        //
        // The first pass is deliberately lenient for the classical detector: it
        // runs on the raw fisheye frame, where the screen's circle is bent by
        // the wide lens. Once centred, the second pass sees a true circle and
        // can be strict.
        detectAim(lenient: remaining > 1) { [weak self] aim, sourceSize in
            guard let self else { return }
            guard aim.found,
                  let direction = self.settings.cameraDirection(
                    forSourcePixel: aim.center,
                    sourceSize: sourceSize) else {
                self.isAligning = false
                self.alignReport = aim.summary
                return
            }

            // Refuse a swing the frame cannot survive; report it instead of
            // wrecking the picture.
            let offAxis = acos(min(max(direction.z, -1), 1)) * 180 / .pi
            guard offAxis <= Self.maxAimDegrees else {
                self.isAligning = false
                self.alignReport = String(format: "认到的机台偏离画面中心 %.0f°（上限 %.0f°），请先把手机大致对准机台。%@",
                                          Double(offAxis), Double(Self.maxAimDegrees),
                                          aim.summary)
                return
            }

            // Show where it locked on *before* the view moves, so a wrong
            // detection is visible instead of mysterious.
            self.detectionMarker = self.mapToPreview?(aim.center, sourceSize)
            // Fades on its own so it does not sit over the preview forever.
            DispatchQueue.main.asyncAfter(deadline: .now() + 4) { [weak self] in
                self?.detectionMarker = nil
            }

            self.motion.reLock(lookingAlong: direction)
            if remaining > 1 {
                self.alignPass(remaining: remaining - 1)
            } else {
                self.isAligning = false
                self.alignReport = String(format: "已对准（偏离 %.0f°）· %@",
                                          Double(offAxis), aim.summary)
            }
        }
    }

    func start() {
        guard !started else { return }
        started = true

        // Bring the camera up on the same lens whose calibration was restored.
        camera.initialLens = settings.activeLens

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
