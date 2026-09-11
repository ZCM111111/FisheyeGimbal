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
    @Published private(set) var isFraming = false

    /// Aligns on the cabinet and then zooms so its screen always fills the same
    /// fraction of the frame. Same size, same place, every recording.
    func lockFraming() {
        guard !isFraming, !isAligning, started else { return }
        isFraming = true
        alignReport = nil

        camera.requestFrameGrid { [weak self] grid in
            let result = grid.map { CabinetDetector.detect(grid: $0) }
            let sourceSize = grid?.sourceSize
            DispatchQueue.main.async {
                guard let self else { return }
                self.isFraming = false
                guard let result = result, let sourceSize = sourceSize else {
                    self.alignReport = "读不到画面，确认相机在出图"
                    return
                }
                guard result.found,
                      let direction = self.settings.cameraDirection(
                        forSourcePixel: result.centerPixel,
                        sourceSize: sourceSize),
                      let angle = self.settings.cameraAngle(
                        forSourcePixel: SIMD2<Float>(result.centerPixel.x + result.radiusPixel,
                                                     result.centerPixel.y),
                        sourceSize: sourceSize) else {
                    self.alignReport = result.summary
                    return
                }
                // Centre first, then zoom, so the measured radius refers to a
                // circle that is no longer bent by the wide lens.
                self.motion.reLock(lookingAlong: direction)
                self.settings.applyFramingLock(screenAngle: angle)
                self.alignReport = String(format: "已对准并锁定构图 · 占比 %.0f%% · %@",
                                          Double(self.settings.cabinetFillTarget * 100),
                                          result.summary)
            }
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
        camera.requestFrameGrid { [weak self] grid in
            // Only the detection runs off the main thread: it is pure CPU work
            // over a local grid, with no shared state.
            //
            // The first pass is deliberately lenient: it runs on the raw fisheye
            // frame, where the screen's circle is bent by the wide lens. Once
            // centred, the second pass sees a true circle and can be strict.
            let limit: Float = remaining > 1 ? 0.45 : 0.55
            let result = grid.map { CabinetDetector.detect(grid: $0, supportLimit: limit) }
            let sourceSize = grid?.sourceSize
            DispatchQueue.main.async {
                guard let self else { return }
                guard let result = result, let sourceSize = sourceSize else {
                    self.isAligning = false
                    self.alignReport = "读不到画面，确认相机在出图"
                    return
                }
                guard result.found,
                      let direction = self.settings.cameraDirection(
                        forSourcePixel: result.centerPixel,
                        sourceSize: sourceSize) else {
                    self.isAligning = false
                    self.alignReport = result.summary
                    return
                }
                self.motion.reLock(lookingAlong: direction)
                if remaining > 1 {
                    self.alignPass(remaining: remaining - 1)
                } else {
                    self.isAligning = false
                    self.alignReport = "已对准 · " + result.summary
                }
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
