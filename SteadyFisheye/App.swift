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
    @Published private(set) var isCalibrating = false
    @Published private(set) var calibrationReport: String?

    /// One-tap lens calibration: measures the image circle and fits the
    /// distortion from straight edges in the current scene.
    func runAutoCalibration() {
        guard !isCalibrating, started else { return }
        isCalibrating = true
        calibrationReport = nil

        let input = AutoCalibrator.Input(
            lensHalfFovDegrees: settings.lensHalfFov,
            circleScale: settings.circleScale,
            centerX: settings.centerX,
            centerY: settings.centerY,
            k1: settings.k1,
            k2: settings.k2,
            equidistant: settings.projection == .equidistant
        )

        camera.requestCalibrationGrid { [weak self] grid in
            let outcome = grid.flatMap { AutoCalibrator.analyze(grid: $0, input: input) }
            DispatchQueue.main.async {
                guard let self else { return }
                self.isCalibrating = false
                guard let outcome = outcome else {
                    self.calibrationReport = "标定失败：读不到画面或画面太暗。确认相机在出图，然后对着有长直线的场景再试。"
                    return
                }
                self.settings.apply(outcome)
                self.calibrationReport = outcome.summary
            }
        }
    }

    func start() {
        guard !started else { return }
        started = true

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
