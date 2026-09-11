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
