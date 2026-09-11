import SwiftUI
import MetalKit
import UIKit

struct ContentView: View {
    @ObservedObject var app: CameraApp
    @ObservedObject var settings: FisheyeSettings
    @ObservedObject var motion: MotionStabilizer
    @ObservedObject var camera: CameraService

    @State private var showControls = false
    @State private var rendererFailed = false
    @State private var baseFov: Float?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            MetalPreview(camera: camera,
                         settings: settings,
                         motion: motion,
                         failed: $rendererFailed)
                .ignoresSafeArea()
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            if baseFov == nil { baseFov = settings.outputFov }
                            let next = (baseFov ?? settings.outputFov) / Float(value)
                            settings.outputFov = min(max(next, 30), 130)
                        }
                        .onEnded { _ in baseFov = nil }
                )

            VStack(spacing: 0) {
                statusBar
                Spacer()
                bottomBar
            }

            if rendererFailed {
                Color.black.opacity(0.9).ignoresSafeArea()
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.largeTitle)
                    Text("Metal is unavailable on this device")
                        .font(.headline)
                }
                .foregroundStyle(.white)
            }
        }
        .sheet(isPresented: $showControls) {
            SettingsSheet(settings: settings,
                          motion: motion,
                          camera: camera,
                          dismiss: { showControls = false })
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .onAppear { app.start() }
        .onDisappear { app.stop() }
    }

    private var statusBar: some View {
        HStack(spacing: 10) {
            Label(camera.running ? "LIVE" : "WAITING",
                  systemImage: camera.running ? "video.fill" : "video.slash")
                .foregroundStyle(camera.running ? .green : .orange)
            VStack(alignment: .leading, spacing: 1) {
                Text(camera.formatText)
                    .lineLimit(1)
                Text("target 60  /  actual \(camera.measuredFPS) fps")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.65))
            }
            Spacer()
            Label(motion.locked ? "LOCKED" : "IMU…", systemImage: "gyroscope")
                .foregroundStyle(motion.available ? .green : .orange)
            Button { motion.recenter() } label: {
                Image(systemName: "scope")
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            Button { showControls = true } label: {
                Image(systemName: "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            .accessibilityLabel("Open camera settings")
        }
        .font(.caption.bold())
        .foregroundStyle(.white)
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(.horizontal, 10)
        .padding(.top, 8)
    }

    private var bottomBar: some View {
        HStack(spacing: 12) {
            Text("FOV \(Int(settings.outputFov))°")
            Text("•")
                .foregroundStyle(.white.opacity(0.4))
            Text(motion.mode.title)
            Spacer()
            Text("Pinch to zoom")
                .foregroundStyle(.white.opacity(0.6))
        }
        .font(.caption2.monospaced())
        .foregroundStyle(.white.opacity(0.85))
        .padding(.horizontal, 13)
        .padding(.vertical, 9)
        .background(.black.opacity(0.55), in: Capsule())
        .padding(.horizontal, 10)
        .padding(.bottom, 8)
    }
}

private struct SettingsSheet: View {
    @ObservedObject var settings: FisheyeSettings
    @ObservedObject var motion: MotionStabilizer
    @ObservedObject var camera: CameraService
    let dismiss: () -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Stabilization", selection: Binding(
                        get: { motion.mode },
                        set: { motion.setMode($0) }
                    )) {
                        ForEach(MotionStabilizer.Mode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    Text("60 FPS target · \(camera.measuredFPS) FPS measured")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("Camera lens", selection: Binding(
                        get: { camera.lens },
                        set: { camera.setLens($0) }
                    )) {
                        ForEach(CameraService.Lens.allCases) { lens in
                            Text(lens.title).tag(lens)
                        }
                    }
                } header: {
                    Text("Camera")
                }

                Section {
                    slider("Output FOV", value: $settings.outputFov, range: 30...130, suffix: "°")
                    slider("Lens half FOV", value: $settings.lensHalfFov, range: 60...120, suffix: "°")
                    slider("Circle scale", value: $settings.circleScale, range: 0.70...1.15, suffix: "x")
                    Picker("Projection", selection: $settings.projection) {
                        ForEach(FisheyeProjection.allCases) { projection in
                            Text(projection.title).tag(projection)
                        }
                    }
                } header: {
                    Text("Lens geometry")
                } footer: {
                    Text("Start with the lens half FOV and circle scale. Use a straight door frame or table edge as the reference.")
                }

                Section {
                    slider("Radial k1", value: $settings.k1, range: -0.35...0.35, suffix: "")
                    slider("Radial k2", value: $settings.k2, range: -0.20...0.20, suffix: "")
                    slider("Center X", value: $settings.centerX, range: -0.12...0.12, suffix: "")
                    slider("Center Y", value: $settings.centerY, range: -0.12...0.12, suffix: "")
                    slider("Edge feather", value: $settings.edgeFeather, range: 0...0.12, suffix: "")
                    Button("Reset lens calibration", role: .destructive) {
                        settings.resetLens()
                    }
                } header: {
                    Text("Manual calibration")
                } footer: {
                    Text("Adjust k1/k2 until straight lines become straight. Center X/Y correct a clip-on lens that is not perfectly centered.")
                }

                Section {
                    slider("Sensor smoothing", value: Binding(
                        get: { Float(motion.smoothing) },
                        set: { motion.smoothing = Double($0) }
                    ), range: 0...0.30, suffix: "s")
                    slider("Display smoothing", value: Binding(
                        get: { Float(motion.displaySmoothing) },
                        set: { motion.displaySmoothing = Double($0) }
                    ), range: 0...0.18, suffix: "s")
                    if motion.mode == .follow {
                        slider("Follow time", value: Binding(
                            get: { Float(motion.dampingTime) },
                            set: { motion.dampingTime = Double($0) }
                        ), range: 0.2...3.0, suffix: "s")
                    }
                    Button("Recenter / lock current direction") {
                        motion.recenter()
                    }
                } header: {
                    Text("Stabilization")
                } footer: {
                    Text("More smoothing is steadier but adds delay. Display smoothing removes tiny sample-to-display timing jumps.")
                }
            }
            .navigationTitle("SteadyFisheye")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done", action: dismiss)
                }
            }
        }
    }

    private func slider(_ title: String,
                        value: Binding<Float>,
                        range: ClosedRange<Float>,
                        suffix: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: "%.3f", value.wrappedValue) + suffix)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(value: value, in: range)
        }
        .padding(.vertical, 2)
    }
}

private struct MetalPreview: UIViewRepresentable {
    let camera: CameraService
    let settings: FisheyeSettings
    let motion: MotionStabilizer
    @Binding var failed: Bool

    func makeCoordinator() -> Coordinator { Coordinator(camera: camera) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.backgroundColor = .black
        view.contentMode = .scaleAspectFill
        do {
            let renderer = try MetalRenderer(view: view, settings: settings, motion: motion)
            context.coordinator.renderer = renderer
            camera.onFrame = { [weak renderer] pixelBuffer, timestamp in
                renderer?.enqueue(pixelBuffer: pixelBuffer, timestamp: timestamp)
            }
        } catch {
            failed = true
        }
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        let scale = uiView.window?.screen.scale ?? UIScreen.main.scale
        let size = uiView.bounds.size
        if size.width > 0, size.height > 0 {
            uiView.drawableSize = CGSize(width: size.width * scale, height: size.height * scale)
        }
    }

    static func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.camera.onFrame = nil
        uiView.delegate = nil
    }

    final class Coordinator {
        let camera: CameraService
        var renderer: MetalRenderer?

        init(camera: CameraService) { self.camera = camera }
    }
}
