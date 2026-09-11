import SwiftUI
import MetalKit
import UIKit

struct ContentView: View {
    @ObservedObject var app: CameraApp
    @ObservedObject var settings: FisheyeSettings
    @ObservedObject var motion: MotionStabilizer
    @ObservedObject var camera: CameraService
    @State private var showControls = true
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
                topBar
                Spacer()
                if showControls {
                    ControlPanel(app: app,
                                 settings: settings,
                                 motion: motion,
                                 camera: camera)
                        .frame(maxWidth: 430)
                        .padding(.horizontal, 10)
                        .padding(.bottom, 8)
                } else {
                    bottomBar
                }
            }

            if let error = camera.error {
                VStack {
                    Spacer()
                    Text(error)
                        .font(.footnote)
                        .multilineTextAlignment(.center)
                        .padding(10)
                        .background(.red.opacity(0.88), in: RoundedRectangle(cornerRadius: 8))
                        .padding()
                }
            }

            if rendererFailed {
                Color.black.opacity(0.88).ignoresSafeArea()
                VStack(spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.largeTitle)
                    Text("Metal is unavailable on this device")
                        .font(.headline)
                }
                .foregroundStyle(.white)
            }
        }
        .onAppear { app.start() }
        .onDisappear { app.stop() }
    }

    private var topBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 6) {
                Circle()
                    .fill(camera.running ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(camera.running ? camera.formatText : "Camera stopped")
                    .lineLimit(1)
            }

            Spacer()

            HStack(spacing: 6) {
                Image(systemName: motion.locked ? "gyroscope" : "gyroscope")
                Text(motion.status)
                    .lineLimit(1)
            }
            .foregroundStyle(motion.available ? Color.green : Color.orange)

            Button {
                motion.recenter()
            } label: {
                Label("Recenter", systemImage: "scope")
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)

            Button {
                withAnimation(.easeInOut(duration: 0.15)) { showControls.toggle() }
            } label: {
                Image(systemName: showControls ? "slider.horizontal.3" : "slider.horizontal.3")
            }
            .buttonStyle(.bordered)
            .accessibilityLabel("Camera controls")
        }
        .font(.caption)
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.black.opacity(0.48), in: Capsule())
        .padding(.horizontal, 10)
        .padding(.top, 8)
    }

    private var bottomBar: some View {
        HStack {
            Text(camera.running ? "\(camera.formatText)  |  FOV \(Int(settings.outputFov))°" : "Waiting for camera")
                .font(.caption2.monospaced())
                .foregroundStyle(.white.opacity(0.85))
            Spacer()
            Text(motion.mode.title)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.85))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(.black.opacity(0.48), in: Capsule())
        .padding(.horizontal, 10)
        .padding(.bottom, 8)
    }
}

private struct ControlPanel: View {
    @ObservedObject var app: CameraApp
    @ObservedObject var settings: FisheyeSettings
    @ObservedObject var motion: MotionStabilizer
    @ObservedObject var camera: CameraService

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Stabilized fisheye preview")
                        .font(.headline)
                    Spacer()
                    Button {
                        settings.resetLens()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                    }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Reset lens calibration")
                }

                Picker("Stabilization", selection: Binding(
                    get: { motion.mode },
                    set: { motion.setMode($0) }
                )) {
                    ForEach(MotionStabilizer.Mode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Camera lens", selection: Binding(
                    get: { camera.lens },
                    set: { camera.setLens($0) }
                )) {
                    ForEach(CameraService.Lens.allCases) { lens in
                        Text(lens.title).tag(lens)
                    }
                }
                .pickerStyle(.segmented)

                slider("Output FOV", value: $settings.outputFov, range: 30...130, suffix: "°")
                slider("Lens half FOV", value: $settings.lensHalfFov, range: 60...120, suffix: "°")
                slider("Circle scale", value: $settings.circleScale, range: 0.70...1.15, suffix: "x")

                Picker("Fisheye projection", selection: $settings.projection) {
                    ForEach(FisheyeProjection.allCases) { projection in
                        Text(projection.title).tag(projection)
                    }
                }
                .pickerStyle(.menu)

                Divider().overlay(.white.opacity(0.25))
                Text("Calibration")
                    .font(.caption.bold())
                    .foregroundStyle(.white.opacity(0.75))
                slider("Radial k1", value: $settings.k1, range: -0.35...0.35, suffix: "")
                slider("Radial k2", value: $settings.k2, range: -0.20...0.20, suffix: "")
                slider("Center X", value: $settings.centerX, range: -0.12...0.12, suffix: "")
                slider("Center Y", value: $settings.centerY, range: -0.12...0.12, suffix: "")
                slider("Edge feather", value: $settings.edgeFeather, range: 0...0.12, suffix: "")

                Divider().overlay(.white.opacity(0.25))
                Text("Motion")
                    .font(.caption.bold())
                    .foregroundStyle(.white.opacity(0.75))
                slider("Smoothing", value: Binding(
                    get: { Float(motion.smoothing) },
                    set: { motion.smoothing = Double($0) }
                ), range: 0...0.35, suffix: "s")
                if motion.mode == .follow {
                    slider("Follow time", value: Binding(
                        get: { Float(motion.dampingTime) },
                        set: { motion.dampingTime = Double($0) }
                    ), range: 0.2...3.0, suffix: "s")
                }
            }
            .padding(14)
        }
        .frame(maxHeight: 430)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .environment(\.colorScheme, .dark)
    }

    private func slider(_ title: String,
                        value: Binding<Float>,
                        range: ClosedRange<Float>,
                        suffix: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.caption2)
                Spacer()
                Text(String(format: "%.3f", value.wrappedValue) + suffix)
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.65))
            }
            Slider(value: value, in: range)
        }
    }
}

private struct MetalPreview: UIViewRepresentable {
    let camera: CameraService
    let settings: FisheyeSettings
    let motion: MotionStabilizer
    @Binding var failed: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(camera: camera)
    }

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

        init(camera: CameraService) {
            self.camera = camera
        }
    }
}
