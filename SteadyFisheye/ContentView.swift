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
    @State private var panelOffset: CGSize = .zero
    @State private var dragStartOffset: CGSize?

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()

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

            // The console floats at the top-trailing corner so the centre of
            // the frame stays unobstructed, and it can be dragged anywhere.
            if showControls {
                VStack {
                    HStack(alignment: .top) {
                        Spacer(minLength: 0)
                        ControlPanel(settings: settings,
                                     motion: motion,
                                     camera: camera,
                                     dismiss: {
                                         withAnimation(.easeOut(duration: 0.2)) { showControls = false }
                                     })
                            .offset(panelOffset)
                            .gesture(
                                DragGesture()
                                    .onChanged { value in
                                        if dragStartOffset == nil { dragStartOffset = panelOffset }
                                        let start = dragStartOffset ?? panelOffset
                                        panelOffset = CGSize(
                                            width: start.width + value.translation.width,
                                            height: start.height + value.translation.height
                                        )
                                    }
                                    .onEnded { _ in dragStartOffset = nil }
                            )
                            .transition(.scale(scale: 0.96).combined(with: .opacity))
                    }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, Theme.sp3)
                .padding(.top, 62)
                .padding(.bottom, 70)
            }

            if rendererFailed {
                Theme.bgElevated.ignoresSafeArea()
                VStack(spacing: Theme.sp3) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title)
                        .foregroundColor(Theme.danger)
                    Text("METAL UNAVAILABLE")
                        .font(Theme.label(12))
                        .tracking(1.4)
                        .foregroundColor(Theme.text)
                }
            }
        }
        .onAppear { app.start() }
        .onDisappear { app.stop() }
    }

    // MARK: - Status bar

    private var statusBar: some View {
        HStack(spacing: Theme.sp3) {
            Circle()
                .fill(camera.running ? Theme.success : Theme.warning)
                .frame(width: 7, height: 7)

            VStack(alignment: .leading, spacing: 1) {
                Text(camera.running ? "LIVE" : "WAITING")
                    .font(Theme.label(11))
                    .tracking(1.4)
                    .foregroundColor(Theme.text)
                Text(camera.formatText)
                    .font(Theme.value(9))
                    .foregroundColor(Theme.textTertiary)
            }

            Text(Bundle.main.buildStamp)
                .font(Theme.value(9))
                .foregroundColor(Theme.textTertiary)

            Spacer(minLength: Theme.sp2)

            Label(motion.locked ? "LOCKED" : "NO IMU", systemImage: "gyroscope")
                .font(Theme.label(10))
                .tracking(0.8)
                .foregroundColor(motion.available ? Theme.success : Theme.warning)

            Button { motion.recenter() } label: {
                Image(systemName: "scope")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(Theme.text)
                    .frame(width: 32, height: 32)
                    .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Recenter lock")

            Button { withAnimation(.easeOut(duration: 0.2)) { showControls.toggle() } } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(showControls ? Theme.onAccent : Theme.text)
                    .frame(width: 32, height: 32)
                    .background(showControls ? Theme.accent : Theme.surface2,
                                in: RoundedRectangle(cornerRadius: Theme.rBase))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Toggle control panel")
        }
        .padding(.horizontal, Theme.sp3)
        .padding(.vertical, Theme.sp2)
        .background(Theme.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: Theme.rLg))
        .padding(.horizontal, Theme.sp3)
        .padding(.top, Theme.sp2)
    }

    // MARK: - Bottom bar

    private var bottomBar: some View {
        HStack(spacing: Theme.sp4) {
            metric("fov", "\(Int(settings.outputFov))°")
            metric("mode", motion.mode.title.lowercased())
            metric("rate", "\(camera.measuredFPS) fps")
            Spacer(minLength: 0)
            Text("PINCH TO ZOOM")
                .font(Theme.label(8))
                .tracking(1.0)
                .foregroundColor(Theme.textDisabled)
        }
        .padding(.horizontal, Theme.sp3)
        .padding(.vertical, Theme.sp2)
        .background(Theme.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: Theme.rLg))
        .padding(.horizontal, Theme.sp3)
        .padding(.bottom, Theme.sp2)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        HStack(spacing: Theme.sp1) {
            Text(label.uppercased())
                .font(Theme.label(8))
                .tracking(1.0)
                .foregroundColor(Theme.textTertiary)
            Text(value)
                .font(Theme.value(11))
                .foregroundColor(Theme.text)
        }
    }
}

// MARK: - Metal preview host

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

// MARK: - Build marker

extension Bundle {
    /// Short build marker so the running install can be identified on-device
    /// without guessing which IPA was sideloaded.
    var buildStamp: String {
        let version = infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "v\(version).\(build)"
    }
}
