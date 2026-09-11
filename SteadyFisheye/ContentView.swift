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
        HStack(spacing: Theme.sp2) {
            Circle()
                .fill(camera.running ? Theme.success : Theme.warning)
                .frame(width: 7, height: 7)

            Text(camera.running ? "LIVE" : "WAIT")
                .font(Theme.label(11))
                .tracking(1.2)
                .foregroundColor(Theme.text)
                .lineLimit(1)
                .fixedSize()

            Text(camera.formatText)
                .font(Theme.value(9))
                .foregroundColor(Theme.textTertiary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .truncationMode(.middle)

            Spacer(minLength: Theme.sp1)

            Text(Bundle.main.buildStamp)
                .font(Theme.value(9))
                .foregroundColor(Theme.textTertiary)
                .lineLimit(1)
                .fixedSize()

            Image(systemName: "gyroscope")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(motion.available ? Theme.success : Theme.warning)
                .frame(width: 16)
                .accessibilityLabel(motion.locked ? "Locked" : "No motion data")

            if motion.mode == .horizon {
                // Gravity keeps the horizon level on its own, so there is no
                // reference to latch and nothing to press.
                Image(systemName: "level")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(abs(motion.horizonTilt) < 0.5 ? Theme.success : Theme.accent)
                    .frame(width: 32, height: 32)
                    .accessibilityLabel("Horizon locked to gravity")
            } else {
                Button { motion.recenter() } label: {
                    Image(systemName: "scope")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(Theme.text)
                        .frame(width: 32, height: 32)
                        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Recenter lock")
            }

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
        .lineLimit(1)
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
            metric("mode", motion.mode.title)
            if motion.mode == .horizon {
                metric("tilt", String(format: "%+.1f°", Double(motion.horizonTilt)))
            }
            metric("rate", "\(camera.measuredFPS) fps")
            Spacer(minLength: 0)
        }
        .lineLimit(1)
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
        .lineLimit(1)
        .fixedSize()
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
        // Drawable sizing is left to MTKView's auto-resize. Assigning
        // drawableSize by hand during layout could install a stale drawable,
        // which showed up as the image sitting in a black box mid-screen.
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
