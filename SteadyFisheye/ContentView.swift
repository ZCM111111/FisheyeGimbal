import SwiftUI
import MetalKit
import UIKit

struct ContentView: View {
    @ObservedObject var app: CameraApp
    @ObservedObject var settings: FisheyeSettings
    @ObservedObject var motion: MotionStabilizer
    @ObservedObject var camera: CameraService
    @ObservedObject var recorder: VideoRecorder

    /// Closed on launch so the preview is unobstructed; the labelled 校正
    /// button opens the fisheye console.
    @State private var showControls = false
    @State private var rendererFailed = false
    @State private var baseFov: Float?
    @State private var panelOffset: CGSize = .zero
    @State private var dragStartOffset: CGSize?
    /// Where the last tap landed, in preview coordinates, for the reticle.
    @State private var focusPoint: CGPoint?
    @State private var showExposure = false
    @State private var ignoreNextTap = false
    @State private var renderer: MetalRenderer?
    @State private var toast: String?
    /// Set while the share sheet is up, with the recording to hand over.
    @State private var shareURL: URL?
    /// Held in a reference box: bumping a plain `@State` on every drag event
    /// invalidates the whole view, which is not something a slider should do.
    @State private var tokens = CancellationTokens()
    @State private var containerSize = CGSize(width: 393, height: 852)

    var body: some View {
        ZStack {
            Theme.bg.ignoresSafeArea()

            MetalPreview(camera: camera,
                         settings: settings,
                         motion: motion,
                         recorder: recorder,
                         failed: $rendererFailed,
                         onRendererReady: { renderer = $0 })
                .ignoresSafeArea()
                .simultaneousGesture(
                    SpatialTapGesture()
                        .onEnded { value in handleTap(at: value.location) }
                )
                .onLongPressGesture(minimumDuration: 0.55) { toggleAEAFLock() }
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            if baseFov == nil { baseFov = settings.outputFov }
                            let next = (baseFov ?? settings.outputFov) / Float(value)
                            settings.outputFov = min(max(next, 30), 130)
                        }
                        .onEnded { _ in baseFov = nil }
                )

            // Straight screen-space lines to judge real edges against while
            // calibrating by hand.
            if settings.showGrid {
                ReferenceGrid()
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
            }

            // Focus reticle and exposure slider, drawn straight over the frame
            // the way the system camera does it.
            if let point = focusPoint {
                FocusReticle()
                    .position(point)
                    .allowsHitTesting(false)

                if showExposure {
                    ExposureSlider(value: camera.exposureBias,
                                   range: camera.exposureBiasRange,
                                   onApply: { camera.setExposureBias($0) },
                                   onInteraction: { scheduleFocusHide() })
                        .position(exposureSliderPosition(for: point))
                }
            }

            VStack(spacing: 0) {
                statusBar
                Spacer()
                if let toast = toast {
                    Text(toast)
                        .font(Theme.label(11))
                        .tracking(0.3)
                        .foregroundColor(Theme.text)
                        .padding(.horizontal, Theme.sp3)
                        .padding(.vertical, Theme.sp2)
                        .background(Theme.surface.opacity(0.94),
                                    in: RoundedRectangle(cornerRadius: Theme.rBase))
                        .padding(.bottom, Theme.sp2)
                        .transition(.opacity)
                }
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
                                         withAnimation(.easeOut(duration: 0.2)) {
                                             showControls = false
                                             // Never let a dragged panel come
                                             // back off-screen next time.
                                             panelOffset = .zero
                                         }
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
                    Text("图形渲染不可用")
                        .font(Theme.label(13))
                        .tracking(0.5)
                        .foregroundColor(Theme.text)
                }
            }
        }
        .onAppear {
            // The preview is the product: the screen must not dim while the
            // user is framing a shot.
            UIApplication.shared.isIdleTimerDisabled = true
            // Ask for the photo library now, so the first recording saves
            // itself without a permission prompt in the middle.
            recorder.preparePhotoAccess()
            // Audio from the capture pipeline goes straight into the recorder.
            // Hopped to the main thread because all recorder state lives there,
            // and appending from the audio queue would race with stop().
            camera.onAudioSample = { [weak recorder] buffer in
                DispatchQueue.main.async {
                    recorder?.appendAudio(buffer)
                }
            }
            app.start()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            camera.onAudioSample = nil
            if recorder.isRecording { recorder.stop() }
            app.stop()
        }
        .onChange(of: recorder.message) { value in
            if let value = value { showToast(value) }
        }
        .background(
            GeometryReader { geo in
                Color.clear
                    .onAppear { containerSize = geo.size }
                    .onChange(of: geo.size) { containerSize = $0 }
            }
        )
        .sheet(isPresented: Binding(get: { shareURL != nil },
                                    set: { if !$0 { shareURL = nil } })) {
            if let url = shareURL {
                ActivityView(items: [url])
            }
        }
    }

    /// Tiny box so token bumps do not invalidate the view.
    final class CancellationTokens {
        private var value = 0
        func bump() -> Int {
            value += 1
            return value
        }
        func isCurrent(_ token: Int) -> Bool { token == value }
    }

    // MARK: - Recording

    private func toggleRecording() {
        if recorder.isRecording {
            recorder.stop()
            return
        }
        guard let size = renderer?.preferredRecordingSize() else {
            showToast("渲染器还没准备好")
            return
        }
        if !recorder.start(size: size, audioSettings: camera.audioWriterSettings) {
            showToast(recorder.message ?? "录像启动失败")
        }
    }

    private var recordingTimeText: String {
        let total = Int(recorder.duration.rounded())
        return String(format: "%02d:%02d", total / 60, total % 60)
    }

    private func showToast(_ text: String) {
        let token = tokens.bump()
        withAnimation(.easeOut(duration: 0.2)) { toast = text }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            guard tokens.isCurrent(token) else { return }
            withAnimation(.easeOut(duration: 0.3)) { toast = nil }
        }
    }

    // MARK: - Focus and exposure interaction

    /// A tap means "focus and meter here", exactly like the system camera. The
    /// point has to be traced back through the undistortion and the lock before
    /// the camera can use it.
    private func handleTap(at location: CGPoint) {
        if ignoreNextTap {
            // A long press already handled this gesture; do not also re-focus.
            ignoreNextTap = false
            return
        }
        if camera.aeafLocked {
            // While AE/AF is locked a tap releases it, as the system camera does.
            toggleAEAFLock()
            withAnimation(.easeOut(duration: 0.2)) { focusPoint = nil; showExposure = false }
            return
        }

        // Trace the tap back to the sensor; fall back to the centre so a tap
        // still does something if the lens state is not ready yet.
        let devicePoint = renderer?.devicePoint(forViewPoint: location)
            ?? CGPoint(x: 0.5, y: 0.5)
        camera.focusAndExpose(atDevicePoint: devicePoint)

        withAnimation(.easeOut(duration: 0.15)) {
            focusPoint = location
            showExposure = true
        }
        scheduleFocusHide()
    }

    private func toggleAEAFLock() {
        let locked = !camera.aeafLocked
        camera.setAEAFLocked(locked)
        ignoreNextTap = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { ignoreNextTap = false }
        if locked, focusPoint == nil {
            let center = CGPoint(x: containerSize.width * 0.5,
                                 y: containerSize.height * 0.5)
            withAnimation(.easeOut(duration: 0.15)) { focusPoint = center }
        }
        scheduleFocusHide()
    }

    /// The reticle disappears quickly; the exposure slider lingers, then goes.
    private func scheduleFocusHide() {
        let token = tokens.bump()
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.6) {
            guard tokens.isCurrent(token) else { return }
            withAnimation(.easeOut(duration: 0.3)) {
                focusPoint = nil
                showExposure = false
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 9.0) {
            guard tokens.isCurrent(token) else { return }
            withAnimation(.easeOut(duration: 0.3)) { showExposure = false }
        }
    }

    /// Keeps the slider on screen beside the reticle.
    private func exposureSliderPosition(for point: CGPoint) -> CGPoint {
        let wantsRight = point.x + 62 + 46 < containerSize.width
        let x = wantsRight ? point.x + 62 + 46 : max(point.x - 62 - 46, 46)
        let boundedX = min(max(x, 46), max(containerSize.width - 46, 46))
        return CGPoint(x: boundedX,
                       y: min(max(point.y, 100), max(containerSize.height - 110, 100)))
    }

    // MARK: - Status bar

    private var statusBar: some View {
        HStack(spacing: Theme.sp2) {
            Circle()
                .fill(camera.running ? Theme.success : Theme.warning)
                .frame(width: 7, height: 7)

            Text(camera.running ? "实时" : "等待")
                .font(Theme.label(12))
                .tracking(0.5)
                .foregroundColor(Theme.text)
                .lineLimit(1)
                .fixedSize()

            Text(camera.formatText)
                .font(Theme.value(9))
                .foregroundColor(Theme.textTertiary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .truncationMode(.middle)

            if camera.aeafLocked {
                Button { toggleAEAFLock() } label: {
                    Text("AE/AF")
                        .font(Theme.label(9))
                        .tracking(0.3)
                        .foregroundColor(Theme.onAccent)
                        .padding(.horizontal, 6)
                        .frame(height: 18)
                        .background(Theme.accent, in: Capsule())
                }
                .buttonStyle(.plain)
                .fixedSize()
                .accessibilityLabel("解除AE/AF锁定")
            }

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
                .accessibilityLabel(motion.locked ? "已锁定" : "无陀螺仪数据")

            if motion.mode == .horizon {
                // Gravity keeps the horizon level on its own, so there is no
                // reference to latch and nothing to press.
                Image(systemName: "level")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(abs(motion.horizonTilt) < 0.5 ? Theme.success : Theme.accent)
                    .frame(width: 32, height: 32)
                    .accessibilityLabel("地平线已锁定到重力")
            } else {
                Button { motion.recenter() } label: {
                    Image(systemName: "scope")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(Theme.text)
                        .frame(width: 32, height: 32)
                        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("重新回中")
            }

            Button { withAnimation(.easeOut(duration: 0.2)) { showControls.toggle() } } label: {
                HStack(spacing: 5) {
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 11, weight: .semibold))
                    Text(showControls ? "收起" : "校正")
                        .font(Theme.label(11))
                        .tracking(0.3)
                }
                .foregroundColor(showControls ? Theme.onAccent : Theme.text)
                .padding(.horizontal, 9)
                .frame(height: 32)
                .background(showControls ? Theme.accent : Theme.surface2,
                            in: RoundedRectangle(cornerRadius: Theme.rBase))
            }
            .buttonStyle(.plain)
            .fixedSize()
            .accessibilityLabel("开关鱼眼校正面板")
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
            metric("视场", "\(Int(settings.outputFov))°")
            metric("模式", motion.mode.title)
            if motion.mode == .horizon {
                metric("倾斜", String(format: "%+.1f°", Double(motion.horizonTilt)))
            }
            metric("帧率", "\(camera.measuredFPS) 帧")
            Spacer(minLength: 0)
            if !recorder.isRecording, recorder.lastRecordingURL != nil {
                Button { shareURL = recorder.lastRecordingURL } label: {
                    Text("分享")
                        .font(Theme.label(11))
                        .tracking(0.3)
                        .foregroundColor(Theme.textSecondary)
                        .padding(.horizontal, 9)
                        .frame(height: 28)
                        .background(Theme.surface2,
                                    in: RoundedRectangle(cornerRadius: Theme.rBase))
                }
                .buttonStyle(.plain)
                .fixedSize()
                .accessibilityLabel("分享刚录制的视频")
            }
            recordButton
        }
        .lineLimit(1)
        .padding(.horizontal, Theme.sp3)
        .padding(.vertical, Theme.sp2)
        .background(Theme.surface.opacity(0.92), in: RoundedRectangle(cornerRadius: Theme.rLg))
        .padding(.horizontal, Theme.sp3)
        .padding(.bottom, Theme.sp2)
    }

    private var recordButton: some View {
        Button { toggleRecording() } label: {
            HStack(spacing: 5) {
                if recorder.isRecording {
                    RoundedRectangle(cornerRadius: 1)
                        .fill(Color.white)
                        .frame(width: 8, height: 8)
                } else {
                    Circle()
                        .fill(Theme.danger)
                        .frame(width: 8, height: 8)
                }
                Text(recorder.isRecording ? recordingTimeText : "录像")
                    .font(Theme.value(11))
                    .foregroundColor(recorder.isRecording ? .white : Theme.text)
                    .lineLimit(1)
            }
            .padding(.horizontal, 9)
            .frame(height: 28)
            .background(recorder.isRecording ? Theme.danger : Theme.surface2,
                        in: RoundedRectangle(cornerRadius: Theme.rBase))
        }
        .buttonStyle(.plain)
        .fixedSize()
        .accessibilityLabel(recorder.isRecording ? "停止录像" : "开始录像")
    }

    private func metric(_ label: String, _ value: String) -> some View {
        HStack(spacing: Theme.sp1) {
            Text(label.uppercased())
                .font(Theme.label(10))
                .tracking(0.3)
                .foregroundColor(Theme.textTertiary)
            Text(value)
                .font(Theme.value(11))
                .foregroundColor(Theme.text)
        }
        .lineLimit(1)
        .fixedSize()
    }
}

/// Straight screen-space lines drawn over the preview.
///
/// Manual calibration means judging by eye whether a real edge came out
/// straight, which is unreliable without something straight to compare it to.
private struct ReferenceGrid: View {
    var body: some View {
        Canvas { context, size in
            var thin = Path()
            let columns = 6
            let rows = 10
            for index in 1..<columns {
                let x = size.width * CGFloat(index) / CGFloat(columns)
                thin.move(to: CGPoint(x: x, y: 0))
                thin.addLine(to: CGPoint(x: x, y: size.height))
            }
            for index in 1..<rows {
                let y = size.height * CGFloat(index) / CGFloat(rows)
                thin.move(to: CGPoint(x: 0, y: y))
                thin.addLine(to: CGPoint(x: size.width, y: y))
            }
            context.stroke(thin,
                           with: .color(Theme.accent.opacity(0.30)),
                           lineWidth: 0.5)

            // Centre cross, for lining the optical axis up against.
            var cross = Path()
            cross.move(to: CGPoint(x: size.width / 2, y: size.height * 0.2))
            cross.addLine(to: CGPoint(x: size.width / 2, y: size.height * 0.8))
            cross.move(to: CGPoint(x: 0, y: size.height / 2))
            cross.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            context.stroke(cross,
                           with: .color(Theme.accent.opacity(0.55)),
                           lineWidth: 1)
        }
    }
}

// MARK: - Metal preview host
private struct MetalPreview: UIViewRepresentable {
    let camera: CameraService
    let settings: FisheyeSettings
    let motion: MotionStabilizer
    let recorder: VideoRecorder
    @Binding var failed: Bool
    let onRendererReady: (MetalRenderer) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(camera: camera) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.backgroundColor = .black
        view.contentMode = .scaleAspectFill
        do {
            let renderer = try MetalRenderer(view: view, settings: settings, motion: motion)
            context.coordinator.renderer = renderer
            renderer.recorder = recorder
            renderer.coverageHandler = { [weak camera] percent in
                camera?.reportCoverage(percent)
            }
            // Hand the renderer back so a tap can be traced to the sensor. The
            // hand-off is deferred: mutating state while the view is being
            // built would be a SwiftUI violation.
            DispatchQueue.main.async { onRendererReady(renderer) }
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
