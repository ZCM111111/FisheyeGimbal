import SwiftUI
import Foundation

/// The calibration console. Deliberately not a `Form`: the stock iOS settings
/// look was replaced with a compact industrial-editorial panel that can float
/// over the live preview without hiding the centre of the frame.
struct ControlPanel: View {

    @ObservedObject var settings: FisheyeSettings
    @ObservedObject var motion: MotionStabilizer
    @ObservedObject var camera: CameraService
    let dismiss: () -> Void

    @State private var showGeometry = true
    @State private var showFinish = true
    @State private var showCalibration = false
    @State private var showStabilization = false

    var body: some View {
        VStack(spacing: 0) {
            header

            separator

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: Theme.sp5) {
                    telemetry
                    cameraGroup
                    geometryGroup
                    finishGroup
                    calibrationGroup
                    stabilizationGroup
                }
                .padding(.horizontal, Theme.sp4)
                .padding(.top, Theme.sp4)
                .padding(.bottom, Theme.sp5)
            }
            .frame(maxHeight: 470)
        }
        .frame(width: 330)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.rLg))
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: Theme.sp3) {
            Text("CONTROL")
                .font(Theme.label(12))
                .tracking(2.0)
                .foregroundColor(Theme.text)

            Spacer(minLength: Theme.sp2)

            Text(Bundle.main.buildStamp)
                .font(Theme.value(10))
                .foregroundColor(Theme.accent)

            Button(action: dismiss) {
                Image(systemName: "xmark")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(Theme.textSecondary)
                    .frame(width: 28, height: 28)
                    .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Close control panel")
        }
        .padding(.horizontal, Theme.sp4)
        .padding(.vertical, Theme.sp3)
    }

    private var separator: some View {
        Rectangle()
            .fill(Theme.line)
            .frame(height: 1)
    }

    // MARK: - Telemetry (data first)

    private var telemetry: some View {
        VStack(alignment: .leading, spacing: Theme.sp2) {
            ReadoutRow(label: "source", value: camera.formatText)
            ReadoutRow(label: "rate",
                       value: "\(camera.measuredFPS) / 60 fps",
                       valueColor: camera.measuredFPS >= 55 ? Theme.success : Theme.warning)
            ReadoutRow(label: "lens", value: camera.lens.title)
            ReadoutRow(label: "lock",
                       value: motion.locked ? "locked" : "no imu",
                       valueColor: motion.available ? Theme.success : Theme.danger)
        }
        .padding(Theme.sp3)
        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
    }

    // MARK: - Groups

    private var cameraGroup: some View {
        VStack(alignment: .leading, spacing: Theme.sp3) {
            groupTitle("camera")

            Segmented(values: CameraService.Lens.allCases,
                      titles: CameraService.Lens.allCases.map { $0.title },
                      selection: Binding(
                        get: { camera.lens },
                        set: { camera.setLens($0) }))

            Segmented(values: MotionStabilizer.Mode.allCases,
                      titles: MotionStabilizer.Mode.allCases.map { $0.title },
                      selection: Binding(
                        get: { motion.mode },
                        set: { motion.setMode($0) }))
        }
    }

    private var geometryGroup: some View {
        PanelSection(title: "lens geometry", expanded: $showGeometry) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "output fov", unit: "°", digits: 0,
                                 value: $settings.outputFov, range: 30...130)
                IndustrialSlider(title: "lens half fov", unit: "°", digits: 0,
                                 value: $settings.lensHalfFov, range: 60...120)
                IndustrialSlider(title: "circle scale", unit: "x", digits: 2,
                                 value: $settings.circleScale, range: 0.70...1.15)

                Segmented(values: FisheyeProjection.allCases,
                          titles: FisheyeProjection.allCases.map { $0.title },
                          selection: $settings.projection)
            }
        }
    }

    private var finishGroup: some View {
        PanelSection(title: "image finish", expanded: $showFinish) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "sharpness", unit: "", digits: 2,
                                 value: $settings.sharpness, range: 0...0.6)
                IndustrialSlider(title: "local contrast", unit: "", digits: 2,
                                 value: $settings.localContrast, range: 0...0.5)
                IndustrialSlider(title: "haze / dirty lens", unit: "", digits: 2,
                                 value: $settings.hazeCompensation, range: 0...0.5)
            }
        }
    }

    private var calibrationGroup: some View {
        PanelSection(title: "manual calibration", expanded: $showCalibration) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "radial k1", unit: "", digits: 3,
                                 value: $settings.k1, range: -0.35...0.35)
                IndustrialSlider(title: "radial k2", unit: "", digits: 3,
                                 value: $settings.k2, range: -0.20...0.20)
                IndustrialSlider(title: "center x", unit: "", digits: 3,
                                 value: $settings.centerX, range: -0.12...0.12)
                IndustrialSlider(title: "center y", unit: "", digits: 3,
                                 value: $settings.centerY, range: -0.12...0.12)
                IndustrialSlider(title: "edge feather", unit: "", digits: 3,
                                 value: $settings.edgeFeather, range: 0...0.12)

                PanelButton(title: "reset lens", isDestructive: true) {
                    settings.resetLens()
                }
            }
        }
    }

    private var stabilizationGroup: some View {
        PanelSection(title: "stabilization", expanded: $showStabilization) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "sensor smoothing", unit: "s", digits: 2,
                                 value: Binding(get: { Float(motion.smoothing) },
                                                set: { motion.smoothing = Double($0) }),
                                 range: 0...0.30)
                IndustrialSlider(title: "display smoothing", unit: "s", digits: 2,
                                 value: Binding(get: { Float(motion.displaySmoothing) },
                                                set: { motion.displaySmoothing = Double($0) }),
                                 range: 0...0.18)

                if motion.mode == .follow {
                    IndustrialSlider(title: "follow time", unit: "s", digits: 2,
                                     value: Binding(get: { Float(motion.dampingTime) },
                                                    set: { motion.dampingTime = Double($0) }),
                                     range: 0.2...3.0)
                }

                PanelButton(title: "recenter") {
                    motion.recenter()
                }
            }
        }
    }

    private func groupTitle(_ text: String) -> some View {
        Text(text.uppercased())
            .font(Theme.label(10))
            .tracking(1.2)
            .foregroundColor(Theme.textTertiary)
    }
}
