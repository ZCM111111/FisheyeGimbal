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
            Text("控制台")
                .font(Theme.label(13))
                .tracking(1.0)
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
            .accessibilityLabel("关闭控制面板")
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
            ReadoutRow(label: "信号源", value: camera.formatText)
            ReadoutRow(label: "帧率",
                       value: "\(camera.measuredFPS) / 60 帧",
                       valueColor: camera.measuredFPS >= 55 ? Theme.success : Theme.warning)
            ReadoutRow(label: "镜头", value: camera.lens.title)
            ReadoutRow(label: "锁定",
                       value: motion.locked ? "已锁定" : "无陀螺仪",
                       valueColor: motion.available ? Theme.success : Theme.danger)
            if motion.mode == .horizon {
                // Live gravity reference: this keeps updating even while the
                // phone is held still, and it is never latched by a button.
                ReadoutRow(label: "倾斜",
                           value: String(format: "%+.1f°  实时", Double(motion.horizonTilt)),
                           valueColor: abs(motion.horizonTilt) < 0.5 ? Theme.success : Theme.accent)
            }
        }
        .padding(Theme.sp3)
        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
    }

    // MARK: - Groups

    private var cameraGroup: some View {
        VStack(alignment: .leading, spacing: Theme.sp3) {
            groupTitle("相机")

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
        PanelSection(title: "镜头几何", expanded: $showGeometry) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "输出视场", unit: "°", digits: 0,
                                 value: $settings.outputFov, range: 30...130)
                IndustrialSlider(title: "镜头半视场", unit: "°", digits: 0,
                                 value: $settings.lensHalfFov, range: 60...120)
                IndustrialSlider(title: "成像圈比例", unit: "×", digits: 2,
                                 value: $settings.circleScale, range: 0.70...1.15)

                Segmented(values: FisheyeProjection.allCases,
                          titles: FisheyeProjection.allCases.map { $0.title },
                          selection: $settings.projection)

                // Portrait screens are much taller than wide, so the display
                // mapping decides whether the fisheye circle is cropped to
                // fill the screen or fitted inside it.
                Segmented(values: [true, false],
                          titles: ["铺满屏幕", "完整视野"],
                          selection: $settings.fillScreen)
            }
        }
    }

    private var finishGroup: some View {
        PanelSection(title: "画质", expanded: $showFinish) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "锐度", unit: "", digits: 2,
                                 value: $settings.sharpness, range: 0...0.6)
                IndustrialSlider(title: "局部对比", unit: "", digits: 2,
                                 value: $settings.localContrast, range: 0...0.5)
                IndustrialSlider(title: "雾化 / 镜头脏污", unit: "", digits: 2,
                                 value: $settings.hazeCompensation, range: 0...0.5)
            }
        }
    }

    private var calibrationGroup: some View {
        PanelSection(title: "手动标定", expanded: $showCalibration) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                IndustrialSlider(title: "径向 K1", unit: "", digits: 3,
                                 value: $settings.k1, range: -0.35...0.35)
                IndustrialSlider(title: "径向 K2", unit: "", digits: 3,
                                 value: $settings.k2, range: -0.20...0.20)
                IndustrialSlider(title: "中心 X", unit: "", digits: 3,
                                 value: $settings.centerX, range: -0.12...0.12)
                IndustrialSlider(title: "中心 Y", unit: "", digits: 3,
                                 value: $settings.centerY, range: -0.12...0.12)
                IndustrialSlider(title: "边缘羽化", unit: "", digits: 3,
                                 value: $settings.edgeFeather, range: 0...0.12)

                PanelButton(title: "重置镜头", isDestructive: true) {
                    settings.resetLens()
                }
            }
        }
    }

    private var stabilizationGroup: some View {
        PanelSection(title: "防抖", expanded: $showStabilization) {
            VStack(alignment: .leading, spacing: Theme.sp4) {
                if motion.mode == .follow {
                    IndustrialSlider(title: "跟随时间", unit: "秒", digits: 2,
                                     value: Binding(get: { Float(motion.dampingTime) },
                                                    set: { motion.dampingTime = Double($0) }),
                                     range: 0.2...3.0)
                }

                // Any value above zero deliberately leaves part of the shake
                // uncorrected, so the default is a hard zero.
                IndustrialSlider(title: "锁定延迟", unit: "秒", digits: 2,
                                 value: Binding(get: { Float(motion.displaySmoothing) },
                                                set: { motion.displaySmoothing = Double($0) }),
                                 range: 0...0.18)

                if motion.mode == .horizon {
                    Text("地平线持续参考重力，无需手动校准")
                        .font(Theme.label(10))
                        .tracking(0.3)
                        .foregroundColor(Theme.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    PanelButton(title: "重新回中") {
                        motion.recenter()
                    }
                }
            }
        }
    }

    private func groupTitle(_ text: String) -> some View {
        Text(text.uppercased())
            .font(Theme.label(11))
            .tracking(0.5)
            .foregroundColor(Theme.textTertiary)
    }
}
