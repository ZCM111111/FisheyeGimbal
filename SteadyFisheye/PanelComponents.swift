import SwiftUI
import Foundation

// MARK: - Segmented control

/// Token-styled segmented control. Replaces the stock iOS picker so the panel
/// keeps one visual language.
struct Segmented<T: Hashable>: View {
    let values: [T]
    let titles: [String]
    @Binding var selection: T

    var body: some View {
        HStack(spacing: 3) {
            ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                let isActive = value == selection
                Button {
                    selection = value
                } label: {
                    Text(index < titles.count ? titles[index] : "")
                        .font(Theme.value(11))
                        .foregroundColor(isActive ? Theme.onAccent : Theme.textSecondary)
                        .frame(maxWidth: .infinity)
                        .frame(height: 26)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.rBase)
                                .fill(isActive ? Theme.accent : Color.clear)
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(3)
        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rLg))
    }
}

// MARK: - Slider

/// Flat capsule slider with a monospaced readout, matching the design language
/// specification of a 28x16 pill thumb and a 2pt track.
struct IndustrialSlider: View {
    let title: String
    let unit: String
    let digits: Int
    @Binding var value: Float
    let range: ClosedRange<Float>

    private var fraction: CGFloat {
        let span = max(range.upperBound - range.lowerBound, 0.0001)
        return CGFloat(min(max((value - range.lowerBound) / span, 0), 1))
    }

    private var readout: String {
        String(format: "%.\(digits)f", value) + unit
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.sp1) {
            HStack(spacing: Theme.sp2) {
                Text(title.uppercased())
                    .font(Theme.label())
                    .tracking(0.8)
                    .foregroundColor(Theme.textSecondary)
                Spacer(minLength: Theme.sp2)
                Text(readout)
                    .font(Theme.value(12))
                    .foregroundColor(Theme.accent)
            }

            GeometryReader { geo in
                let width = max(geo.size.width, 1)
                let thumbWidth: CGFloat = 28
                let travel = max(width - thumbWidth, 1)

                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Theme.surface3)
                        .frame(height: 2)

                    Capsule()
                        .fill(Theme.accent)
                        .frame(width: max(travel * fraction + thumbWidth, thumbWidth), height: 2)
                        .frame(width: width, alignment: .leading)

                    Capsule()
                        .fill(Theme.accent)
                        .frame(width: thumbWidth, height: 16)
                        .shadow(color: Theme.accentGlow, radius: 6)
                        .offset(x: travel * fraction)
                }
                .frame(height: 28)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { gesture in
                            let x = min(max(gesture.location.x - thumbWidth / 2, 0), travel)
                            let span = range.upperBound - range.lowerBound
                            value = range.lowerBound + Float(x / travel) * span
                        }
                )
            }
            .frame(height: 28)
        }
    }
}

// MARK: - Collapsible section

/// Progressive disclosure: low-frequency groups start collapsed so the panel
/// stays compact over the live preview.
struct PanelSection<Content: View>: View {
    let title: String
    @Binding var expanded: Bool
    let content: () -> Content

    init(title: String,
         expanded: Binding<Bool>,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self._expanded = expanded
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.sp2) {
            Button {
                withAnimation(.easeOut(duration: 0.22)) { expanded.toggle() }
            } label: {
                HStack(spacing: Theme.sp2) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(expanded ? Theme.accent : Theme.textTertiary)
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                    Text(title.uppercased())
                        .font(Theme.label(10))
                        .tracking(1.2)
                        .foregroundColor(Theme.textTertiary)
                    Spacer(minLength: 0)
                }
                .frame(height: 22)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expanded {
                content()
                    .padding(.leading, Theme.sp3)
            }
        }
    }
}

// MARK: - Small pieces

/// One label + monospaced value row, used for telemetry.
struct ReadoutRow: View {
    let label: String
    let value: String
    var valueColor: Color = Theme.text

    var body: some View {
        HStack(spacing: Theme.sp2) {
            Text(label.uppercased())
                .font(Theme.label(9))
                .tracking(0.8)
                .foregroundColor(Theme.textTertiary)
            Spacer(minLength: Theme.sp2)
            Text(value)
                .font(Theme.value(11))
                .foregroundColor(valueColor)
        }
    }
}

/// Toggle rendered as a compact chip instead of a stock iOS switch.
struct ChipToggle: View {
    let title: String
    @Binding var isOn: Bool

    var body: some View {
        Button {
            isOn.toggle()
        } label: {
            Text(title.uppercased())
                .font(Theme.label(10))
                .tracking(0.8)
                .foregroundColor(isOn ? Theme.onAccent : Theme.textSecondary)
                .frame(maxWidth: .infinity)
                .frame(height: 26)
                .background(
                    RoundedRectangle(cornerRadius: Theme.rBase)
                        .fill(isOn ? Theme.accent : Theme.surface2)
                )
        }
        .buttonStyle(.plain)
    }
}

/// Action button in the panel's restrained grey style.
struct PanelButton: View {
    let title: String
    var isDestructive = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.sp2) {
                if isDestructive {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(Theme.danger)
                }
                Text(title.uppercased())
                    .font(Theme.label(10))
                    .tracking(0.8)
                    .foregroundColor(Theme.textSecondary)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 30)
            .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.rBase))
        }
        .buttonStyle(.plain)
    }
}
