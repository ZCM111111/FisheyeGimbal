import SwiftUI

/// The focal reticle shown where the user tapped, matching the system camera's
/// yellow square that springs in and fades away.
struct FocusReticle: View {
    @State private var settled = false

    var body: some View {
        RoundedRectangle(cornerRadius: 4)
            .strokeBorder(Theme.accent, lineWidth: 1.5)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.black.opacity(0.35), lineWidth: 3)
                    .offset(y: 0.5)
            )
            .frame(width: 74, height: 74)
            .scaleEffect(settled ? 1 : 1.25)
            .opacity(settled ? 1 : 0.5)
            .shadow(color: .black.opacity(0.45), radius: 3, y: 1)
            .onAppear {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.68)) {
                    settled = true
                }
            }
    }
}

/// Vertical exposure compensation control, laid out the way the system camera
/// does it: readout, track with a sun icon underneath, drag to bias.
///
/// While a drag is in flight the thumb is driven by local state instead of by
/// the value the camera reports back. Feeding a high-rate published value back
/// into the view that is currently being dragged rebuilds it under the finger
/// on every event and brings SwiftUI's graph down with it.
struct ExposureSlider: View {
    @Binding var value: Float
    let range: ClosedRange<Float>
    let onInteraction: () -> Void

    @State private var dragValue: Float?
    @State private var lastPing = Date.distantPast

    private var displayed: Float { dragValue ?? value }

    private var fraction: CGFloat {
        let span = max(range.upperBound - range.lowerBound, 0.001)
        return CGFloat(min(max((displayed - range.lowerBound) / span, 0), 1))
    }

    var body: some View {
        VStack(spacing: Theme.sp2) {
            Text(String(format: "%+.1f", Double(displayed)))
                .font(Theme.value(11))
                .foregroundColor(Theme.accent)
                .lineLimit(1)
                .fixedSize()

            GeometryReader { geo in
                let height = max(geo.size.height, 1)
                let travel = max(height - 12, 1)

                ZStack(alignment: .top) {
                    Capsule()
                        .fill(Color.white.opacity(0.28))
                        .frame(width: 2, height: height)

                    Capsule()
                        .fill(Theme.accent)
                        .frame(width: 2, height: max(height * fraction, 1))
                        .frame(height: height, alignment: .bottom)

                    Circle()
                        .fill(Theme.accent)
                        .frame(width: 12, height: 12)
                        .offset(y: travel * (1 - fraction))
                }
                .frame(maxWidth: .infinity)
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { gesture in
                            let y = min(max(gesture.location.y, 0), height)
                            let position = 1 - Float(y / height)
                            let span = range.upperBound - range.lowerBound
                            let next = range.lowerBound + position * span
                            dragValue = next
                            value = next
                            // Keep the auto-hide deadline pushed out while the
                            // finger is down, but do not schedule a timer per
                            // drag event.
                            let now = Date()
                            if now.timeIntervalSince(lastPing) > 1.0 {
                                lastPing = now
                                onInteraction()
                            }
                        }
                        .onEnded { _ in
                            dragValue = nil
                            onInteraction()
                        }
                )
            }
            .frame(width: 34, height: 130)

            Image(systemName: "sun.max.fill")
                .font(.system(size: 12))
                .foregroundColor(Theme.accent)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, Theme.sp2)
        .background(Color.black.opacity(0.32), in: RoundedRectangle(cornerRadius: Theme.rLg))
    }
}

/// "AE/AF 锁定" chip. Long-pressing the preview shows it; tapping it releases
/// the lock, which is how the system camera behaves.
struct AEAFLockChip: View {
    let onRelease: () -> Void

    var body: some View {
        Button(action: onRelease) {
            Text("AE/AF 已锁定")
                .font(Theme.label(11))
                .tracking(0.3)
                .foregroundColor(Theme.onAccent)
                .padding(.horizontal, Theme.sp3)
                .frame(height: 26)
                .background(Theme.accent, in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("解除自动曝光自动对焦锁定")
    }
}
