import SwiftUI

/// Industrial-editorial design tokens adapted to native SwiftUI.
///
/// The web-oriented token names from the design language are preserved
/// (bg / surface / text / accent / line), but expressed as native values.
/// Rules kept from the source design language:
///  - no pure black or pure white
///  - containers separate by surface depth, not borders or shadows
///  - accent only for values, active states and slider fill
///  - small radii (4pt controls, 6pt containers), no large rounded corners
enum Theme {

    // MARK: - Surfaces

    static let bg = Color(hex: 0x0A0A0A)
    static let bgElevated = Color(hex: 0x0F0F0F)
    static let surface = Color(hex: 0x141414)
    static let surface2 = Color(hex: 0x1A1A1A)
    static let surface3 = Color(hex: 0x222222)
    static let surfaceHover = Color(hex: 0x282828)

    // MARK: - Text

    static let text = Color(hex: 0xF0F0F0)
    static let textSecondary = Color(hex: 0xA0A0A0)
    static let textTertiary = Color(hex: 0x606060)
    static let textDisabled = Color(hex: 0x404040)

    // MARK: - Accent (dark theme keeps a warm industrial accent)

    static let accent = Color(hex: 0xFFD60A)
    static let accentPressed = Color(hex: 0xD4B008)
    static let accentDim = Color(hex: 0xFFD60A).opacity(0.08)
    static let accentGlow = Color(hex: 0xFFD60A).opacity(0.25)
    static let onAccent = Color.black.opacity(0.75)

    // MARK: - Semantic

    static let danger = Color(hex: 0xFF3B30)
    static let success = Color(hex: 0x30D158)
    static let warning = Color(hex: 0xFF9F0A)

    // MARK: - Lines

    static let line = Color.white.opacity(0.06)
    static let lineStrong = Color.white.opacity(0.12)

    // MARK: - Spacing (4pt base)

    static let sp1: CGFloat = 4
    static let sp2: CGFloat = 8
    static let sp3: CGFloat = 12
    static let sp4: CGFloat = 16
    static let sp5: CGFloat = 20
    static let sp6: CGFloat = 24

    // MARK: - Radius

    static let rBase: CGFloat = 4
    static let rLg: CGFloat = 6

    // MARK: - Typography

    /// Uppercase micro label used for section headings and control names.
    static func label(_ size: CGFloat = 10) -> Font {
        .system(size: size, weight: .semibold, design: .default)
    }

    /// Monospaced numeric readout, used for every measured value.
    static func value(_ size: CGFloat = 12) -> Font {
        .system(size: size, weight: .bold, design: .monospaced)
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: 1.0
        )
    }
}
