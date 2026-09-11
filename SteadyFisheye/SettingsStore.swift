import Foundation

/// Everything the app measures or adjusts for one lens.
///
/// Calibration is per lens on purpose: the clip-on sits in front of a specific
/// camera, and the 0.5x and 1x lenses have completely different geometry, so
/// one lens's measured values must never be applied to the other.
struct LensProfile: Codable {
    var lensHalfFov: Float = 90
    var circleScale: Float = 1
    var outputFov: Float = 78
    var projection: Int = 0
    var k1: Float = 0
    var k2: Float = 0
    var centerX: Float = 0
    var centerY: Float = 0
    var edgeFeather: Float = 0.025
    var fillScreen: Bool = true
    var showGrid: Bool = false
    var sharpness: Float = 0.18
    var localContrast: Float = 0.10
    var hazeCompensation: Float = 0.05
}

/// Persists calibration and preferences between launches.
///
/// Stored in `UserDefaults` rather than a file: the payload is a handful of
/// numbers, and it survives reinstalls over the same bundle identifier, so a
/// calibrated lens does not have to be measured again after an update.
enum SettingsStore {

    private static let profilesKey = "lensProfiles.v1"
    private static let activeLensKey = "activeLens.v1"
    private static let stabilizerModeKey = "stabilizerMode.v1"

    static func loadProfiles() -> [String: LensProfile] {
        guard let data = UserDefaults.standard.data(forKey: profilesKey),
              let decoded = try? JSONDecoder().decode([String: LensProfile].self, from: data) else {
            return [:]
        }
        return decoded
    }

    static func saveProfiles(_ profiles: [String: LensProfile]) {
        guard let data = try? JSONEncoder().encode(profiles) else { return }
        UserDefaults.standard.set(data, forKey: profilesKey)
    }

    static func loadActiveLens() -> String? {
        UserDefaults.standard.string(forKey: activeLensKey)
    }

    static func saveActiveLens(_ key: String) {
        UserDefaults.standard.set(key, forKey: activeLensKey)
    }

    static func loadStabilizerMode() -> String? {
        UserDefaults.standard.string(forKey: stabilizerModeKey)
    }

    static func saveStabilizerMode(_ raw: String) {
        UserDefaults.standard.set(raw, forKey: stabilizerModeKey)
    }
}
