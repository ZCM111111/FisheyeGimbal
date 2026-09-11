import Foundation
import Combine
import SwiftUI
import simd

enum FisheyeProjection: Int, CaseIterable, Identifiable {
    case equidistant = 0
    case equisolid = 1

    var id: Int { rawValue }
    var title: String {
        switch self {
        case .equidistant: return "等距投影"
        case .equisolid: return "等立体角"
        }
    }
}

struct FisheyeParameters {
    var center: SIMD2<Float>
    var focal: Float
    var maxRadius: Float
    var maxTheta: Float
    var outputFov: Float
    var projection: Float
    var k1: Float
    var k2: Float
    var edgeFeather: Float
    var sharpness: Float
    var localContrast: Float
    var hazeCompensation: Float
}

final class FisheyeSettings: ObservableObject {
    /// Screen fill target for the framing lock, as a fraction of frame width.
    @Published var cabinetFillTarget: Float = 0.8

    /// Whether the lens in use already has stored values, for the panel readout.
    @Published private(set) var hasStoredProfile = false

    /// Saved values per lens, keyed by `CameraService.Lens.rawValue`.
    private var profiles: [String: LensProfile]
    private(set) var activeLensKey: String
    private var autosave: Set<AnyCancellable> = []

    /// The lens whose profile is loaded, so the camera can start on the same one.
    var activeLens: CameraService.Lens {
        CameraService.Lens(rawValue: activeLensKey) ?? .ultraWide
    }

    init() {
        let stored = SettingsStore.loadProfiles()
        profiles = stored
        let key = SettingsStore.loadActiveLens() ?? CameraService.Lens.ultraWide.rawValue
        activeLensKey = key
        if let profile = stored[key] {
            applyProfile(profile)
            hasStoredProfile = true
        }

        // Writes are debounced by a second, so dragging a slider or running the
        // auto-calibration does not hit the disk on every value change.
        objectWillChange
            .debounce(for: .seconds(1), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in self?.persistCurrentProfile() }
            .store(in: &autosave)
    }

    /// Switches profiles. The values in use are stored under the outgoing lens
    /// first, so nothing measured is lost when the user changes lens.
    func activate(lensKey: String) {
        guard lensKey != activeLensKey else { return }
        persistCurrentProfile()
        activeLensKey = lensKey
        SettingsStore.saveActiveLens(lensKey)
        if let profile = profiles[lensKey] {
            applyProfile(profile)
            hasStoredProfile = true
        } else {
            // No measurements for this lens yet: start from neutral geometry
            // rather than carrying the other lens's calibration across.
            applyProfile(LensProfile())
            hasStoredProfile = false
        }
    }

    private func currentProfile() -> LensProfile {
        LensProfile(lensHalfFov: lensHalfFov,
                    circleScale: circleScale,
                    outputFov: outputFov,
                    projection: projection.rawValue,
                    k1: k1,
                    k2: k2,
                    centerX: centerX,
                    centerY: centerY,
                    edgeFeather: edgeFeather,
                    fillScreen: fillScreen,
                    showGrid: showGrid,
                    cabinetFillTarget: cabinetFillTarget,
                    sharpness: sharpness,
                    localContrast: localContrast,
                    hazeCompensation: hazeCompensation)
    }

    private func applyProfile(_ profile: LensProfile) {
        lensHalfFov = profile.lensHalfFov
        circleScale = profile.circleScale
        outputFov = profile.outputFov
        projection = FisheyeProjection(rawValue: profile.projection) ?? .equidistant
        k1 = profile.k1
        k2 = profile.k2
        centerX = profile.centerX
        centerY = profile.centerY
        edgeFeather = profile.edgeFeather
        fillScreen = profile.fillScreen
        showGrid = profile.showGrid
        cabinetFillTarget = profile.cabinetFillTarget
        sharpness = profile.sharpness
        localContrast = profile.localContrast
        hazeCompensation = profile.hazeCompensation
    }

    private func persistCurrentProfile() {
        profiles[activeLensKey] = currentProfile()
        SettingsStore.saveProfiles(profiles)
        // Guarded: `@Published` fires on every assignment, so setting this
        // unconditionally would re-trigger the debounced save forever.
        if !hasStoredProfile { hasStoredProfile = true }
    }

    @Published var projection: FisheyeProjection = .equidistant
    /// Half of the angular coverage the glass delivers. This does not create
    /// black edges any more (fill mode pins samples to the rim), so its job is
    /// purely geometric: how many degrees of real lens sit inside the image
    /// circle. Too small stretches the picture, too large squeezes it.
    @Published var lensHalfFov: Float = 90
    @Published var circleScale: Float = 1.0
    @Published var outputFov: Float = 78
    @Published var k1: Float = 0
    @Published var k2: Float = 0
    @Published var centerX: Float = 0
    @Published var centerY: Float = 0
    @Published var edgeFeather: Float = 0.025

    /// Display mapping. `true` derives the projection focal from the long edge
    /// so the corrected image crops to fill the whole screen (portrait phone
    /// screens are much taller than wide). `false` derives it from the short
    /// edge, which fits the widest view but leaves black bars above/below.
    @Published var fillScreen: Bool = true

    /// Screen-space reference grid over the preview.
    ///
    /// Calibration is manual, and judging whether a real edge is straight by eye
    /// is unreliable. A fixed straight grid to compare against turns it into
    /// something you can actually see.
    @Published var showGrid: Bool = false

    // Image finishing controls. These are deliberately conservative because
    // sharpening cannot recover detail lost to a soft sensor or dirty glass.
    @Published var sharpness: Float = 0.18
    @Published var localContrast: Float = 0.10
    @Published var hazeCompensation: Float = 0.05

    /// Centre saved before the last measurement, so a bad measurement is one
    /// tap away from being undone.
    private var previousCenter: (Float, Float)?
    @Published private(set) var hasCenterUndo = false

    /// Angle, in camera coordinates, of the ray that lands on a source pixel.
    ///
    /// Undoes the radial distortion and then the projection, which is the
    /// inverse of what the shader does.
    func cameraAngle(forSourcePixel pixel: SIMD2<Float>,
                     sourceSize: CGSize) -> Float? {
        let parameters = parameters(sourceSize: sourceSize)
        let offset = pixel - parameters.center
        let radius = simd_length(offset)
        guard radius > 1, radius.isFinite else { return nil }

        let normalized = radius / max(parameters.maxRadius, 1)
        let factor = 1 + parameters.k1 * normalized * normalized
            + parameters.k2 * normalized * normalized * normalized * normalized
        let modelRadius = radius / max(factor, 0.25)

        let theta: Float
        if parameters.projection < 0.5 {
            theta = modelRadius / max(parameters.focal, 1)
        } else {
            let ratio = min(max(modelRadius / (2 * max(parameters.focal, 1)), -1), 1)
            theta = 2 * asin(ratio)
        }
        guard theta.isFinite, theta < 1.5 else { return nil }
        return theta
    }

    /// Direction, in camera coordinates (+X right, +Y down, +Z out of the back
    /// camera), of the ray that lands on a given source pixel.
    ///
    /// This is the inverse of the shader's mapping, so a point measured in the
    /// raw frame can be turned into the direction the view has to look along to
    /// put it in the middle.
    func cameraDirection(forSourcePixel pixel: SIMD2<Float>,
                         sourceSize: CGSize) -> SIMD3<Float>? {
        guard let theta = cameraAngle(forSourcePixel: pixel, sourceSize: sourceSize) else {
            return nil
        }
        let parameters = parameters(sourceSize: sourceSize)
        let offset = pixel - parameters.center
        let radius = simd_length(offset)
        guard radius > 1 else { return nil }
        let direction = offset / radius
        return SIMD3<Float>(sin(theta) * direction.x,
                            sin(theta) * direction.y,
                            cos(theta))
    }

    /// Zooms the view so a screen of the measured angular radius fills the
    /// requested fraction of the frame width.
    ///
    /// The whole point is repeatability: align once, lock the framing, and every
    /// later recording has the cabinet the same size in the same place.
    func applyFramingLock(screenAngle: Float) {
        let target = min(max(cabinetFillTarget, 0.3), 0.95)
        let tangent = tan(screenAngle)
        guard tangent > 0.001, tangent.isFinite else { return }
        let halfFovRadians = atan(tangent / target)
        let degrees = halfFovRadians * 2 * 180 / .pi
        outputFov = min(max(degrees, 30), 130)
    }

    /// Applies a measured circle centre. Only these two values are touched, so
    /// a measurement can never disturb the distortion terms that were tuned by
    /// hand. The previous centre is kept so the change can be undone.
    func applyMeasuredCenter(centerX newX: Float, centerY newY: Float) {
        previousCenter = (centerX, centerY)
        hasCenterUndo = true
        centerX = min(max(newX, -0.25), 0.25)
        centerY = min(max(newY, -0.25), 0.25)
    }

    func undoMeasuredCenter() {
        guard hasCenterUndo, let previous = previousCenter else { return }
        centerX = previous.0
        centerY = previous.1
        hasCenterUndo = false
        previousCenter = nil
    }

    func resetLens() {
        projection = .equidistant
        lensHalfFov = 90
        circleScale = 1.0
        outputFov = 78
        k1 = 0
        k2 = 0
        centerX = 0
        centerY = 0
        edgeFeather = 0.025
        fillScreen = true
        showGrid = false
        cabinetFillTarget = 0.8
        sharpness = 0.18
        localContrast = 0.10
        hazeCompensation = 0.05
    }

    func parameters(sourceSize: CGSize) -> FisheyeParameters {
        let width = max(Float(sourceSize.width), 1)
        let height = max(Float(sourceSize.height), 1)
        let shortSide = min(width, height)
        let radius = max(shortSide * 0.5 * circleScale, 1)
        let theta = max(lensHalfFov, 1) * .pi / 180
        let focal: Float
        switch projection {
        case .equidistant:
            focal = radius / theta
        case .equisolid:
            focal = radius / max(2 * sin(theta * 0.5), 0.001)
        }

        return FisheyeParameters(
            center: SIMD2<Float>(width * 0.5 + centerX * shortSide,
                                height * 0.5 + centerY * shortSide),
            focal: focal,
            maxRadius: radius,
            maxTheta: theta,
            outputFov: max(outputFov, 10) * .pi / 180,
            projection: Float(projection.rawValue),
            k1: k1,
            k2: k2,
            edgeFeather: max(edgeFeather, 0),
            sharpness: min(max(sharpness, 0), 1),
            localContrast: min(max(localContrast, 0), 1),
            hazeCompensation: min(max(hazeCompensation, 0), 1)
        )
    }
}
