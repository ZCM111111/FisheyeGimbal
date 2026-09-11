import Foundation
import SwiftUI

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
    var travel: Float
    var projection: Float
    var k1: Float
    var k2: Float
    var edgeFeather: Float
    var sharpness: Float
    var localContrast: Float
    var hazeCompensation: Float
}

final class FisheyeSettings: ObservableObject {
    @Published var projection: FisheyeProjection = .equidistant
    /// Half of the angular coverage the glass delivers. This does not create
    /// black edges any more (fill mode pins samples to the rim), so its job is
    /// purely geometric: how many degrees of real lens sit inside the image
    /// circle. Too small stretches the picture, too large squeezes it.
    @Published var lensHalfFov: Float = 90
    @Published var circleScale: Float = 1.0
    @Published var outputFov: Float = 78
    /// How far the lock may swing the view away from where the phone points
    /// before it starts following, in degrees.
    ///
    /// This has to be an explicit budget rather than something derived from the
    /// lens: a frame that fills the whole image circle leaves no room to pan
    /// before it runs off the glass, so the view size is computed *after*
    /// reserving this much travel.
    @Published var lockTravelDegrees: Float = 12
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

    // Image finishing controls. These are deliberately conservative because
    // sharpening cannot recover detail lost to a soft sensor or dirty glass.
    @Published var sharpness: Float = 0.18
    @Published var localContrast: Float = 0.10
    @Published var hazeCompensation: Float = 0.05

    /// Writes an automatic calibration back into the live parameters.
    ///
    /// Circle geometry is always taken, because it is measured rather than
    /// fitted. The distortion terms are only replaced when the fit genuinely
    /// straightened the picture, so a failed calibration cannot make things
    /// worse than the user's current settings.
    func apply(_ outcome: AutoCalibrator.Outcome) {
        if outcome.circleFound {
            centerX = min(max(outcome.centerX, -0.25), 0.25)
            centerY = min(max(outcome.centerY, -0.25), 0.25)
            circleScale = min(max(outcome.circleScale, 0.4), 1.6)
        }
        guard outcome.applied else { return }
        k1 = min(max(outcome.k1, -0.35), 0.35)
        lensHalfFov = min(max(outcome.lensHalfFovDegrees, 45), 120)
    }

    func resetLens() {
        projection = .equidistant
        lensHalfFov = 90
        circleScale = 1.0
        outputFov = 78
        lockTravelDegrees = 12
        k1 = 0
        k2 = 0
        centerX = 0
        centerY = 0
        edgeFeather = 0.025
        fillScreen = true
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
            travel: min(max(lockTravelDegrees, 3), 30) * .pi / 180,
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
