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

    // Image finishing controls. These are deliberately conservative because
    // sharpening cannot recover detail lost to a soft sensor or dirty glass.
    @Published var sharpness: Float = 0.18
    @Published var localContrast: Float = 0.10
    @Published var hazeCompensation: Float = 0.05

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
