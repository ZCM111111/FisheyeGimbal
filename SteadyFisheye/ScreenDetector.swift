import CoreGraphics
import CoreML
import CoreImage
import Foundation
import ImageIO
import Vision

/// The trained screen detector, run through Vision.
///
/// The model is optional by design. If it is missing from the bundle, or Vision
/// refuses it, every caller falls back to the classical circle detector, so a
/// missing model costs accuracy rather than the build.
enum ScreenDetector {

    struct Detection {
        /// Centre in capture-buffer pixels — the same space `FrameGrid.sourceSize`
        /// and `CabinetDetector` report in.
        var center: CGPoint
        var radius: CGFloat
        var confidence: Float

        var summary: String { String(format: "模型 %.2f", Double(confidence)) }
    }

    private static let modelName = "ScreenDetector"

    /// Below this the box is not worth acting on: a wrong alignment is worse
    /// than no alignment.
    ///
    /// Low on purpose. This model scores its correct boxes 0.13–0.41 — checked
    /// against ultralytics' own ONNX predictor, which lands IoU 0.82–0.95 on
    /// those same frames. A 0.5-style threshold here would report nothing found
    /// while a *saturated* wrong input would look confident, which is exactly
    /// the failure the ONNX check script exists to catch.
    private static let scoreThreshold: Float = 0.08

    /// The target is a circle, so a box far from square means the model locked
    /// onto something else and the measurement cannot be trusted.
    private static let aspectLimits: ClosedRange<CGFloat> = 0.55...1.8

    private enum LoadState {
        case untried
        case ready(VNCoreMLRequest)
        case unavailable
    }

    private static var state: LoadState = .untried
    private static let stateLock = NSLock()
    private static let context = CIContext()

    /// True when a usable model is in the bundle. Loads it on first use.
    static var isAvailable: Bool { request() != nil }

    private static func request() -> VNCoreMLRequest? {
        stateLock.lock()
        defer { stateLock.unlock() }
        switch state {
        case .ready(let request):
            return request
        case .unavailable:
            return nil
        case .untried:
            // Xcode compiles the .mlpackage in the resources phase, so the
            // bundle holds the compiled .mlmodelc under the model's name.
            guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc"),
                  let model = try? MLModel(contentsOf: url),
                  let visionModel = try? VNCoreMLModel(for: model) else {
                state = .unavailable
                return nil
            }
            let request = VNCoreMLRequest(model: visionModel)
            // The model was trained on square, centre-cropped frames.
            request.imageCropAndScaleOption = .centerCrop
            state = .ready(request)
            return request
        }
    }

    /// Runs the model on one frame. Pure work over a local image, so it is safe
    /// to call off the main thread.
    static func detect(in image: CGImage) -> Detection? {
        guard let request = request() else { return nil }

        // The capture connection already delivers portrait buffers, which is the
        // orientation the model was trained on. A landscape buffer would reach
        // the model at 90°, so turn it upright here and undo that turn on the
        // answer. The classic `sourceSize.height >= sourceSize.width` test is
        // what the renderer uses for the same distinction.
        let landscape = image.height < image.width
        let source = landscape ? upright(image) : image

        let handler = VNImageRequestHandler(cgImage: source, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        guard let array = firstMultiArray(in: request.results),
              let inputSide = inputSide(of: request),
              let best = bestBox(in: array, inputSide: CGFloat(inputSide)) else {
            return nil
        }

        let width = CGFloat(source.width)
        let height = CGFloat(source.height)
        let side = min(width, height)
        let originX = (width - side) * 0.5
        let originY = (height - side) * 0.5
        let scale = side / CGFloat(inputSide)

        var center = CGPoint(x: originX + best.box.midX * scale,
                             y: originY + best.box.midY * scale)
        var radius = (best.box.width + best.box.height) * 0.25 * scale

        if landscape {
            center = uprightInverse(center, uprightSize: CGSize(width: width, height: height))
        }

        guard radius > 4 else { return nil }
        return Detection(center: center, radius: radius, confidence: best.score)
    }

    // MARK: - Vision plumbing

    private static func firstMultiArray(in results: [VNObservation]?) -> MLMultiArray? {
        for observation in results ?? [] {
            if let feature = observation as? VNCoreMLFeatureValueObservation,
               let array = feature.featureValue.multiArrayValue {
                return array
            }
        }
        return nil
    }

    private static func inputSide(of request: VNCoreMLRequest) -> Int? {
        for description in request.model.modelDescription.inputDescriptionsByName.values {
            if let constraint = description.imageConstraint, constraint.pixelsWide > 0 {
                return constraint.pixelsWide
            }
        }
        return nil
    }

    // MARK: - Decoding

    /// YOLOv8 exports a single tensor holding cx, cy, w, h in input pixels
    /// followed by one score per class — five values here, since there is one
    /// class. The layout is read from the shape rather than assumed: getting
    /// channels-first and anchors-first backwards produces a plausible-looking
    /// box in the wrong place, which is exactly the failure that is hard to
    /// notice.
    private static func bestBox(in array: MLMultiArray,
                                inputSide: CGFloat) -> (box: CGRect, score: Float)? {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 3, shape[0] == 1 else { return nil }

        let attributes = 5
        let first = shape[1]
        let second = shape[2]
        let channelsFirst: Bool
        if first == attributes && second != attributes {
            channelsFirst = true
        } else if second == attributes && first != attributes {
            channelsFirst = false
        } else {
            return nil
        }
        let anchors = channelsFirst ? second : first

        // Subscript access instead of a typed pointer: it costs milliseconds on
        // a one-shot detection and works for every element type CoreML emits.
        func value(_ attribute: Int, _ anchor: Int) -> Float {
            let indices = channelsFirst
                ? [NSNumber(value: 0), NSNumber(value: attribute), NSNumber(value: anchor)]
                : [NSNumber(value: 0), NSNumber(value: anchor), NSNumber(value: attribute)]
            return array[indices].floatValue
        }

        var best: (box: CGRect, score: Float)?
        var bestScore = scoreThreshold
        for anchor in 0..<anchors {
            let score = value(4, anchor)
            guard score > bestScore else { continue }
            let width = CGFloat(value(2, anchor))
            let height = CGFloat(value(3, anchor))
            guard width > 1, height > 1 else { continue }
            guard aspectLimits.contains(width / height) else { continue }
            let box = CGRect(x: CGFloat(value(0, anchor)) - width * 0.5,
                             y: CGFloat(value(1, anchor)) - height * 0.5,
                             width: width, height: height)
            guard box.minX > -inputSide, box.minY > -inputSide else { continue }
            best = (box, score)
            bestScore = score
        }
        return best
    }

    // MARK: - Landscape buffers

    /// Turns the buffer 90° clockwise, so a landscape capture looks like the
    /// portrait frames the model was trained on.
    private static func upright(_ image: CGImage) -> CGImage {
        let oriented = CIImage(cgImage: image).oriented(.right)
        return context.createCGImage(oriented, from: oriented.extent) ?? image
    }

    /// Undo of `upright`. Turning a (W, H) buffer 90° clockwise sends its pixel
    /// (x, y) to (H - 1 - y, x) in the upright image, so reading back is
    /// x = uprightY, y = H - 1 - uprightX — and H is the upright image's width.
    ///
    /// Dead code on a phone whose capture connection already delivers portrait
    /// buffers, which is why it is only a safe guess rather than a verified one.
    private static func uprightInverse(_ point: CGPoint,
                                       uprightSize: CGSize) -> CGPoint {
        CGPoint(x: point.y, y: uprightSize.width - point.x)
    }
}
