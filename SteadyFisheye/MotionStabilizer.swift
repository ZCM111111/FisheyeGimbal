import Foundation
import Combine
import CoreMotion
import simd

struct MotionSnapshot {
    var cameraFromLocked: simd_float3x3 = matrix_identity_float3x3
    /// Relative rotation in Core Motion's device coordinate frame.
    var relativeDeviceQuaternion = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    var lockVersion: UInt64 = 0
    var timestamp: TimeInterval = 0
    var valid = false
}

private struct MotionSample {
    var timestamp: TimeInterval
    var quaternion: simd_quatf
    var lockVersion: UInt64
}

final class MotionStabilizer: ObservableObject {
    enum Mode: String, CaseIterable, Identifiable {
        case hold
        case follow

        var id: String { rawValue }
        var title: String {
            switch self {
            case .hold: return "Hold"
            case .follow: return "Follow"
            }
        }
    }

    @Published private(set) var available = false
    @Published private(set) var locked = false
    @Published private(set) var mode: Mode = .hold
    @Published private(set) var status = "Waiting for motion"

    private let manager = CMMotionManager()
    private let motionQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "steadyfisheye.motion"
        queue.qualityOfService = .userInteractive
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private let lock = NSLock()

    private var latest = MotionSnapshot()
    private var samples: [MotionSample] = []
    private let maxSampleCount = 36
    private var filtered = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    private var lockedQuaternion = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    private var hasSample = false
    private var lastTimestamp: TimeInterval = 0
    private var modeValue: Mode = .hold
    private var smoothingValue: Double = 0.055
    private var dampingValue: Double = 1.0
    private var lockVersion: UInt64 = 0
    private var renderQuaternion = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    private var renderVersion: UInt64 = 0
    private var lastRenderedSampleTime: TimeInterval = 0
    private var hasRenderedSample = false
    private var displaySmoothingValue: Double = 0.035
    private var activeStatusPublished = false

    // Camera coordinates are +X right, +Y down, +Z out through the back camera.
    // Core Motion uses +X right, +Y up, +Z toward the screen/user.
    private let cameraToDevice = simd_float3x3(columns: (
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(0, -1, 0),
        SIMD3<Float>(0, 0, -1)
    ))

    var smoothing: Double {
        get {
            lock.lock(); defer { lock.unlock() }
            return smoothingValue
        }
        set {
            lock.lock()
            smoothingValue = min(max(newValue, 0), 0.8)
            lock.unlock()
        }
    }

    var displaySmoothing: Double {
        get {
            lock.lock(); defer { lock.unlock() }
            return displaySmoothingValue
        }
        set {
            lock.lock()
            displaySmoothingValue = min(max(newValue, 0), 0.25)
            lock.unlock()
        }
    }

    var dampingTime: Double {
        get {
            lock.lock(); defer { lock.unlock() }
            return dampingValue
        }
        set {
            lock.lock()
            dampingValue = min(max(newValue, 0.15), 4)
            lock.unlock()
        }
    }

    func snapshot() -> MotionSnapshot {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    /// Gets the pose that belongs to one camera frame. Motion arrives at a
    /// different rate from video, so selecting the nearest/interpolated IMU
    /// sample prevents alternating old/new poses from becoming visible jitter.
    func renderSnapshot(forFrameAt frameTime: TimeInterval) -> MotionSnapshot {
        lock.lock()
        let current = latest
        guard current.valid else {
            lock.unlock()
            return current
        }

        let timeIsUsable = frameTime.isFinite && frameTime > 0
            && abs(frameTime - current.timestamp) < 1.0
        let requestedTime = timeIsUsable ? frameTime : current.timestamp
        let target = timeIsUsable ? sampleLocked(at: requestedTime) : MotionSample(
            timestamp: current.timestamp,
            quaternion: current.relativeDeviceQuaternion,
            lockVersion: current.lockVersion
        )

        if !hasRenderedSample || target.lockVersion != renderVersion {
            renderQuaternion = target.quaternion
            renderVersion = target.lockVersion
            hasRenderedSample = true
        } else if target.timestamp > lastRenderedSampleTime + 0.0005 {
            let dt = min(max(target.timestamp - lastRenderedSampleTime, 1.0 / 120.0), 0.1)
            let alpha = displaySmoothingValue <= 0
                ? 1
                : 1 - exp(-dt / displaySmoothingValue)
            renderQuaternion = slerpShortest(
                renderQuaternion,
                target.quaternion,
                amount: Float(min(max(alpha, 0), 1))
            )
        }
        lastRenderedSampleTime = max(lastRenderedSampleTime, target.timestamp)
        let outputQuaternion = renderQuaternion
        lock.unlock()

        var result = current
        result.timestamp = target.timestamp
        result.relativeDeviceQuaternion = outputQuaternion
        result.cameraFromLocked = cameraToDevice
            * simd_float3x3(outputQuaternion)
            * cameraToDevice
        return result
    }

    func start() {
        guard manager.isDeviceMotionAvailable else {
            publishAvailability(false, status: "Device motion unavailable")
            return
        }

        manager.deviceMotionUpdateInterval = 1.0 / 120.0
        publishAvailability(true, status: "Starting motion")
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical,
                                         to: motionQueue) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                self.publishAvailability(false, status: error.localizedDescription)
                return
            }
            guard let motion else { return }
            self.consume(motion)
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        lock.lock()
        latest = MotionSnapshot()
        samples.removeAll(keepingCapacity: true)
        lockVersion = 0
        renderQuaternion = identityQuaternion()
        renderVersion = 0
        hasRenderedSample = false
        lastRenderedSampleTime = 0
        activeStatusPublished = false
        hasSample = false
        lastTimestamp = 0
        lock.unlock()
        publishAvailability(false, status: "Motion stopped")
    }

    func recenter() {
        lock.lock()
        if hasSample {
            lockedQuaternion = filtered
            lockVersion &+= 1
            let identity = identityQuaternion()
            latest.relativeDeviceQuaternion = identity
            latest.cameraFromLocked = matrix_identity_float3x3
            latest.lockVersion = lockVersion
            latest.valid = true
            samples.removeAll(keepingCapacity: true)
            samples.append(MotionSample(timestamp: lastTimestamp,
                                        quaternion: identity,
                                        lockVersion: lockVersion))
            renderQuaternion = identity
            renderVersion = lockVersion
            lastRenderedSampleTime = lastTimestamp
            hasRenderedSample = true
        }
        lock.unlock()
    }

    func setMode(_ newMode: Mode) {
        lock.lock()
        modeValue = newMode
        if hasSample {
            lockedQuaternion = filtered
            lockVersion &+= 1
            let identity = identityQuaternion()
            latest.relativeDeviceQuaternion = identity
            latest.cameraFromLocked = matrix_identity_float3x3
            latest.lockVersion = lockVersion
            samples.removeAll(keepingCapacity: true)
            samples.append(MotionSample(timestamp: lastTimestamp,
                                        quaternion: identity,
                                        lockVersion: lockVersion))
            renderQuaternion = identity
            renderVersion = lockVersion
            lastRenderedSampleTime = lastTimestamp
            hasRenderedSample = true
        }
        lock.unlock()
        DispatchQueue.main.async { [weak self] in
            self?.mode = newMode
        }
    }

    private func consume(_ deviceMotion: CMDeviceMotion) {
        let attitude = deviceMotion.attitude.quaternion
        let current = simd_quatf(ix: Float(attitude.x),
                                  iy: Float(attitude.y),
                                  iz: Float(attitude.z),
                                  r: Float(attitude.w))
        let timestamp = deviceMotion.timestamp

        lock.lock()
        if !hasSample {
            hasSample = true
            filtered = current
            lockedQuaternion = current
            lastTimestamp = timestamp
        }

        let dt = min(max(timestamp - lastTimestamp, 1.0 / 240.0), 0.25)
        lastTimestamp = timestamp
        let alpha = smoothingValue <= 0 ? 1 : 1 - exp(-dt / smoothingValue)
        filtered = slerpShortest(filtered, current,
                                 amount: Float(min(max(alpha, 0), 1)))

        if modeValue == .follow {
            let beta = 1 - exp(-dt / max(dampingValue, 0.15))
            lockedQuaternion = slerpShortest(lockedQuaternion, filtered,
                                             amount: Float(min(max(beta, 0), 1)))
        }

        // qCurrent^-1 * qLocked maps a ray in the locked device frame into
        // the current device frame. The basis conversion wraps camera pixels.
        let relativeDeviceQuaternion = filtered.inverse * lockedQuaternion
        let relativeDevice = simd_float3x3(relativeDeviceQuaternion)
        let relativeCamera = cameraToDevice * relativeDevice * cameraToDevice
        latest.cameraFromLocked = relativeCamera
        latest.relativeDeviceQuaternion = relativeDeviceQuaternion
        latest.lockVersion = lockVersion
        latest.timestamp = timestamp
        latest.valid = true
        samples.append(MotionSample(timestamp: timestamp,
                                    quaternion: relativeDeviceQuaternion,
                                    lockVersion: lockVersion))
        if samples.count > maxSampleCount {
            samples.removeFirst(samples.count - maxSampleCount)
        }
        lock.unlock()

        lock.lock()
        let shouldPublishActive = !activeStatusPublished
        activeStatusPublished = true
        lock.unlock()
        if shouldPublishActive {
            DispatchQueue.main.async { [weak self] in
                self?.locked = true
                self?.status = "Motion active"
            }
        }
    }

    private func sampleLocked(at time: TimeInterval) -> MotionSample {
        guard let first = samples.first, let last = samples.last else {
            return MotionSample(timestamp: latest.timestamp,
                                quaternion: latest.relativeDeviceQuaternion,
                                lockVersion: latest.lockVersion)
        }
        if time <= first.timestamp { return first }
        if time >= last.timestamp { return last }

        for index in 1..<samples.count {
            let upper = samples[index]
            guard time <= upper.timestamp else { continue }
            let lower = samples[index - 1]
            guard upper.lockVersion == lower.lockVersion else {
                return time < upper.timestamp ? lower : upper
            }
            let span = max(upper.timestamp - lower.timestamp, 0.000001)
            let amount = Float(min(max((time - lower.timestamp) / span, 0), 1))
            return MotionSample(
                timestamp: time,
                quaternion: slerpShortest(lower.quaternion, upper.quaternion,
                                           amount: amount),
                lockVersion: upper.lockVersion
            )
        }
        return last
    }

    private func publishAvailability(_ value: Bool, status: String) {
        DispatchQueue.main.async { [weak self] in
            self?.available = value
            self?.status = status
        }
    }

    private func identityQuaternion() -> simd_quatf {
        simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
    }

    private func slerpShortest(_ a: simd_quatf,
                               _ b: simd_quatf,
                               amount: Float) -> simd_quatf {
        let av = SIMD4<Float>(a.imag.x, a.imag.y, a.imag.z, a.real)
        var bv = SIMD4<Float>(b.imag.x, b.imag.y, b.imag.z, b.real)
        var dot = simd_dot(av, bv)
        if dot < 0 {
            bv = -bv
            dot = -dot
        }

        let t = min(max(amount, 0), 1)
        let result: SIMD4<Float>
        if dot > 0.9995 {
            result = simd_normalize(av + (bv - av) * t)
        } else {
            let theta = acos(min(max(dot, -1), 1))
            let sinTheta = max(sin(theta), 0.00001)
            let wa = sin((1 - t) * theta) / sinTheta
            let wb = sin(t * theta) / sinTheta
            result = simd_normalize(av * wa + bv * wb)
        }
        return simd_quatf(ix: result.x, iy: result.y, iz: result.z, r: result.w)
    }
}
