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
        /// Roll-only stabilisation referenced to gravity, the way a 360
        /// camera's horizon lock behaves: the horizon stays level while yaw
        /// and pitch follow the phone freely.
        case horizon

        var id: String { rawValue }
        var title: String {
            switch self {
            case .hold: return "锁定"
            case .follow: return "跟随"
            case .horizon: return "地平线"
            }
        }
    }

    @Published private(set) var available = false
    @Published private(set) var locked = false
    @Published private(set) var mode: Mode = .hold
    @Published private(set) var status = "等待陀螺仪"
    /// How far the phone is rolled away from level, in degrees. Derived from
    /// gravity continuously, never latched from a button, so it is always the
    /// live reference rather than a stale snapshot.
    @Published private(set) var horizonTilt: Float = 0

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
    /// No post-hoc smoothing by default: any smoothing applied to the
    /// correction signal is a residual error, and leaving small fast shake
    /// uncorrected is exactly the bug this default removes.
    private var displaySmoothingValue: Double = 0
    private var gravityFiltered = SIMD3<Float>(0, -1, 0)
    private var hasGravity = false
    private var horizonQuaternion = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1))
    private var horizonTiltValue: Float = 0
    private var lastHorizonPublish: TimeInterval = 0
    private var activeStatusPublished = false

    // Camera coordinates are +X right, +Y down, +Z out through the back camera.
    // Core Motion uses +X right, +Y up, +Z toward the screen/user.
    private let cameraToDevice = simd_float3x3(columns: (
        SIMD3<Float>(1, 0, 0),
        SIMD3<Float>(0, -1, 0),
        SIMD3<Float>(0, 0, -1)
    ))

    /// Restores the mode the user last chose, so a launch does not silently
    /// drop back to the default stabilisation.
    init() {
        if let raw = SettingsStore.loadStabilizerMode(), let saved = Mode(rawValue: raw) {
            modeValue = saved
            mode = saved
        }
    }

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

    /// The attitude to latch when the lock is established or re-centred: the
    /// same pointing direction the phone has right now, but with the roll taken
    /// from gravity instead of from the hand.
    ///
    /// Latching the raw attitude bakes in whatever tilt the phone happens to
    /// have at that instant, so pressing re-centre (or simply starting the app)
    /// while holding the phone crooked leaves the picture permanently crooked.
    /// Re-centring should only choose *where* you are looking, never how level
    /// the horizon is.
    ///
    /// Must be called with `lock` held.
    private func levelLockedAttitude(from attitude: simd_quatf) -> simd_quatf {
        levelLockedAttitude(from: attitude, cameraDirection: nil)
    }

    private func levelLockedAttitude(from attitude: simd_quatf,
                                     cameraDirection: SIMD3<Float>?) -> simd_quatf {
        let gravityWorld = attitude.act(gravityFiltered)
        let gravityLength = simd_length(gravityWorld)
        guard gravityLength > 0.05, hasGravity else { return attitude }
        let vertical = gravityWorld / gravityLength

        let cameraToWorld = simd_float3x3(attitude) * cameraToDevice
        // Camera coordinates are +X right, +Y down, +Z along the optical axis,
        // and `cameraToWorld` already carries the camera-to-device flip — so the
        // aim vector must go through it exactly once, the same way the plain
        // optical axis does.
        //
        // It used to be multiplied by `cameraToDevice` an extra time, which
        // cancels the flip and reads a camera-frame ray as a device-frame one:
        // straight ahead (0, 0, 1) then meant "at the user", so the lock ended up
        // aimed roughly backwards, the renderer sampled past the lens rim and
        // pinned every pixel to it, and the frame smeared into radial streaks
        // the moment anything was detected. The axis-only path never had the
        // error, which is why the preview looked normal until then.
        var forward = cameraToWorld * (cameraDirection ?? SIMD3<Float>(0, 0, 1))
        let forwardLength = simd_length(forward)
        guard forwardLength > 0.05 else { return attitude }
        forward /= forwardLength

        // The horizon is level exactly when the camera's right axis has no
        // vertical component, and the basis also has to stay orthogonal to the
        // optical axis. A cross product satisfies both at once; projecting out
        // the vertical component alone would leave a skewed, non-rotational
        // basis, and converting that to a quaternion is meaningless.
        let cameraRight = cameraToWorld * SIMD3<Float>(1, 0, 0)
        var right = simd_cross(forward, vertical)
        let rightLength = simd_length(right)
        // Pointing straight down or up leaves roll undefined; keep the attitude.
        guard rightLength > 0.15 else { return attitude }
        right /= rightLength
        // Keep the picture the right way round instead of mirrored.
        if simd_dot(right, cameraRight) < 0 { right = -right }

        // Camera basis is right / down / forward, and X cross Y equals Z.
        let down = simd_cross(forward, right)
        let lockedCameraToWorld = simd_float3x3(columns: (right, down, forward))
        let lockedDeviceToWorld = lockedCameraToWorld * cameraToDevice
        return simd_quatf(lockedDeviceToWorld)
    }

    /// Roll-only correction taken straight from gravity, the way a 360 camera
    /// holds its horizon: yaw and pitch stay free, the horizon stays level.
    /// Nothing is latched here, which is what makes it work even when the
    /// phone is already tilted before the mode is selected.
    ///
    /// Must be called with `lock` held.
    private func horizonCorrection() -> simd_quatf {
        // Gravity in camera coordinates: +X right, +Y down, +Z out the back.
        let g = cameraToDevice * gravityFiltered
        let planar = (g.x * g.x + g.y * g.y).squareRoot()
        // Roll is undefined when the optical axis points at the ground or the
        // sky, so hold the previous correction rather than snapping wildly.
        guard planar > 0.2 else { return horizonQuaternion }

        // Pick the angle that rotates gravity exactly onto the image "down"
        // axis, which is what puts the horizon level.
        let theta = atan2(-g.x, g.y)
        horizonTiltValue = theta * 180 / Float.pi
        // Rolling about the camera's optical axis is a rotation about -Z in
        // the Core Motion device frame.
        horizonQuaternion = simd_quatf(angle: -theta, axis: SIMD3<Float>(0, 0, 1))
        return horizonQuaternion
    }

    func start() {
        guard manager.isDeviceMotionAvailable else {
            publishAvailability(false, status: "设备不支持陀螺仪数据")
            return
        }

        manager.deviceMotionUpdateInterval = 1.0 / 120.0
        publishAvailability(true, status: "正在启动陀螺仪")
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
        gravityFiltered = SIMD3<Float>(0, -1, 0)
        hasGravity = false
        horizonQuaternion = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 0, 1))
        horizonTiltValue = 0
        lastHorizonPublish = 0
        activeStatusPublished = false
        hasSample = false
        lastTimestamp = 0
        lock.unlock()
        publishAvailability(false, status: "陀螺仪已停止")
    }

    func recenter() {
        lock.lock()
        // Horizon mode is referenced to gravity every sample, so latching the
        // current attitude would bake in whatever tilt the phone happens to
        // have and leave the image permanently crooked. Ignore it there.
        guard modeValue != .horizon else {
            lock.unlock()
            return
        }
        if hasSample {
            // Level-latched: re-centring chooses the pointing direction only.
            lockedQuaternion = levelLockedAttitude(from: filtered)
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

    /// Re-aims the lock so a direction seen off-axis becomes the centre of the
    /// frame, with the horizon kept level.
    ///
    /// This is the live equivalent of dragging a viewport back to the middle in
    /// post production: one rotation assignment instead of keyframes.
    func reLock(lookingAlong cameraDirection: SIMD3<Float>) {
        lock.lock()
        defer { lock.unlock() }
        guard hasSample, simd_length(cameraDirection) > 0.01 else { return }
        lockedQuaternion = levelLockedAttitude(from: filtered,
                                               cameraDirection: cameraDirection)
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

    /// Where the lock is pointing right now, expressed in the current camera
    /// frame.
    ///
    /// Snapping the lock onto a new direction every time makes the picture jump,
    /// so the automatic search glides instead: each pass nudges the lock a
    /// fraction of the way toward its target, which needs to know how far apart
    /// the two currently are.
    func lockedCameraDirection() -> SIMD3<Float>? {
        lock.lock()
        defer { lock.unlock() }
        guard hasSample else { return nil }
        // The lock is a device-to-world attitude, so its forward axis is the
        // camera axis written in device coordinates.
        let worldForward = simd_float3x3(lockedQuaternion)
            * (cameraToDevice * SIMD3<Float>(0, 0, 1))
        let relative = simd_float3x3(filtered).inverse * worldForward
        return cameraToDevice * relative
    }

    func setMode(_ newMode: Mode) {        lock.lock()
        modeValue = newMode
        SettingsStore.saveStabilizerMode(newMode.rawValue)
        if hasSample {
            // Level-latched: re-centring chooses the pointing direction only.
            lockedQuaternion = levelLockedAttitude(from: filtered)
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
        let gravity = SIMD3<Float>(Float(deviceMotion.gravity.x),
                                   Float(deviceMotion.gravity.y),
                                   Float(deviceMotion.gravity.z))

        lock.lock()
        if !hasSample {
            hasSample = true
            filtered = current
            // Seed gravity before latching, so the very first lock is level too.
            gravityFiltered = gravity
            hasGravity = true
            lockedQuaternion = levelLockedAttitude(from: current)
            lastTimestamp = timestamp
        }

        let dt = min(max(timestamp - lastTimestamp, 1.0 / 240.0), 0.25)
        lastTimestamp = timestamp
        let alpha = smoothingValue <= 0 ? 1 : 1 - exp(-dt / smoothingValue)
        filtered = slerpShortest(filtered, current,
                                 amount: Float(min(max(alpha, 0), 1)))

        // Gravity is a reference direction rather than a rate, so gently
        // smoothing it removes accelerometer noise without the horizon ever
        // lagging behind the phone.
        if !hasGravity {
            gravityFiltered = gravity
            hasGravity = true
        }
        let gravityAlpha = Float(1 - exp(-dt / 0.06))
        gravityFiltered += (gravity - gravityFiltered) * gravityAlpha
        let gravityLength = simd_length(gravityFiltered)
        if gravityLength > 0.001 {
            gravityFiltered /= gravityLength
        }

        if modeValue == .follow {
            let beta = 1 - exp(-dt / max(dampingValue, 0.15))
            lockedQuaternion = slerpShortest(lockedQuaternion, filtered,
                                             amount: Float(min(max(beta, 0), 1)))
        }

        // The correction has to be built from the RAW attitude. Using the
        // smoothed attitude as the base cancels only the slow component of the
        // motion: with R = q_filtered^-1 * q_locked the direction seen at an
        // output pixel works out to e(t) * constant, where e(t) is exactly the
        // high-frequency part the low pass removed. That left small, fast hand
        // shake completely uncorrected while slow pans looked stabilised.
        // Raw attitude gives R = q_true^-1 * q_locked and a world-locked image.
        let relativeDeviceQuaternion: simd_quatf
        switch modeValue {
        case .hold, .follow:
            // qCurrent^-1 * qLocked maps a ray in the locked device frame into
            // the current device frame. Built from the raw attitude so every
            // movement, small or large, is corrected in full.
            relativeDeviceQuaternion = current.inverse * lockedQuaternion
        case .horizon:
            relativeDeviceQuaternion = horizonCorrection()
        }
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
        let tiltToPublish: Float? = modeValue == .horizon ? horizonTiltValue : nil
        let tiltIsDue = timestamp - lastHorizonPublish > 0.08
        if tiltIsDue {
            lastHorizonPublish = timestamp
        }
        lock.unlock()

        if let tiltToPublish = tiltToPublish, tiltIsDue {
            DispatchQueue.main.async { [weak self] in
                self?.horizonTilt = tiltToPublish
            }
        }

        lock.lock()
        let shouldPublishActive = !activeStatusPublished
        activeStatusPublished = true
        lock.unlock()
        if shouldPublishActive {
            DispatchQueue.main.async { [weak self] in
                self?.locked = true
                self?.status = "陀螺仪工作中"
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
