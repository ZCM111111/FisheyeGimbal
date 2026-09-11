# SteadyFisheye

A clean iOS prototype for a clip-on fisheye lens:

- GPU inverse mapping corrects fisheye geometry in one render pass.
- Core Motion locks the camera to the direction captured when the app starts or when Recenter is pressed.
- Video frames are matched to a timestamped IMU history and then lightly smoothed at display time to reduce 120 Hz motion / 60 Hz rendering jitter.
- Hold mode keeps the world direction fixed. Follow mode lets the lock drift toward a deliberate slow turn.
- The preview uses the rear camera and disables Apple's video stabilization and geometric distortion correction so the lens can be calibrated from raw pixels.

## Build

Open `SteadyFisheye.xcodeproj` on macOS with Xcode 15 or newer. Select an iPhone target, set a signing team, and run on a real device. The simulator cannot provide a useful camera/IMU result.

Codemagic is configured in `codemagic.yaml`. Connect the GitHub repository, select the `ios-unsigned` workflow, and start a build from `main`. The `SteadyFisheye-unsigned.ipa` artifact is deliberately not signed; install it on Windows with Sideloadly or AltStore using your Apple ID.

The project is intentionally portrait-only. Roll, pitch, and yaw are compensated in camera coordinates while the interface remains portrait.

## Calibration order

1. Mount the lens without changing its position.
2. Point at a straight door frame or table edge.
3. Open the settings sheet from the slider button. Adjust `Lens half FOV` and `Circle scale` until the image circle is fully covered without excessive black edges.
4. Select `Equidistant` or `Equisolid`, then adjust `Radial k1` and `Radial k2` until straight lines are straight.
5. Adjust `Center X` and `Center Y` if the lens is off-center. `Output FOV` controls how much of the corrected image is shown.
6. Use `Sensor smoothing` for physical motion filtering and `Display smoothing` for small residual jitter. Start low and increase only as needed.
7. Press `Recenter`, then move the phone and check the locked direction.

A pure IMU has gyro bias. It can hold a direction for a useful period but will slowly drift; pressing Recenter is expected. Large rotations can expose black edges because the requested ray leaves the fisheye image circle. This is a field-of-view limit, not a rendering bug.
