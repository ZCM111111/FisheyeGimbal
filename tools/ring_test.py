import tempfile
import os
import sys

# Frame folders are passed on the command line so this file carries no
# personal paths: python ring_test.py <folder> [folder ...]
DATA_DIRS = sys.argv[1:]
if not DATA_DIRS:
    print("usage: python ring_test.py <frames folder> [more folders]")
    sys.exit(1)

"""Verifies the lit-button-ring detector across the whole set.

This is the approach that survived testing: the maimai buttons are the most
distinctive structure in the frame (lit, saturated, arranged in a ring), and
they are present whether the cabinet is idle or mid-song. The gate is a circle
fit through the blob centres, which is what rejects stray saturated content
elsewhere in the scene.

Same thresholds as will be ported to CabinetDetector.swift.
"""

import glob
import os

import cv2
import numpy as np

OUT = os.path.join(tempfile.gettempdir(), "fisheye_detector_out")
FILES = sorted(sum([glob.glob(os.path.join(d, "*")) for d in DATA_DIRS], []))

MIN_BLOBS = 5
MIN_BLOB_AREA = 4
MAX_BLOB_AREA = 1400
MAX_RADIUS_SPREAD = 0.22      # std(radius) / mean(radius)
MIN_ANGULAR_COVERAGE = 200    # degrees of the ring that must be occupied


def analyse(path, width=256):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = raw.shape[:2]
    step = max(1, w // width)
    small = raw[::step, ::step]
    gh, gw = small.shape[:2]

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Violet / purple, which is what the lit buttons read as.
    mask = (((hue >= 122) & (hue <= 172)) & (sat > 60) & (val > 70)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            blobs.append((float(centroids[i][0]), float(centroids[i][1]), int(area)))

    result = dict(path=path, ok=False, blobs=len(blobs), why="")
    if len(blobs) < MIN_BLOBS:
        result["why"] = f"只找到 {len(blobs)} 个按键色块"
        return result, small, mask, blobs

    pts = np.array([[b[0], b[1]] for b in blobs], dtype=np.float64)

    # Circle fit with iterative outlier rejection: scene clutter that happens to
    # be violet shows up as a blob off the ring, and one stray point is enough
    # to spoil a plain least-squares fit.
    for _ in range(3):
        x, y = pts[:, 0], pts[:, 1]
        A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
        bvec = x * x + y * y
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        cx, cy, c = sol
        r = np.sqrt(max(c + cx * cx + cy * cy, 1e-6))
        d = np.hypot(x - cx, y - cy)
        if len(pts) <= MIN_BLOBS:
            break
        keep = np.abs(d - d.mean()) <= max(0.28 * d.mean(), 3)
        if keep.all():
            break
        pts = pts[keep]
    blobs = [(p[0], p[1], 0) for p in pts]

    x, y = pts[:, 0], pts[:, 1]
    d = np.hypot(x - cx, y - cy)
    spread = float(d.std() / max(d.mean(), 1e-6))
    angles = np.sort(np.degrees(np.arctan2(y - cy, x - cx)))
    if len(angles) > 1:
        gaps = np.diff(np.concatenate([angles, [angles[0] + 360]]))
        coverage = 360 - float(gaps.max())
    else:
        coverage = 0.0

    short = float(min(gw, gh))
    result.update(cx=(cx - gw * 0.5) / short, cy=(cy - gh * 0.5) / short,
                  radius=r / short, spread=spread, coverage=coverage)

    if spread > MAX_RADIUS_SPREAD:
        result["why"] = f"半径离散度 {spread:.2f} 太大，不像环"
        return result, small, mask, blobs
    if coverage < MIN_ANGULAR_COVERAGE:
        result["why"] = f"只覆盖 {coverage:.0f}° 的圆环"
        return result, small, mask, blobs

    result["ok"] = True
    result["why"] = "通过"
    return result, small, mask, blobs


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'file':<14}{'blobs':>6}{'cx':>9}{'cy':>9}{'r':>7}{'spread':>8}{'cover':>7}  result")
    print("-" * 84)
    passed = 0
    for path in FILES:
        res, small, mask, blobs = analyse(path)
        name = os.path.basename(path)
        if res["ok"]:
            passed += 1
            print(f"{name:<14}{res['blobs']:>6}{res['cx']:>9.3f}{res['cy']:>9.3f}{res['radius']:>7.3f}"
                  f"{res['spread']:>8.3f}{res['coverage']:>7.0f}  {res['why']}")
        else:
            print(f"{name:<14}{res['blobs']:>6}{'-':>9}{'-':>9}{'-':>7}{'-':>8}{'-':>7}  {res['why']}")

        vis = small.copy()
        vis[mask.astype(bool)] = (80, 200, 80)
        for (bx, by, _) in blobs:
            cv2.circle(vis, (int(bx), int(by)), 2, (255, 255, 0), -1)
        if res["ok"]:
            cx = res["cx"] * min(small.shape[:2]) + small.shape[1] * 0.5
            cy = res["cy"] * min(small.shape[:2]) + small.shape[0] * 0.5
            r = res["radius"] * min(small.shape[:2])
            cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 0, 255), 1)
            cv2.drawMarker(vis, (int(cx), int(cy)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.imwrite(os.path.join(OUT, name.replace(".PNG", ".jpg")), vis)

    print("-" * 84)
    print(f"通过: {passed}/{len(FILES)}")
    print("out:", OUT)


if __name__ == "__main__":
    main()
