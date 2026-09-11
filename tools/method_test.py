import tempfile
import os
import sys

# Frame folders are passed on the command line so this file carries no
# personal paths: python method_test.py <folder> [folder ...]
DATA_DIRS = sys.argv[1:]
if not DATA_DIRS:
    print("usage: python method_test.py <frames folder> [more folders]")
    sys.exit(1)

"""Tries several cabinet-finding strategies on real footage and compares them.

The bright-blob approach already failed here (a wall won in one frame, the white
shell won in another), so this tests what actually survives across the whole
set:

  1. Hough circles on the greyscale - the maimai cabinet is built around strong
     circular edges (screen bezel, button ring).
  2. The lit button ring, segmented by hue/saturation, then a circle fitted
     through the blob centres.
  3. The dark screen disc (inverted threshold), which only exists while the
     cabinet is idle.

A method is only worth shipping if it lands in the same place in every frame.
"""

import glob
import os

import cv2
import numpy as np

OUT = os.path.join(tempfile.gettempdir(), "fisheye_detector_out")
FILES = sorted(sum([glob.glob(os.path.join(d, "*")) for d in DATA_DIRS], []))


def load(path, width=256):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = raw.shape[:2]
    step = max(1, w // width)
    small = raw[::step, ::step]
    return small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def hough(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=60,
                               param1=120, param2=32, minRadius=28, maxRadius=150)
    if circles is None:
        return []
    return [tuple(map(float, c)) for c in circles[0][:3]]


def button_ring(small):
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Purple/violet buttons: OpenCV hue is 0-179.
    mask = (((h >= 125) & (h <= 170)) & (s > 70) & (v > 60)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if 6 <= area <= 1200:
            blobs.append((centroids[i][0], centroids[i][1], area))
    if len(blobs) < 4:
        return None, blobs, mask
    xs = np.array([b[0] for b in blobs])
    ys = np.array([b[1] for b in blobs])
    cx, cy = float(xs.mean()), float(ys.mean())
    r = float(np.hypot(xs - cx, ys - cy).mean())
    return (cx, cy, r), blobs, mask


def dark_disc(gray):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (gray < t).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4)
    if count <= 1:
        return None
    area = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(area))
    ys, xs = np.nonzero(labels == best)
    if len(xs) < 50:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    l1 = (dx * dx).mean() + (dy * dy).mean()
    a = 2 * np.sqrt(max(l1 / 2, 1e-4))
    return (cx, cy, a, len(xs) / float(gray.size))


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'file':<14}{'hough(cx,cy,r)':<26}{'buttons n/cx/cy/r':<28}{'dark cx/cy/r/frac':<26}")
    print("-" * 92)
    for path in FILES:
        small, gray = load(path)
        gh, gw = gray.shape
        name = os.path.basename(path)

        hs = hough(gray)
        hstr = f"{hs[0][0]:.0f},{hs[0][1]:.0f},{hs[0][2]:.0f}" if hs else "-"

        br, blobs, bmask = button_ring(small)
        if br:
            bstr = f"{len(blobs)}/{br[0]:.0f},{br[1]:.0f},{br[2]:.0f}"
        else:
            bstr = f"{len(blobs)}/-"

        dd = dark_disc(gray)
        dstr = f"{dd[0]:.0f},{dd[1]:.0f},{dd[2]:.0f},{dd[3]:.2f}" if dd else "-"

        print(f"{name:<14}{hstr:<26}{bstr:<28}{dstr:<26}")

        vis = small.copy()
        if hs:
            for (cx, cy, r) in hs[:1]:
                cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 215, 255), 1)
                cv2.drawMarker(vis, (int(cx), int(cy)), (0, 215, 255), cv2.MARKER_CROSS, 10, 1)
        if br:
            cv2.circle(vis, (int(br[0]), int(br[1])), int(br[2]), (255, 0, 255), 1)
            cv2.drawMarker(vis, (int(br[0]), int(br[1])), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
        for (bx, by, _) in blobs:
            cv2.circle(vis, (int(bx), int(by)), 2, (255, 255, 0), -1)
        if dd:
            cv2.circle(vis, (int(dd[0]), int(dd[1])), int(dd[2]), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(OUT, name.replace(".PNG", ".jpg")), vis)

    print("-" * 92)
    print("yellow = hough, magenta = lit button ring, cyan dots = button blobs, green = dark disc")
    print("out:", OUT)


if __name__ == "__main__":
    main()
