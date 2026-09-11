"""Colour-free cabinet detection, shared by the labelling tool and the tests.

The buttons are RGB LEDs and change colour with the song, so hue is useless
here. What is stable is structure: the screen's circular bezel. This finds it
from gradients and then checks how much real edge lies along the circle it
proposes — a circle with weak support is a guess, not a detection.

Mirrors CabinetDetector.swift, so pre-labels made here mean the same thing the
app would have decided on its own.
"""

import cv2
import numpy as np


def imread_unicode(path):
    """cv2.imread cannot open non-ASCII paths on Windows."""
    try:
        with open(path, "rb") as handle:
            return cv2.imdecode(np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def edge_maps(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy)


def edge_support(magnitude, threshold, cx, cy, r, samples=180, tolerance=3):
    h, w = magnitude.shape
    strong = 0
    valid = 0
    for a in np.linspace(0, 2 * np.pi, samples, endpoint=False):
        dx, dy = np.cos(a), np.sin(a)
        best = 0.0
        found = False
        for dr in range(-tolerance, tolerance + 1):
            x = int(round(cx + (r + dr) * dx))
            y = int(round(cy + (r + dr) * dy))
            if 0 <= x < w and 0 <= y < h:
                found = True
                best = max(best, float(magnitude[y, x]))
        if found:
            valid += 1
            if best >= threshold:
                strong += 1
    return strong / valid if valid else 0.0


def detect(img, support_limit=0.55):
    """Returns dict(cx, cy, r, support, ok, why) in pixels of the given image.

    Circles come from OpenCV's Hough transform rather than a hand-rolled one:
    this runs per frame in an interactive tool, and the Python version of the
    accumulator loop is orders of magnitude slower for the same proposals.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, _, magnitude = edge_maps(gray)
    h, w = gray.shape
    short = float(min(h, w))

    threshold = 0.55 * float(np.percentile(magnitude, 97))
    if threshold <= 4:
        return dict(ok=False, why="画面太平，没有可用边缘", support=0.0)

    rmin = max(int(short * 0.12), 8)
    rmax = max(int(short * 0.78), rmin + 4)
    proposals = []
    for param2 in (40, 34, 28):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5,
                                   minDist=short * 0.25, param1=140,
                                   param2=param2, minRadius=rmin, maxRadius=rmax)
        if circles is None:
            continue
        for circle in circles[0]:
            cx, cy, r = (float(circle[0]), float(circle[1]), float(circle[2]))
            proposals.append((cx, cy, r))
        if len(proposals) >= 6:
            break

    if not proposals:
        return dict(ok=False, why="没有候选圆", support=0.0)

    best = dict(ok=False, why="候选都不可用", support=0.0)
    for cx, cy, r in proposals[:6]:
        score = edge_support(magnitude, threshold, cx, cy, r)
        # Same shape-aware weighting as the app: a real edge along the circle,
        # and a radius that is plausible for a cabinet in frame.
        weighted = score * min(r / (short * 0.35), 1.0)
        if weighted > best.get("weighted", -1):
            best = dict(ok=True, cx=cx, cy=cy, r=r, support=score,
                        weighted=weighted, threshold=threshold, why="")

    if best.get("ok") and best["support"] < support_limit:
        best["ok"] = False
        best["why"] = f"边缘支持度只有 {best['support']:.2f}"
    return best
