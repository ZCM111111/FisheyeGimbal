"""Runs the app's cabinet detector over real footage, in Python.

This mirrors CabinetDetector.swift exactly: same decimation (256 wide), same
Otsu threshold, same 4-connected largest blob, same moment-based ellipse, same
quality gates. Run it on real photos to see whether the classical approach
actually holds up, instead of assuming it does.
"""

import glob
import os
import sys

import cv2
import numpy as np

TARGET_WIDTH = 256           # matches LensCircleMeasurer.makeGrid
MIN_AREA = 0.02
MAX_AREA = 0.60
MIN_FILL = 0.72
MIN_AXIS_RATIO = 0.45

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_out")


def otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    lo, hi = int(gray.min()), int(gray.max())
    if hi - lo <= 8:
        return 0
    idx = np.arange(256)
    sum_all = float((idx * hist).sum())
    sum_bg = 0.0
    w_bg = 0.0
    best_var, best_t = 0.0, lo
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        m_bg = sum_bg / w_bg
        m_fg = (sum_all - sum_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def analyse(path):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw is None:
        return None
    h, w = raw.shape[:2]
    step = max(1, w // TARGET_WIDTH)
    small = raw[::step, ::step]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape

    t = otsu(gray)
    mask = (gray >= t).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if count <= 1:
        return dict(path=path, ok=False, why="没有亮区", size=(w, h), grid=(gw, gh), threshold=t)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    area = int(stats[best_label, cv2.CC_STAT_AREA])
    area_frac = area / float(gw * gh)

    ys, xs = np.nonzero(labels == best_label)
    cx, cy = float(xs.mean()), float(ys.mean())
    dx = xs - cx
    dy = ys - cy
    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())
    trace = mu20 + mu02
    delta = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    l1 = max((trace + delta) * 0.5, 1e-4)
    l2 = max((trace - delta) * 0.5, 1e-4)
    a = 2 * np.sqrt(l1)
    b = 2 * np.sqrt(l2)
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    fill = area / (np.pi * a * b) if a * b > 0 else 0

    short = float(min(gw, gh))
    res = dict(ok=True, size=(w, h), grid=(gw, gh), threshold=t,
               area_frac=area_frac, fill=fill, axis_ratio=b / max(a, 1e-4),
               cx=(cx - gw * 0.5) / short, cy=(cy - gh * 0.5) / short,
               rx=a / short, ry=b / short, angle=float(np.degrees(angle)))

    if area_frac <= MIN_AREA:
        res["ok"] = False
        res["why"] = "亮区太小"
    elif area_frac >= MAX_AREA:
        res["ok"] = False
        res["why"] = "亮区几乎占满画面"
    elif fill <= MIN_FILL:
        res["ok"] = False
        res["why"] = "形状不规整"
    elif b / max(a, 1e-4) <= MIN_AXIS_RATIO:
        res["ok"] = False
        res["why"] = "太扁"

    # Visualisation: mask + fitted ellipse overlay.
    os.makedirs(OUT, exist_ok=True)
    vis = small.copy()
    vis[mask.astype(bool)] = (vis[mask.astype(bool)] * 0.45 + np.array([0, 90, 0])).astype(np.uint8)
    if res["ok"]:
        colour = (0, 215, 255)
    else:
        colour = (0, 0, 255)
    cv2.ellipse(vis, (int(cx), int(cy)), (int(a), int(b)), res["angle"] if "angle" in res else 0,
                0, 360, colour, 1)
    cv2.drawMarker(vis, (int(cx), int(cy)), colour, cv2.MARKER_CROSS, 12, 1)
    name = os.path.basename(path).replace(".PNG", ".jpg")
    cv2.imwrite(os.path.join(OUT, name), vis)
    return res


def main():
    files = sorted(sum([glob.glob(os.path.join(d, "*")) for d in DATA_DIRS], []))
    print(f"{'file':<14}{'ok':<5}{'area':>7}{'fill':>7}{'a/b':>7}{'cx':>8}{'cy':>8}{'rx':>7}   reason")
    print("-" * 82)
    hits = 0
    for path in files:
        r = analyse(path)
        if r is None:
            continue
        name = os.path.basename(path)
        if not r.get("ok"):
            print(f"{name:<14}{'NO':<5}{'':>7}{'':>7}{'':>7}{'':>8}{'':>8}{'':>7}   {r.get('why', '')}")
            continue
        hits += 1
        print(f"{name:<14}{'yes':<5}{r['area_frac']:>7.3f}{r['fill']:>7.2f}{r['axis_ratio']:>7.2f}"
              f"{r['cx']:>8.3f}{r['cy']:>8.3f}{r['rx']:>7.3f}   thr={r['threshold']}")
    print("-" * 82)
    print(f"通过检测: {hits}/{len(files)}")
    print("可视化输出:", OUT)


if __name__ == "__main__":
    main()
