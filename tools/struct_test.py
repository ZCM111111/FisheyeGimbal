"""Colour-free cabinet detection, tested on both datasets.

The buttons are RGB LEDs and change colour with the song, so any hue test is
luck: it passed on the frames at hand only because those songs happened to be
violet. The stable features are structural — the circular screen bezel, the ring
of button shapes, the decorative frame around them — so this works from edges.

Two independent circle finders are compared, and both are scored the same way:
how much real edge lies along the circle they propose. A circle with weak edge
support is a guess, not a detection.
"""

import glob
import os

import cv2
import numpy as np

HOME = r"C:\Users\93543\Downloads\IMG_77*.PNG"
ARCADE_DIRS = [r"D:\桌面文件\训练2", r"D:\桌面文件\训练数据"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "struct_out")


def imread_unicode(path):
    try:
        with open(path, "rb") as handle:
            return cv2.imdecode(np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def prepare(path, width=320):
    img = imread_unicode(path)
    if img is None:
        return None
    step = max(1, img.shape[1] // width)
    small = img[::step, ::step]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return small, gray


def gradients(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def edge_support(grad, cx, cy, r, samples=180):
    """Fraction of the circle where a real edge is found, plus the mean strength."""
    h, w = grad.shape
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    hits = 0
    total = 0.0
    valid = 0
    for a in angles:
        best = 0.0
        found = False
        # Allow a small radial search: the ring is not perfectly circular under
        # a wide lens, and the edge may be a pixel or two off.
        for dr in range(-3, 4):
            x = int(round(cx + (r + dr) * np.cos(a)))
            y = int(round(cy + (r + dr) * np.sin(a)))
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                found = True
                best = max(best, float(grad[y, x]))
        if found:
            valid += 1
            total += best
    if valid == 0:
        return 0.0, 0.0
    mean = total / valid
    threshold = 0.55 * float(np.percentile(grad, 97))
    strong = 0
    for a in angles:
        best = 0.0
        for dr in range(-3, 4):
            x = int(round(cx + (r + dr) * np.cos(a)))
            y = int(round(cy + (r + dr) * np.sin(a)))
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                best = max(best, float(grad[y, x]))
        if best >= threshold:
            strong += 1
    return strong / valid, mean


def hough_candidates(gray, limit=6):
    h, w = gray.shape
    rmin = int(min(h, w) * 0.12)
    rmax = int(min(h, w) * 0.75)
    out = []
    for p2 in (40, 34, 28):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(h, w) * 0.25,
                                   param1=140, param2=p2, minRadius=rmin, maxRadius=rmax)
        if circles is not None:
            for c in circles[0]:
                out.append((float(c[0]), float(c[1]), float(c[2])))
        if len(out) >= limit:
            break
    return out[:limit]


def radial_candidates(grad, gray, limit=6):
    """Fit a circle to the strongest radial edge, scanning outwards from the
    frame centre — the shape of the screen bezel, whatever the screen shows."""
    h, w = grad.shape
    cx, cy = w * 0.5, h * 0.5
    cuts = np.percentile(grad, 88)
    points = []
    angles = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    for a in angles:
        best_r, best_v = 0.0, 0.0
        rmin = min(h, w) * 0.10
        rmax = min(h, w) * 0.78
        r = rmin
        while r < rmax:
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))
            if 0 <= x < w and 0 <= y < h:
                v = float(grad[y, x])
                if v > best_v:
                    best_v, best_r = v, r
            r += 1.0
        if best_v >= cuts:
            points.append((cx + best_r * np.cos(a), cy + best_r * np.sin(a)))
    if len(points) < 12:
        return []
    p = np.array(points, float)
    found = []
    for _ in range(3):
        x, y = p[:, 0], p[:, 1]
        A = np.stack([2 * x, 2 * y, np.ones_like(x)], 1)
        bv = x * x + y * y
        (fx, fy, c), *_ = np.linalg.lstsq(A, bv, rcond=None)
        r = np.sqrt(max(c + fx * fx + fy * fy, 1e-6))
        d = np.hypot(x - fx, y - fy)
        keep = np.abs(d - d.mean()) <= max(0.25 * d.mean(), 3)
        if keep.all() or keep.sum() < 12:
            break
        p = p[keep]
    found.append((fx, fy, r))
    return found


def analyse(path):
    prepared = prepare(path)
    if prepared is None:
        return None
    small, gray = prepared
    grad = gradients(gray)
    h, w = gray.shape
    short = float(min(h, w))

    candidates = []
    for (cx, cy, r) in hough_candidates(gray):
        s, m = edge_support(grad, cx, cy, r)
        candidates.append(dict(method="hough", cx=cx, cy=cy, r=r, support=s, strength=m))
    for (cx, cy, r) in radial_candidates(grad, gray):
        s, m = edge_support(grad, cx, cy, r)
        candidates.append(dict(method="radial", cx=cx, cy=cy, r=r, support=s, strength=m))

    if not candidates:
        return dict(ok=False, why="无线索", small=small)
    # A detection needs both a real edge along the circle and a plausible size.
    for c in candidates:
        c["score"] = c["support"] * min(c["r"] / (short * 0.35), 1.0)
    candidates.sort(key=lambda c: -c["score"])
    best = candidates[0]
    best["ok"] = best["support"] >= 0.55
    best["short"] = short
    best["small"] = small
    best["cxNorm"] = (best["cx"] - w * 0.5) / short
    best["cyNorm"] = (best["cy"] - h * 0.5) / short
    best["rNorm"] = best["r"] / short
    best["candidates"] = len(candidates)
    if not best["ok"]:
        best["why"] = f"边缘支持度只有 {best['support']:.2f}"
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    arcade = []
    for d in ARCADE_DIRS:
        arcade += sorted(glob.glob(os.path.join(d, "*.png")))
    for name, files in (("你家(暗房)", sorted(glob.glob(HOME))), ("街机厅", arcade)):
        ok = 0
        supports = []
        by_method = {}
        for f in files:
            r = analyse(f)
            if r is None:
                continue
            base = os.path.basename(f)
            if r.get("ok"):
                ok += 1
                supports.append(r["support"])
                by_method[r["method"]] = by_method.get(r["method"], 0) + 1
            if len(files) <= 14:
                if r.get("ok"):
                    print(f"  {base:<24}{r['method']:<8}{r['candidates']}候选 "
                          f"偏移({r['cxNorm']:+.3f},{r['cyNorm']:+.3f}) r={r['rNorm']:.3f} "
                          f"支持度{r['support']:.2f}")
                else:
                    print(f"  {base:<24}失败 {r.get('why','')}")
            vis = r["small"].copy()
            colour = (0, 215, 255) if r.get("ok") else (0, 0, 255)
            cv2.circle(vis, (int(r["cx"]), int(r["cy"])), int(r["r"]), colour, 1)
            cv2.drawMarker(vis, (int(r["cx"]), int(r["cy"])), colour, cv2.MARKER_CROSS, 12, 1)
            cv2.imwrite(os.path.join(OUT, name[:2] + "_" + base), vis)
        avg = float(np.mean(supports)) if supports else 0.0
        print(f"  → 通过 {ok}/{len(files)}  平均边缘支持度 {avg:.2f}  方法分布 {by_method}")
        print()

    print("可视化:", OUT)


if __name__ == "__main__":
    main()
