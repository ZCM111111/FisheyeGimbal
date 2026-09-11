"""Runs the shipped cabinet detector over the arcade first-person set.

The thresholds in the app were tuned on 12 frames from a dim room, where the
buttons read as clearly violet. Arcade frames light the buttons much harder and
closer to white, so this measures whether those thresholds still hold, and
compares them against the "bright also counts" variant that failed on the
earlier set.
"""

import glob
import os

import cv2
import numpy as np

DIRS = [r"D:\桌面文件\训练2", r"D:\桌面文件\训练数据"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcade_out")

MIN_BLOBS = 5
MIN_AREA = 4
MAX_AREA = 1400
MIN_COVERAGE = 200


def imread_unicode(path):
    """cv2.imread cannot open paths with non-ASCII characters on Windows, so the
    bytes are read directly and decoded."""
    try:
        with open(path, "rb") as handle:
            data = np.frombuffer(handle.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def detect(img, chroma_t, luma_t, bright_or, spread_limit=0.28, width=256):
    step = max(1, img.shape[1] // width)
    small = img[::step, ::step]
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycc[..., 1].astype(np.int16) - 128
    cb = ycc[..., 2].astype(np.int16) - 128
    y = ycc[..., 0].astype(np.int16)
    score = np.minimum(cr, cb)
    mask = ((score > chroma_t) & (y > luma_t))
    if bright_or:
        mask |= (y > 235)
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    pts = [(float(cen[i][0]), float(cen[i][1])) for i in range(1, n)
           if MIN_AREA <= st[i, cv2.CC_STAT_AREA] <= MAX_AREA]
    info = dict(blobs=len(pts))
    if len(pts) < MIN_BLOBS:
        info["why"] = f"只找到 {len(pts)} 个色块"
        return info, small, mask

    p = np.array(pts, float)
    for _ in range(3):
        x, yy = p[:, 0], p[:, 1]
        A = np.stack([2 * x, 2 * yy, np.ones_like(x)], 1)
        bv = x * x + yy * yy
        (cx, cy, c), *_ = np.linalg.lstsq(A, bv, rcond=None)
        r = np.sqrt(max(c + cx * cx + cy * cy, 1e-6))
        d = np.hypot(x - cx, yy - cy)
        if len(p) <= MIN_BLOBS:
            break
        keep = np.abs(d - d.mean()) <= max(0.28 * d.mean(), 3)
        if keep.all():
            break
        p = p[keep]

    x, yy = p[:, 0], p[:, 1]
    d = np.hypot(x - cx, yy - cy)
    spread = float(d.std() / max(d.mean(), 1e-6))
    ang = np.sort(np.degrees(np.arctan2(yy - cy, x - cx)))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360]]))
    cover = 360 - float(gaps.max())
    short = float(min(small.shape[:2]))
    info.update(n=len(p), cx=(cx - small.shape[1] * 0.5) / short,
                cy=(cy - small.shape[0] * 0.5) / short, r=r / short,
                spread=spread, cover=cover, center=(cx, cy, r))
    if spread > spread_limit:
        info["why"] = f"离散度 {spread:.2f}"
    elif cover < MIN_COVERAGE:
        info["why"] = f"覆盖 {cover:.0f}°"
    else:
        info["ok"] = True
    return info, small, mask


def main():
    os.makedirs(OUT, exist_ok=True)
    files = []
    for d in DIRS:
        files += sorted(glob.glob(os.path.join(d, "*.png")))
    print(f"素材: {len(files)} 张  ({os.path.basename(DIRS[0])} + {os.path.basename(DIRS[1])})")
    print()

    variants = [
        ("当前发布阈值  chroma>4  Y>90", 4, 90, False),
        ("放宽        chroma>4  Y>70", 4, 70, False),
        ("更宽        chroma>2  Y>70", 2, 70, False),
        ("加高亮      chroma>4  Y>90 |Y>235", 4, 90, True),
        ("加高亮+宽   chroma>2  Y>70 |Y>235", 2, 70, True),
    ]
    print(f"{'方案':<36}{'通过':>8}")
    print("-" * 46)
    results = {}
    for label, ct, yt, bo in variants:
        ok = 0
        for f in files:
            img = imread_unicode(f)
            if img is None:
                continue
            info, _, _ = detect(img, ct, yt, bo)
            if info.get("ok"):
                ok += 1
        results[label] = ok
        print(f"{label:<36}{ok:>4}/{len(files)}")

    # Detail + visualisation for the shipped variant.
    print()
    print("-" * 78)
    print(f"{'file':<26}{'blobs':>6}{'cx':>8}{'cy':>8}{'r':>7}{'spread':>8}{'cover':>7}  结果")
    print("-" * 78)
    for f in files:
        img = imread_unicode(f)
        if img is None:
            continue
        info, small, mask = detect(img, 4, 90, False)
        name = os.path.basename(f)
        if info.get("ok"):
            print(f"{name:<26}{info['n']:>6}{info['cx']:>8.3f}{info['cy']:>8.3f}{info['r']:>7.3f}"
                  f"{info['spread']:>8.3f}{info['cover']:>7.0f}  OK")
        else:
            print(f"{name:<26}{info['blobs']:>6}{'-':>8}{'-':>8}{'-':>7}{'-':>8}{'-':>7}  {info.get('why','')}")
        vis = small.copy()
        vis[mask.astype(bool)] = (80, 200, 80)
        if info.get("ok"):
            cx, cy, r = info["center"]
            cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 0, 255), 1)
            cv2.drawMarker(vis, (int(cx), int(cy)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.imwrite(os.path.join(OUT, name), vis)

    print("-" * 78)
    print("可视化:", OUT)


if __name__ == "__main__":
    main()

