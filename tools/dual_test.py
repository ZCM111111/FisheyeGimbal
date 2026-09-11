import os
import sys

# Frame folders are passed on the command line so this file carries no
# personal paths: python dual_test.py <folder> [folder ...]
DATA_DIRS = sys.argv[1:]
if not DATA_DIRS:
    print("usage: python dual_test.py <frames folder> [more folders]")
    sys.exit(1)

"""Verifies a dual-mask cabinet detector on both real datasets at once.

The two datasets disagreed: the dim-room frames need a strict violet mask, the
arcade frames need brightness to count too because the lit buttons are blown out
towards white. Trying both masks and picking the better result by shape quality
satisfies both, and this is where that is proven before it goes into the app.
"""

import glob
import os

import cv2
import numpy as np

HOME_GLOB = DATA_DIRS[0]
ARCADE_DIRS = DATA_DIRS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dual_out")

MIN_BLOBS = 5
MIN_AREA = 4
MAX_AREA = 1400
MIN_COVERAGE = 200
BRIGHT = 235


def imread_unicode(path):
    try:
        with open(path, "rb") as handle:
            return cv2.imdecode(np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def blobs_from(mask, width):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    return [(float(cen[i][0]), float(cen[i][1])) for i in range(1, n)
            if MIN_AREA <= st[i, cv2.CC_STAT_AREA] <= MAX_AREA]


def fit_ring(pts):
    p = np.array(pts, float)
    cx = cy = r = None
    for _ in range(3):
        x, y = p[:, 0], p[:, 1]
        A = np.stack([2 * x, 2 * y, np.ones_like(x)], 1)
        bv = x * x + y * y
        (cx, cy, c), *_ = np.linalg.lstsq(A, bv, rcond=None)
        r = np.sqrt(max(c + cx * cx + cy * cy, 1e-6))
        d = np.hypot(x - cx, y - cy)
        if len(p) <= MIN_BLOBS:
            break
        keep = np.abs(d - d.mean()) <= max(0.28 * d.mean(), 3)
        if keep.all():
            break
        p = p[keep]
    x, y = p[:, 0], p[:, 1]
    d = np.hypot(x - cx, y - cy)
    if len(d) == 0 or d.mean() <= 0:
        return None
    spread = float(d.std() / d.mean())
    ang = np.sort(np.degrees(np.arctan2(y - cy, x - cx)))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360]]))
    cover = 360 - float(gaps.max())
    return dict(cx=cx, cy=cy, r=r, n=len(p), spread=spread, cover=cover)


def detect(path, width=256):
    img = imread_unicode(path)
    if img is None:
        return None
    step = max(1, img.shape[1] // width)
    small = img[::step, ::step]
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycc[..., 1].astype(np.int16) - 128
    cb = ycc[..., 2].astype(np.int16) - 128
    y = ycc[..., 0].astype(np.int16)
    violet = (np.minimum(cr, cb) > 4) & (y > 90)

    candidates = []
    strict = fit_ring(blobs_from(violet, width)) if violet.sum() else None
    if strict and strict["n"] >= MIN_BLOBS:
        candidates.append(("violet", strict))
    wide = fit_ring(blobs_from(violet | (y > BRIGHT), width))
    if wide and wide["n"] >= MIN_BLOBS:
        candidates.append(("violet+bright", wide))

    good = [c for c in candidates if c[1]["spread"] <= 0.28 and c[1]["cover"] >= MIN_COVERAGE]
    if not good:
        best = min(candidates, key=lambda c: c[1]["spread"]) if candidates else None
        return dict(ok=False, why=(best[0] + f" spread={best[1]['spread']:.2f}" if best else "无线索"),
                    small=small)
    # Prefer the tighter ring; break near-ties towards the larger one, since the
    # cabinet is the biggest ring in frame.
    good.sort(key=lambda c: (c[1]["spread"], -c[1]["r"]))
    label, res = good[0]
    short = float(min(small.shape[:2]))
    return dict(ok=True, label=label, small=small,
                cx=(res["cx"] - small.shape[1] * 0.5) / short,
                cy=(res["cy"] - small.shape[0] * 0.5) / short,
                r=res["r"] / short, n=res["n"], spread=res["spread"],
                cover=res["cover"], raw=(res["cx"], res["cy"], res["r"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    sets = {"你家(暗房)": sorted(glob.glob(HOME_GLOB))}
    arcade = []
    for d in ARCADE_DIRS:
        arcade += sorted(glob.glob(os.path.join(d, "*.png")))
    sets["街机厅"] = arcade

    for name, files in sets.items():
        ok = 0
        labels = {}
        print(f"=== {name}: {len(files)} 张 ===")
        for f in files:
            r = detect(f)
            if r is None:
                continue
            base = os.path.basename(f)
            if r["ok"]:
                ok += 1
                labels[r["label"]] = labels.get(r["label"], 0) + 1
                if len(files) <= 14:
                    print(f"  {base:<24} {r['label']:<15}{r['n']:>4}个  "
                          f"偏移({r['cx']:+.3f},{r['cy']:+.3f}) r={r['r']:.3f} "
                          f"离散{r['spread']:.3f} 覆盖{r['cover']:.0f}°")
                vis = r["small"].copy()
                cx, cy, rad = r["raw"]
                cv2.circle(vis, (int(cx), int(cy)), int(rad), (255, 0, 255), 1)
                cv2.drawMarker(vis, (int(cx), int(cy)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
                cv2.imwrite(os.path.join(OUT, name[:4] + "_" + base), vis)
            else:
                if len(files) <= 14:
                    print(f"  {base:<24} 失败: {r['why']}")
        print(f"  → 通过 {ok}/{len(files)}   来源分布 {labels}")
        print()

    print("可视化:", OUT)


if __name__ == "__main__":
    main()
