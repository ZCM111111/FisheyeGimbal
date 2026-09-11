"""Measure how well a clip holds its subject in the frame.

This is for comparing a reference clip against the app's own output instead of
arguing about adjectives. It runs the classical circle detector over every
frame, then reports the numbers that describe a locked shot:

    centre spread   how much the subject wanders, in pixels and as a
                    percentage of the frame's short side
    drift rate      how fast that wander accumulates, in %/s
    radius spread   whether the framing is holding its size
    centring        how far the subject sits from the middle, on average

A shot that is genuinely pinned has a centre spread near zero; one where the
picture is dragged back to the middle between corrections has a spread that
grows with how fast the camera moves.

Usage:
    python tools/lock_report.py <video or folder of frames> [...] [--montage out.jpg]
"""

import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cabinet_detect  # noqa: E402

MAX_FRAMES = 240
DETECT_WIDTH = 480          # detection cost is per frame; accuracy is unaffected


def frames_from_video(path):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print("cannot open", path)
        return []
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // MAX_FRAMES) if total > 0 else 1
    frames = []
    index = 0
    while True:
        ok = capture.read()
        if not ok[0]:
            break
        if index % step == 0:
            frames.append(ok[1])
        index += 1
        if total > 0 and index >= total:
            break
    capture.release()
    return frames


def frames_from_folder(path):
    files = sorted(f for f in os.listdir(path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    frames = []
    for name in files[:MAX_FRAMES]:
        data = np.fromfile(os.path.join(path, name), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    return frames


def analyse(frames, label, montage_path=None, consecutive=True):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    short = float(min(width, height))
    results = []

    for index, frame in enumerate(frames):
        scale = DETECT_WIDTH / max(width, height)
        small = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame
        found = cabinet_detect.detect(small)
        if found.get("ok"):
            results.append((index,
                            found["cx"] / scale,
                            found["cy"] / scale,
                            found["r"] / scale,
                            found["support"]))

    if len(results) < 3:
        print(f"{label}: only {len(results)}/{len(frames)} frames detected — nothing to measure")
        return

    xs = np.array([r[1] for r in results])
    ys = np.array([r[2] for r in results])
    rs = np.array([r[3] for r in results])
    indices = np.array([r[0] for r in results])

    # Wander of the subject across the clip.
    spread = float(np.hypot(xs - xs.mean(), ys - ys.mean()).mean())

    # How far the subject sits from the middle.
    offset = np.hypot(xs - width / 2, ys - height / 2).mean()

    print()
    print(f"== {label}")
    print(f"   frames detected   {len(results)}/{len(frames)}"
          f"   ({100 * len(results) / len(frames):.0f}%)")
    print(f"   centre spread     {spread:.1f} px  ({100 * spread / short:.2f}% of the short side)")
    if consecutive:
        # Only meaningful between consecutive frames: a folder of unrelated
        # stills would report the movement between shots, not drift within one.
        steps = np.hypot(np.diff(xs), np.diff(ys))
        duration = max(indices[-1] - indices[0] + 1, 1) / 30.0
        rate = float(steps.sum() / duration)
        print(f"   drift rate        {rate:.1f} px/s  ({100 * rate / short:.2f}%/s)")
    print(f"   distance from mid {offset:.1f} px  ({100 * offset / short:.2f}%)")
    print(f"   radius spread     {rs.std():.1f} px  (mean {rs.mean():.0f} px, "
          f"{100 * rs.mean() / short:.1f}% of the short side)")
    print(f"   -> to match this framing, set 机台占比 to about "
          f"{100 * 2 * rs.mean() / short:.0f}%")

    if montage_path:
        every = max(1, len(frames) // 12)
        tiles = []
        for frame, index in zip(frames[::every][:12], range(0, len(frames), every)):
            canvas = frame.copy()
            hit = next((r for r in results if r[0] == index), None)
            if hit:
                cv2.circle(canvas, (int(hit[1]), int(hit[2])), int(hit[3]),
                           (0, 255, 0), 3)
                cv2.drawMarker(canvas, (int(width / 2), int(height / 2)),
                               (255, 210, 0), cv2.MARKER_CROSS, 40, 3)
            tile = cv2.resize(canvas, (240, int(240 * height / width)))
            tiles.append(tile)
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        width_max = max(row.shape[1] for row in rows)
        rows = [np.pad(row, ((0, 0), (0, width_max - row.shape[1]), (0, 0)))
                for row in rows]
        montage = np.vstack(rows)
        cv2.imwrite(montage_path, montage)
        print(f"   montage           {montage_path}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    montage = None
    for i, a in enumerate(sys.argv):
        if a == "--montage" and i + 1 < len(sys.argv):
            montage = sys.argv[i + 1]
    if not args:
        print(__doc__)
        sys.exit(1)

    for target in args:
        if os.path.isdir(target):
            analyse(frames_from_folder(target), os.path.basename(target),
                    montage and montage.replace(".jpg", f"-{os.path.basename(target)}.jpg"))
        elif os.path.isfile(target):
            analyse(frames_from_video(target), os.path.basename(target), montage)
        else:
            print("skip:", target)

    print()
    print("对比时看两个数：centre spread 和 drift rate。")
    print("真正的锁定镜头，两个都接近 0；被'拖回去'的画面，drift rate 会跟着你转身的速度涨。")


if __name__ == "__main__":
    main()
