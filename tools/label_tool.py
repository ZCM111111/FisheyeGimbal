"""Interactive labeller for the cabinet dataset.

The classical detector already proposes a box for most frames, so the work here
is confirming or nudging rather than drawing every box by hand.

Everything can be done from the keyboard, because mouse coordinates in an
OpenCV window are unreliable once the view is scaled. Drag still works and is
mapped back through the window's image rect, but it is a convenience only.

Usage:
    python tools/label_tool.py <frames folder> [--out <dataset folder>]

Keys:
    space / enter   accept the current box and go on
    arrows / wasd   move the box (hold shift for a bigger step)
    - / =           shrink / grow the box
    c               cycle the next detector candidate
    r               re-run the detector on this frame at full resolution
    n               skip this frame
    u               undo the previous label and go back
    q               save and quit

Output (default: %USERPROFILE%\\fisheye_dataset, deliberately outside the repo):
    <out>/images/*.jpg        full-size frames
    <out>/labels/*.txt        YOLO boxes: "0 cx cy w h", all normalised
    <out>/data.yaml           training description
"""

import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cabinet_detect  # noqa: E402

WINDOW = "label tool"
MAX_VIEW = 1100

# waitKeyEx reports arrows as extended codes on Windows, and as 65361+ on the
# other builds, so accept both spellings.
ARROWS = {
    2424832: "left", 65361: "left",
    2490368: "up", 65362: "up",
    2555904: "right", 65363: "right",
    2621440: "down", 65364: "down",
}

LEGEND = (
    "space=ok  arrows/wasd=move  -/=size  c=next candidate  r=redetect  n=skip  u=undo  q=quit",
)


class State:
    def __init__(self):
        self.box = None          # (x0, y0, x1, y1) in original pixels
        self.dragging = False
        self.origin = (0, 0)
        self.scale = 1.0
        self.window_rect = None  # (x, y, w, h) of the image inside the window
        self.view_width = 1
        self.view_height = 1
        self.modified = False


def read_window_rect():
    try:
        return cv2.getWindowImageRect(WINDOW)
    except Exception:
        return None


def on_mouse(event, x, y, flags, state):
    """Drag to redraw. Window coordinates are mapped back to original pixels."""
    if state.box is None or state.scale <= 0:
        return

    def to_original(px, py):
        rect = state.window_rect
        if rect and rect[2] > 0 and rect[3] > 0:
            rx, ry, rw, rh = rect
            view_w = state.view_width
            view_h = state.view_height
            px = (px - rx) * view_w / float(rw)
            py = (py - ry) * view_h / float(rh)
        return px / state.scale, py / state.scale

    if event == cv2.EVENT_LBUTTONDOWN:
        state.dragging = True
        state.origin = to_original(x, y)
        state.box = (*state.origin, *state.origin)
        state.modified = True
    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        cx, cy = to_original(x, y)
        state.box = (state.origin[0], state.origin[1], cx, cy)
        state.modified = True
    elif event == cv2.EVENT_LBUTTONUP:
        state.dragging = False
        x0, y0, x1, y1 = state.box
        state.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        state.modified = True


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    x0 = min(max(x0, 0.0), width - 2.0)
    y0 = min(max(y0, 0.0), height - 2.0)
    x1 = min(max(x1, x0 + 2.0), width - 1.0)
    y1 = min(max(y1, y0 + 2.0), height - 1.0)
    return (x0, y0, x1, y1)


def move_box(box, direction, step, width, height):
    x0, y0, x1, y1 = box
    if direction == "left":
        x0 -= step
        x1 -= step
    elif direction == "right":
        x0 += step
        x1 += step
    elif direction == "up":
        y0 -= step
        y1 -= step
    else:
        y0 += step
        y1 += step
    # Keep the size, only slide the box inside the frame.
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width - 1:
        x0 -= x1 - (width - 1)
        x1 = width - 1
    if y1 > height - 1:
        y0 -= y1 - (height - 1)
        y1 = height - 1
    return clamp_box((x0, y0, x1, y1), width, height)


def resize_box(box, delta, width, height):
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    half_w = max((box[2] - box[0]) * 0.5 + delta, 4.0)
    half_h = max((box[3] - box[1]) * 0.5 + delta, 4.0)
    return clamp_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), width, height)


def box_from_circle(cx, cy, r, width, height):
    return clamp_box((cx - r, cy - r, cx + r, cy + r), width, height)


def normalise(box, width, height):
    x0, y0, x1, y1 = box
    return ((x0 + x1) * 0.5 / width, (y0 + y1) * 0.5 / height,
            abs(x1 - x0) / width, abs(y1 - y0) / height)


def stem_for(index, path):
    """Numbered, so two folders with the same file name cannot collide."""
    return "%04d_%s" % (index, os.path.splitext(os.path.basename(path))[0])


def proposals(image):
    """Candidate circles at full resolution, best first.

    Detection runs on a downscaled copy: the app itself works on a decimated
    grid, and the full-resolution Hough transform costs seconds per frame for
    no accuracy gain.
    """
    h, w = image.shape[:2]
    scale = min(1.0, 480.0 / max(w, h))
    small = image if scale >= 1.0 else cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found = cabinet_detect.candidates(small)
    return [dict(cx=c["cx"] / scale, cy=c["cy"] / scale, r=c["r"] / scale,
                 support=c["support"]) for c in found]


def legend(canvas, lines, colour):
    height = canvas.shape[0]
    for i, text in enumerate(lines):
        y = height - 14 - 26 * (len(lines) - 1 - i)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, 1, cv2.LINE_AA)


def main():
    args = sys.argv[1:]
    sources = []
    out = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
            continue
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
            i += 1
            continue
        sources.append(a)
        i += 1

    if not sources:
        print(__doc__)
        sys.exit(1)
    if out is None:
        out = os.environ.get("USERPROFILE", os.path.expanduser("~"))

    out = os.path.join(out, "fisheye_dataset")
    images_dir = os.path.join(out, "images")
    labels_dir = os.path.join(out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Full paths, so several folders can be labelled in one session.
    files = []
    for source in sources:
        if not os.path.isdir(source):
            print("skip (not a folder):", source)
            continue
        for name in sorted(os.listdir(source)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(source, name))
    if not files:
        print("no images in", sources)
        sys.exit(1)
    print(f"{len(files)} 张待标注，输出到 {out}")
    print("空格=接受  方向键/WASD=移动  -/=缩小放大  c=换候选  r=重检测  n=跳过  u=退回  q=保存退出")
    print("标注对象：内屏那个圆（不含外键和外圈）")
    print()

    # Resume: a frame already labelled keeps its file, so restarting the tool
    # does not mean redoing the first N frames.
    pending = [i for i, path in enumerate(files)
               if not os.path.exists(os.path.join(labels_dir, stem_for(i, path) + ".txt"))]
    already = len(files) - len(pending)
    start = pending[0] if pending else 0
    if already:
        print(f"已有 {already} 张标好，从第 {start + 1} 张继续")
    if not pending:
        print("全部标完了 — 用 u 可以回头改，或直接关掉窗口")

    state = State()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    history = []
    index = start
    saved = 0
    skipped = 0

    while 0 <= index < len(files):
        path = files[index]
        name = os.path.basename(path)
        image = cabinet_detect.imread_unicode(path)
        if image is None:
            index += 1
            continue
        h, w = image.shape[:2]
        state.scale = min(1.0, MAX_VIEW / max(w, h))
        state.view = cv2.resize(image, None, fx=state.scale, fy=state.scale) \
            if state.scale < 1.0 else image.copy()
        state.view_height, state.view_width = state.view.shape[:2]
        state.modified = False

        found = proposals(image)
        chosen = 0
        if found:
            state.box = box_from_circle(found[0]["cx"], found[0]["cy"], found[0]["r"], w, h)
            print(f"[{index + 1}/{len(files)}] {name}: 候选 {len(found)} 个，"
                  f"首选支持度 {found[0]['support']:.2f}")
        else:
            # Something to nudge rather than nothing at all.
            side = min(w, h) * 0.35
            state.box = box_from_circle(w / 2, h / 2, side, w, h)
            print(f"[{index + 1}/{len(files)}] {name}: 无候选圆 — 用方向键和 -/= 自己摆")

        step = max(2, int(min(w, h) * 0.005))
        action = None
        while action is None:
            canvas = state.view.copy()
            state.window_rect = read_window_rect()

            if state.box is not None:
                x0, y0, x1, y1 = state.box
                p0 = (int(x0 * state.scale), int(y0 * state.scale))
                p1 = (int(x1 * state.scale), int(y1 * state.scale))
                if state.modified or not found:
                    colour = (255, 210, 0)      # hand-adjusted: cyan
                else:
                    colour = (0, 255, 0)        # proposal: green
                cv2.rectangle(canvas, p0, p1, colour, 2)
                cv2.circle(canvas, ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2),
                           3, colour, -1)

            lines = [f"{index + 1}/{len(files)}  {name}"]
            if found:
                lines.append(f"candidate {chosen + 1}/{len(found)}  "
                             f"support {found[chosen]['support']:.2f}")
                if state.modified:
                    lines.append("hand-adjusted")
            else:
                lines.append("no proposal - place the box yourself")
            lines.append(LEGEND[0])
            legend(canvas, lines, (255, 255, 255))

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKeyEx(20)

            code = key & 0xFF if key != -1 else -1
            direction = ARROWS.get(key)
            if direction is None and code != -1:
                letter = chr(code) if 32 <= code < 127 else ""
                direction = {"a": "left", "d": "right", "w": "up", "s": "down",
                             "A": "left", "D": "right", "W": "up", "S": "down"}.get(letter)

            if direction:
                # Lower-case wasd nudges, upper-case (shift held) jumps.
                big = 32 <= code < 127 and chr(code).isupper()
                state.box = move_box(state.box, direction, step * (5 if big else 1), w, h)
                state.modified = True
            elif key in (13, 32):                   # enter / space
                if abs(state.box[2] - state.box[0]) > 8:
                    stem = stem_for(index, path)
                    shutil.copyfile(path, os.path.join(images_dir, stem + ".jpg"))
                    label = normalise(state.box, w, h)
                    with open(os.path.join(labels_dir, stem + ".txt"), "w") as handle:
                        handle.write("0 %.6f %.6f %.6f %.6f\n" % label)
                    saved += 1
                    history.append((path, index))
                    action = "next"
                else:
                    print("  框太小了，先调大一点")
            elif code in (ord("-"), ord("_")) and state.box:
                state.box = resize_box(state.box, -step, w, h)
                state.modified = True
            elif code in (ord("="), ord("+")) and state.box:
                state.box = resize_box(state.box, step, w, h)
                state.modified = True
            elif code == ord("c") and found and state.box:
                chosen = (chosen + 1) % len(found)
                pick = found[chosen]
                state.box = box_from_circle(pick["cx"], pick["cy"], pick["r"], w, h)
                state.modified = True
                print(f"  候选 {chosen + 1}/{len(found)} 支持度 {pick['support']:.2f}")
            elif code == ord("r"):
                found = proposals(image)
                chosen = 0
                if found:
                    pick = found[0]
                    state.box = box_from_circle(pick["cx"], pick["cy"], pick["r"], w, h)
                    state.modified = False
            elif code == ord("n"):
                skipped += 1
                action = "next"
            elif code == ord("u"):
                action = "back"
            elif code == ord("q"):
                action = "quit"

        if action == "quit":
            break
        if action == "back":
            if history:
                previous_path, previous_index = history.pop()
                stem = stem_for(previous_index, previous_path)
                for folder, suffix in ((images_dir, ".jpg"), (labels_dir, ".txt")):
                    candidate = os.path.join(folder, stem + suffix)
                    if os.path.exists(candidate):
                        os.remove(candidate)
                saved = max(saved - 1, 0)
                index = previous_index
                continue
            index = max(index - 1, 0)
            continue
        index += 1

    cv2.destroyAllWindows()

    with open(os.path.join(out, "data.yaml"), "w") as handle:
        handle.write("path: %s\n" % out.replace("\\", "/"))
        handle.write("train: images\nval: images\n")
        handle.write("names:\n  0: screen\n")

    print()
    total = len([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
    print(f"本次新增 {saved} 张，跳过 {skipped} 张")
    print(f"数据集共 {total} 张：{out}")
    print("现在可以训练:")
    print(f"  pip install ultralytics")
    print(f"  yolo detect train data={os.path.join(out, 'data.yaml')} model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
