"""Interactive labeller for the cabinet dataset.

The classical detector already proposes a box for most frames, so the work here
is confirming or nudging rather than drawing every box by hand. The proposal is
shown in green; press space to accept it, or drag to redraw.

Usage:
    python tools/label_tool.py <frames folder> [--out <dataset folder>]

Keys:
    space / enter   accept the current box
    drag            draw a new box (press and drag)
    r               re-run the detector on this frame
    n               skip this frame
    u               back to the previous frame
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

WINDOW = "label - space=accept  drag=redraw  r=redetect  n=skip  u=back  q=quit"
MAX_VIEW = 1100


class State:
    def __init__(self):
        self.box = None          # (x0, y0, x1, y1) in original pixels
        self.dragging = False
        self.origin = (0, 0)
        self.scale = 1.0
        self.frame = None
        self.view = None


def to_view(point, state):
    return int(point[0] * state.scale), int(point[1] * state.scale)


def to_original(point, state):
    return point[0] / state.scale, point[1] / state.scale


def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.dragging = True
        state.origin = (x, y)
        state.box = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.dragging:
        state.box = (state.origin[0], state.origin[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        state.dragging = False
        x0, y0, x1, y1 = state.box
        state.box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def normalise(box, width, height):
    x0, y0, x1, y1 = box
    return ((x0 + x1) * 0.5 / width, (y0 + y1) * 0.5 / height,
            abs(x1 - x0) / width, abs(y1 - y0) / height)


def detect_small(image):
    """Detects on a downscaled copy and scales the result back.

    The app itself works on a decimated grid, and running the detector at full
    resolution costs seconds per frame for no accuracy gain.
    """
    h, w = image.shape[:2]
    scale = min(1.0, 480.0 / max(w, h))
    if scale >= 1.0:
        return cabinet_detect.detect(image)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = cabinet_detect.detect(small)
    if not result.get("ok"):
        return result
    scaled = dict(result)
    scaled["cx"] = result["cx"] / scale
    scaled["cy"] = result["cy"] / scale
    scaled["r"] = result["r"] / scale
    return scaled


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    folders = [a for a in sys.argv[1:] if not a.startswith("--")]
    args = [a for a in sys.argv[1:] if a.startswith("--")]
    source = folders[0]
    out = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    for i, a in enumerate(args):
        if a == "--out" and i + 1 < len(folders):
            out = folders[i + 1]
        elif a.startswith("--out="):
            out = a.split("=", 1)[1]
    out = os.path.join(out, "fisheye_dataset")
    images_dir = os.path.join(out, "images")
    labels_dir = os.path.join(out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(source)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        print("no images in", source)
        sys.exit(1)

    state = State()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    history = []
    index = 0
    saved = 0
    skipped = 0

    while 0 <= index < len(files):
        name = files[index]
        image = cabinet_detect.imread_unicode(os.path.join(source, name))
        if image is None:
            index += 1
            continue
        h, w = image.shape[:2]
        state.scale = min(1.0, MAX_VIEW / max(w, h))
        state.view = cv2.resize(image, None, fx=state.scale, fy=state.scale) \
            if state.scale < 1.0 else image.copy()
        state.frame = image
        state.box = None

        result = detect_small(image)
        if result.get("ok"):
            cx, cy, r = result["cx"], result["cy"], result["r"]
            state.box = (max(cx - r, 0), max(cy - r, 0),
                         min(cx + r, w - 1), min(cy + r, h - 1))
            print(f"[{index + 1}/{len(files)}] {name}: 提议支持度 {result['support']:.2f}")
        else:
            print(f"[{index + 1}/{len(files)}] {name}: {result.get('why', '无提议')} — 请手动画框")

        action = None
        while action is None:
            canvas = state.view.copy()
            if state.box is not None:
                x0, y0 = to_view((state.box[0], state.box[1]), state)
                x1, y1 = to_view((state.box[2], state.box[3]), state)
                colour = (0, 255, 0) if result.get("ok") else (0, 165, 255)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
                cv2.putText(canvas, "accept with space" if result.get("ok") else "drag a box",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "drag a box around the screen", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):                     # enter / space
                if state.box and abs(state.box[2] - state.box[0]) > 8:
                    shutil.copyfile(os.path.join(source, name),
                                    os.path.join(images_dir, name))
                    label = normalise(state.box, w, h)
                    with open(os.path.join(labels_dir,
                                           os.path.splitext(name)[0] + ".txt"), "w") as handle:
                        handle.write("0 %.6f %.6f %.6f %.6f\n" % label)
                    saved += 1
                    history.append(name)
                    action = "next"
                else:
                    print("  box too small — drag one first")
            elif key == ord("r"):
                result = cabinet_detect.detect(image)
                if result.get("ok"):
                    cx, cy, r = result["cx"], result["cy"], result["r"]
                    state.box = (max(cx - r, 0), max(cy - r, 0),
                                 min(cx + r, w - 1), min(cy + r, h - 1))
            elif key == ord("n"):
                skipped += 1
                action = "next"
            elif key == ord("u"):
                action = "back"
            elif key == ord("q"):
                action = "quit"

        if action == "quit":
            break
        if action == "back":
            if history:
                previous = history.pop()
                for folder in (images_dir, labels_dir):
                    candidate = os.path.join(folder, previous)
                    if os.path.exists(candidate):
                        os.remove(candidate)
                    stem = os.path.splitext(previous)[0]
                    candidate = os.path.join(folder, stem + ".txt")
                    if os.path.exists(candidate):
                        os.remove(candidate)
                saved = max(saved - 1, 0)
                index = files.index(previous)
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
    print(f"已标注 {saved} 张，跳过 {skipped} 张")
    print("数据集:", out)
    print("现在可以训练:")
    print(f"  pip install ultralytics")
    print(f"  yolo detect train data={os.path.join(out, 'data.yaml')} model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
