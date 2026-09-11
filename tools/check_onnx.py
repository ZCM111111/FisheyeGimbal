"""Check the exported ONNX detector: graph, layout, and decoded box.

The Swift side in ScreenDetector.swift decodes `output0` by hand and the CoreML
conversion has to declare the same preprocessing, so both the tensor layout and
the expected input scaling have to be confirmed against the real file rather
than assumed. This sweeps the plausible combinations and reports how well the
decoded box matches the label — a wrong channel order or a double-normalised
input both show up here as boxes in absurd places.

Run with a Python that has onnxruntime (D:\\Python312 works):
    D:\\Python312\\python.exe tools\\check_onnx.py
"""

import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "screen.onnx")
DATASET = os.path.join(os.environ.get("USERPROFILE", ""), "fisheye_yolo")
INPUT_SIDE = 640
# This model is a weak classifier on confident-looking boxes: on the val split
# its correct detections score 0.13-0.41, so a 0.5-style threshold looks like
# "nothing detected" while a saturated, wrong input looks like success. Verified
# against ultralytics' own ONNX predictor, which reports IoU 0.82-0.95 on these
# same frames at those scores.
SCORE_THRESHOLD = 0.08


def read_image(path):
    # cv2.imread cannot open non-ASCII Windows paths.
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_label(stem):
    path = os.path.join(DATASET, "labels", "val", stem + ".txt")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        parts = [float(v) for v in handle.read().split()]
    return parts[1], parts[2], parts[3], parts[4]


def make_tensor(image, scale, channels, fit):
    """Two ways to hand an image to the model:

    "crop"  centre-crop to a square then resize — Vision's .centerCrop.
    "letterbox" scale to fit, pad the short side with grey — how ultralytics
           feeds frames at training and validation time.
    """
    h, w = image.shape[:2]
    if fit == "crop":
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        prepared = image[y0:y0 + side, x0:x0 + side]
        prepared = cv2.resize(prepared, (INPUT_SIDE, INPUT_SIDE),
                              interpolation=cv2.INTER_LINEAR)
    else:
        ratio = min(INPUT_SIDE / w, INPUT_SIDE / h)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        prepared = np.full((INPUT_SIDE, INPUT_SIDE, 3), 114, dtype=np.uint8)
        top, left = (INPUT_SIDE - new_h) // 2, (INPUT_SIDE - new_w) // 2
        prepared[top:top + new_h, left:left + new_w] = resized
    if channels == "rgb":
        prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    tensor = prepared.astype(np.float32) * scale
    return np.transpose(tensor, (2, 0, 1))[None, ...]


def decode(output):
    """Same walk as the Swift decoder: attributes are channels, anchors are the
    other axis — cx, cy, w, h then one score per class."""
    shape = output.shape
    if not (len(shape) == 3 and shape[0] == 1):
        raise SystemExit(f"unexpected output shape {shape}")
    attributes = 5
    if shape[1] == attributes:
        channels_first, anchors = True, shape[2]
    elif shape[2] == attributes:
        channels_first, anchors = False, shape[1]
    else:
        raise SystemExit(f"unexpected output shape {shape}")

    def value(attribute, anchor):
        return output[0, attribute, anchor] if channels_first else output[0, anchor, attribute]

    best, best_score = None, SCORE_THRESHOLD
    for anchor in range(anchors):
        score = float(value(4, anchor))
        if score <= best_score:
            continue
        cx, cy, w, h = (float(value(i, anchor)) for i in range(4))
        if w <= 1 or h <= 1:
            continue
        best = (cx, cy, w, h)
        best_score = score
    return channels_first, best, best_score


def label_in_crop(label, width, height, fit):
    """The label in the space the tensor lives in."""
    if fit == "crop":
        side = min(width, height)
        cx = (label[0] * width - (width - side) / 2) / side * INPUT_SIDE
        cy = (label[1] * height - (height - side) / 2) / side * INPUT_SIDE
        return cx, cy, label[2] * width / side * INPUT_SIDE, label[3] * height / side * INPUT_SIDE

    ratio = min(INPUT_SIDE / width, INPUT_SIDE / height)
    offset_x = (INPUT_SIDE - width * ratio) / 2
    offset_y = (INPUT_SIDE - height * ratio) / 2
    return (label[0] * width * ratio + offset_x,
            label[1] * height * ratio + offset_y,
            label[2] * width * ratio,
            label[3] * height * ratio)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def box_of(box):
    cx, cy, w, h = box
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def main():
    if not os.path.exists(MODEL):
        sys.exit("missing " + MODEL)
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("input ", input_name, session.get_inputs()[0].shape)
    print("output", output_name, session.get_outputs()[0].shape)

    paths = sorted(glob.glob(os.path.join(DATASET, "images", "val", "*.jpg")))
    if not paths:
        sys.exit("no val images in " + DATASET)

    first_channels_first = None
    for fit in ("crop", "letterbox"):
        for channels in ("rgb", "bgr"):
            for scale_label, scale in (("0-255", 1.0), ("0-1", 1.0 / 255.0)):
                hits, scores, overlaps = 0, [], []
                for path in paths:
                    image = read_image(path)
                    tensor = make_tensor(image, scale, channels, fit)
                    output = np.asarray(session.run([output_name], {input_name: tensor})[0])
                    channels_first, box, score = decode(output)
                    if first_channels_first is None:
                        first_channels_first = channels_first
                    if box is None:
                        continue
                    label = load_label(os.path.splitext(os.path.basename(path))[0])
                    if label is None:
                        continue
                    h, w = image.shape[:2]
                    truth = label_in_crop(label, w, h, fit)
                    hits += 1
                    scores.append(score)
                    overlaps.append(iou(box_of(box), box_of(truth)))
                mean_iou = sum(overlaps) / len(overlaps) if overlaps else 0.0
                mean_score = sum(scores) / len(scores) if scores else 0.0
                print(f"{fit:10s} {channels.upper():4s} {scale_label:6s} -> "
                      f"{hits}/{len(paths)} found  mean score {mean_score:.2f}  "
                      f"mean IoU vs label {mean_iou:.2f}")

    print()
    print(f"output layout is {'channels-first (1, 5, N)' if first_channels_first else 'anchors-first (1, N, 5)'}")
    print("the winning line is what ScreenDetector.swift and the CoreML conversion must mirror")


if __name__ == "__main__":
    main()
