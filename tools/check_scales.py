"""How small can the cabinet get before the detector stops finding it?

The model was trained on square crops where the cabinet fills the frame, but in
the field the phone sees a portrait view in which the cabinet may cover only
part of the square that Vision centre-crops out. This measures that: it shrinks
a labelled frame into a grey canvas — which is exactly what a smaller cabinet in
the same crop looks like — and reports detection rate and IoU per scale.

That number decides whether the app needs to try more than one scale.

Run with a Python that has onnxruntime:
    D:\\Python312\\python.exe tools\\check_scales.py
"""

import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_onnx as probe  # noqa: E402

SCALES = (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2)
CANVAS = 640
SCORE_THRESHOLD = 0.08


def square_crop_truth(label, width, height):
    """The label in the coordinates of the square centre crop, 0..1."""
    side = min(width, height)
    offset_x = (width - side) / 2
    offset_y = (height - side) / 2
    return ((label[0] * width - offset_x) / side,
            (label[1] * height - offset_y) / side,
            label[2] * width / side,
            label[3] * height / side)


def main():
    session = ort.InferenceSession(probe.MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    paths = sorted(glob.glob(os.path.join(probe.DATASET, "images", "val", "*.jpg")))
    if not paths:
        sys.exit("no val images")
    frames = []
    for path in paths:
        image = probe.read_image(path)
        label = probe.load_label(os.path.splitext(os.path.basename(path))[0])
        if image is None or label is None:
            continue
        h, w = image.shape[:2]
        side = min(h, w)
        crop = image[(h - side) // 2:(h - side) // 2 + side,
                     (w - side) // 2:(w - side) // 2 + side]
        frames.append((crop, square_crop_truth(label, w, h)))

    print(f"{len(frames)} frames, canvas {CANVAS}, score threshold {SCORE_THRESHOLD}")
    print()
    print(f"{'scale':>6} {'cabinet px':>11} {'found':>7} {'mean score':>11} {'mean IoU':>9}")
    for scale in SCALES:
        side = int(round(CANVAS * scale))
        offset = (CANVAS - side) // 2
        hits, scores, overlaps = 0, [], []
        for crop, truth in frames:
            canvas = np.full((CANVAS, CANVAS, 3), 114, dtype=np.uint8)
            resized = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
            canvas[offset:offset + side, offset:offset + side] = resized

            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
            output = np.asarray(session.run([output_name], {input_name: tensor})[0])
            _, box, score = probe.decode(output)
            if box is None:
                continue

            truth_box = (offset + (truth[0] - truth[2] / 2) * side,
                         offset + (truth[1] - truth[3] / 2) * side,
                         offset + (truth[0] + truth[2] / 2) * side,
                         offset + (truth[1] + truth[3] / 2) * side)
            hits += 1
            scores.append(score)
            overlaps.append(probe.iou(probe.box_of(box), truth_box))

        mean_score = np.mean(scores) if scores else 0.0
        mean_iou = np.mean(overlaps) if overlaps else 0.0
        cabinet_px = side * frames[0][1][2]
        print(f"{scale:>6.2f} {cabinet_px:>11.0f} {hits:>4}/{len(frames)} "
              f"{mean_score:>11.2f} {mean_iou:>9.2f}")


if __name__ == "__main__":
    main()
